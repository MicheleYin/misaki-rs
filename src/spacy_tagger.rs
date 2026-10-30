//! spaCy `en_core_web_sm` POS tagger, ported so tags match misaki's exactly.
//!
//! misaki tags with spaCy and its pronunciation is POS-conditional: 790 of the
//! 90,201 gold entries are tag-keyed, and `get_special_case` branches on the
//! tag directly (`A` as `DT` is `ɐ`, as `NNP` it is `ˈA`). An approximate
//! tagger therefore caps achievable parity -- measured, the perceptron tagger
//! agreed with spaCy on ~89% of tokens and accounted for 64 of the 73
//! wrong-phoneme tokens remaining in the prose corpus.
//!
//! The model is small: MultiHashEmbed + 4x residual MaxoutWindowEncoder +
//! softmax over 50 tags, 1,570,514 f32 (6.3 MB) in `data/tagger.bin`, written
//! by `tools/extract_tagger.py`. Every operation below was first reproduced in
//! numpy and checked against spaCy's own output before being written here.

const NO: usize = 96; // tok2vec width
const NP: usize = 3; // maxout pieces
const NTAGS: usize = 50;
/// Thinc's LayerNorm adds this to the variance (`var + 1e-08`), not 1e-5.
const EPS: f32 = 1e-8;
/// `with_array(pad=4)` wraps the encoder input in four zero rows each side.
const PAD: usize = 4;

/// `nV` per hash-embedding table, in column order NORM/PREFIX/SUFFIX/SHAPE/
/// SPACY/IS_SPACE. Asserted against the live model by the extractor.
const NV: [usize; 6] = [5000, 1000, 2500, 2500, 50, 50];
/// Per-column hash seeds, likewise asserted by the extractor.
const SEEDS: [u32; 6] = [8, 9, 10, 11, 12, 13];

/// Thinc's `MurmurHash3_x86_128_uint64` (`thinc/backends/numpy_ops.pyx`).
///
/// NOT generic MurmurHash3 over bytes -- Thinc hand-specialises it for a single
/// u64, and a stock murmur3 crate does not reproduce it. Verified against
/// `NumpyOps.hash` for every seed 8..13 over inputs including 0, 1 and 2^63.
#[inline]
fn thinc_hash(val: u64, seed: u32) -> [u32; 4] {
    let mut h1 = val
        .wrapping_mul(0x87c3_7b91_1142_53d5)
        .rotate_left(31)
        .wrapping_mul(0x4cf5_ad43_2745_937f);
    h1 ^= seed as u64;
    h1 ^= 8;
    let mut h2 = (seed as u64) ^ 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);

    h1 ^= h1 >> 33;
    h1 = h1.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h1 ^= h1 >> 33;
    h1 = h1.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h1 ^= h1 >> 33;

    h2 ^= h2 >> 33;
    h2 = h2.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h2 ^= h2 >> 33;
    h2 = h2.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h2 ^= h2 >> 33;

    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as u32, (h1 >> 32) as u32, h2 as u32, (h2 >> 32) as u32]
}

/// spaCy's StringStore hash: MurmurHash64A over the UTF-8 bytes, seed 1.
/// Verified against `spacy.strings.hash_string` on ASCII and non-ASCII input.
pub fn hash_string(s: &str) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    const R: u32 = 47;
    let data = s.as_bytes();
    let mut h: u64 = 1u64 ^ (data.len() as u64).wrapping_mul(M);
    let nblocks = data.len() / 8;
    for i in 0..nblocks {
        let mut k = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
    let rem = &data[nblocks * 8..];
    if !rem.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rem.len()].copy_from_slice(rem);
        h ^= u64::from_le_bytes(buf);
        h = h.wrapping_mul(M);
    }
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// spaCy's orthographic shape: each character becomes `X` (upper), `x`
/// (lower), `d` (digit) or itself, and a run of the same class is capped at
/// four. `barked` -> `xxxx`, `Smith` -> `Xxxxx`, `Mr.` -> `Xx.`, `n't` -> `x'x`.
fn shape_of(text: &str) -> String {
    // spaCy's `word_shape` (`spacy/lang/lex_attrs.py`). Two details that are
    // easy to get wrong and both were:
    //
    //   * `char.isalpha()` is tested FIRST, and only then upper vs lower. A
    //     caseless alphabetic character is therefore `x`, not itself -- `ˈ`
    //     (MODIFIER LETTER VERTICAL LINE) is alpha, neither upper nor lower, so
    //     spaCy shapes it `x`. Checking case first kept it literally and cost
    //     510 feature mismatches across the corpus.
    //   * a text of 100 characters or more shapes to the literal `LONG`.
    if text.chars().count() >= 100 {
        return "LONG".to_string();
    }
    let mut out = String::new();
    let mut last = '\0';
    let mut run = 0usize;
    for c in text.chars() {
        let m = if c.is_alphabetic() {
            if c.is_uppercase() {
                'X'
            } else {
                'x'
            }
        } else if c.is_numeric() {
            'd'
        } else {
            c
        };
        if m == last {
            run += 1;
        } else {
            run = 1;
            last = m;
        }
        if run <= 4 {
            out.push(m);
        }
    }
    out
}

/// The six feature columns spaCy's tok2vec consumes, as StringStore hashes for
/// the first four and booleans for the last two.
fn features(text: &str, norm: &str, trailing_space: bool) -> [u64; 6] {
    let prefix: String = text.chars().take(1).collect();
    let n = text.chars().count();
    let suffix: String = text.chars().skip(n.saturating_sub(3)).collect();
    // Not `hash_string` directly: spaCy's StringStore returns a fixed small ID
    // for any of its 457 known symbols and only hashes the rest. The shape `X`
    // is symbol 101, so a one-character uppercase token's SHAPE is 101 rather
    // than its hash; hashing it anyway shifted the feature on every such token.
    let sid = |s: &str| crate::symbols::string_id(s, hash_string);
    [
        sid(norm),
        sid(&prefix),
        sid(&suffix),
        sid(&shape_of(text)),
        u64::from(trailing_space),
        u64::from(!text.is_empty() && text.chars().all(char::is_whitespace)),
    ]
}

struct Maxout {
    w: Vec<f32>, // (NO, NP, nI)
    b: Vec<f32>, // (NO*NP)
    ni: usize,
}

struct LayerNorm {
    g: Vec<f32>,
    b: Vec<f32>,
}

/// The loaded model. `Tagger::load` reads `data/tagger.bin` positionally in the
/// order `tools/extract_tagger.py` writes it.
pub struct Tagger {
    e: Vec<Vec<f32>>, // 6 tables, each NV[i] * NO
    mx: Vec<Maxout>,  // 5
    ln: Vec<LayerNorm>,
    sm_w: Vec<f32>, // (NTAGS, NO)
    sm_b: Vec<f32>,
    pub labels: Vec<String>,
}

impl Tagger {
    pub fn load() -> Self {
        let blob: &[u8] = include_bytes!("../data/tagger.bin");
        let labels_txt: &str = include_str!("../data/tagger_labels.txt");
        let f32s: Vec<f32> = blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut at = 0usize;
        let mut take = |n: usize| -> Vec<f32> {
            let v = f32s[at..at + n].to_vec();
            at += n;
            v
        };

        let e: Vec<Vec<f32>> = (0..6).map(|i| take(NV[i] * NO)).collect();
        let mut mx = Vec::with_capacity(5);
        let mut ln = Vec::with_capacity(5);
        for i in 0..5 {
            let ni = if i == 0 { NO * 6 } else { NO * 3 };
            let w = take(NO * NP * ni);
            let b = take(NO * NP);
            mx.push(Maxout { w, b, ni });
            let g = take(NO);
            let lb = take(NO);
            ln.push(LayerNorm { g, b: lb });
        }
        let sm_w = take(NTAGS * NO);
        let sm_b = take(NTAGS);
        debug_assert_eq!(at, f32s.len(), "tagger.bin size disagrees with the layout");

        Tagger {
            e,
            mx,
            ln,
            sm_w,
            sm_b,
            labels: labels_txt.lines().map(|s| s.to_string()).collect(),
        }
    }

    fn embed(&self, feats: &[[u64; 6]]) -> Vec<f32> {
        let n = feats.len();
        let mut x = vec![0f32; n * NO * 6];
        for (t, f) in feats.iter().enumerate() {
            for c in 0..6 {
                // HashEmbed: four keys, taken % nV, their rows SUMMED
                // (`gather_add`).
                let keys = thinc_hash(f[c], SEEDS[c]);
                let base = t * NO * 6 + c * NO;
                for k in keys {
                    let row = (k as usize % NV[c]) * NO;
                    for j in 0..NO {
                        x[base + j] += self.e[c][row + j];
                    }
                }
            }
        }
        x
    }

    fn maxout(&self, i: usize, x: &[f32], n: usize) -> Vec<f32> {
        let m = &self.mx[i];
        let mut out = vec![0f32; n * NO];
        for t in 0..n {
            let xr = &x[t * m.ni..(t + 1) * m.ni];
            for o in 0..NO {
                let mut best = f32::NEG_INFINITY;
                for p in 0..NP {
                    let wr = &m.w[(o * NP + p) * m.ni..(o * NP + p + 1) * m.ni];
                    let mut acc = m.b[o * NP + p];
                    for k in 0..m.ni {
                        acc += xr[k] * wr[k];
                    }
                    if acc > best {
                        best = acc;
                    }
                }
                out[t * NO + o] = best;
            }
        }
        out
    }

    fn layernorm(&self, i: usize, x: &mut [f32], n: usize) {
        let l = &self.ln[i];
        for t in 0..n {
            let row = &mut x[t * NO..(t + 1) * NO];
            let mean = row.iter().sum::<f32>() / NO as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / NO as f32;
            let inv = 1.0 / (var + EPS).sqrt();
            for j in 0..NO {
                row[j] = (row[j] - mean) * inv * l.g[j] + l.b[j];
            }
        }
    }

    /// `expand_window(1)`: each position becomes `[x[t-1], x[t], x[t+1]]`, with
    /// zeros off the ends.
    fn expand_window(x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n * NO * 3];
        for t in 0..n {
            let dst = t * NO * 3;
            if t > 0 {
                out[dst..dst + NO].copy_from_slice(&x[(t - 1) * NO..t * NO]);
            }
            out[dst + NO..dst + 2 * NO].copy_from_slice(&x[t * NO..(t + 1) * NO]);
            if t + 1 < n {
                out[dst + 2 * NO..dst + 3 * NO].copy_from_slice(&x[(t + 1) * NO..(t + 2) * NO]);
            }
        }
        out
    }

    /// Tag a whole sentence. Context matters -- the encoder has a receptive
    /// field of four windows -- so tokens must be passed together, in order.
    ///
    /// `tokens` is `(text, norm, has_trailing_space)`. `norm` is normally the
    /// lowercased text; spaCy's tokenizer exceptions override it (`n't` ->
    /// `not`), which the caller supplies.
    pub fn tag(&self, tokens: &[(String, String, bool)]) -> Vec<&str> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        let feats: Vec<[u64; 6]> = tokens
            .iter()
            .map(|(t, nm, sp)| features(t, nm, *sp))
            .collect();

        let x0 = self.embed(&feats);
        let mut emb = self.maxout(0, &x0, n);
        self.layernorm(0, &mut emb, n);

        // `with_array(pad=4)`: the encoder runs on the sequence wrapped in FOUR
        // zero rows on each side, which are stripped afterwards
        // (`thinc/layers/with_array.py::_list_forward` -> `ops.flatten(Xs, pad=4)`).
        //
        // This is NOT the same as zero-padding inside expand_window. The pad
        // rows themselves pass through maxout and layernorm, so from the second
        // block onward a boundary token's neighbour is the layernorm bias, not
        // zero. Omitting it put every sentence-edge token in the wrong context
        // and held tag agreement at 99.0% -- it is why `Shhh!` tagged NNP
        // instead of UH and `!` tagged NN instead of `.`.
        let m = n + 2 * PAD;
        let mut x = vec![0f32; m * NO];
        x[PAD * NO..(PAD + n) * NO].copy_from_slice(&emb);

        for i in 1..5 {
            let w = Self::expand_window(&x, m);
            let mut y = self.maxout(i, &w, m);
            self.layernorm(i, &mut y, m);
            for j in 0..m * NO {
                x[j] += y[j];
            }
        }
        let x = &x[PAD * NO..(PAD + n) * NO];

        (0..n)
            .map(|t| {
                let row = &x[t * NO..(t + 1) * NO];
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for k in 0..NTAGS {
                    let mut acc = self.sm_b[k];
                    for j in 0..NO {
                        acc += row[j] * self.sm_w[k * NO + j];
                    }
                    if acc > bv {
                        bv = acc;
                        best = k;
                    }
                }
                self.labels[best].as_str()
            })
            .collect()
    }
}

/// Test hook: the six feature columns for one token, so a parity test can
/// attribute a tag mismatch to the features or to the network.
pub fn debug_features(text: &str, norm: &str, trailing_space: bool) -> [u64; 6] {
    features(text, norm, trailing_space)
}
