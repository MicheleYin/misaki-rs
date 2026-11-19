use crate::fallback::{Fallback, FallbackError};
use crate::language::Language;
#[cfg(feature = "espeak")]
use crate::fallback::EspeakFallback;
use crate::languages::{LanguageRules, english::English};
use crate::lexicon::Lexicon;
use thiserror::Error;
use crate::token::MToken;
use fancy_regex::Regex as FancyRegex;
use num2words::Num2Words;
use regex::Regex;
use std::collections::HashMap;

#[derive(Error, Debug)]
pub enum G2PError {
    #[error("fallback error: {0}")]
    Fallback(#[from] FallbackError),
}

/// A per-word override parsed from `[text](feature)` markdown-link syntax.
///
/// Matches the set of features accepted by Python misaki's
/// `Lexicon.preprocess` in `misaki/en.py`. The four shapes are:
///
/// * `[hello](+2)` / `[hello](-1)` / `[hello](2)` — integer stress
/// * `[hello](0.5)` / `(+0.5)` / `(-0.5)` — half-stress (these literals only)
/// * `[xyzzy](/həˈloʊ/)` — direct phoneme override
/// * `[5](#cardinal#)` — number-formatting flags
///
/// Anything else (including `=` or plain words) is dropped silently, matching
/// the upstream `else: f = None` branch.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkFeature {
    Stress(f64),
    Phonemes(String),
    NumFlags(String),
}

/// A parsed `[text](feature)` marker, anchored to byte offsets in the
/// stripped text returned by [`G2P::preprocess_links`].
#[derive(Debug, Clone)]
pub struct LinkMarker {
    pub byte_start: usize,
    pub byte_end: usize,
    pub feature: LinkFeature,
}

fn parse_link_feature(raw: &str) -> Option<LinkFeature> {
    // Mirrors the feature-parsing ladder in misaki/en.py::Lexicon.preprocess.
    // No trim — the inner text is consumed verbatim.
    if raw.is_empty() {
        return None;
    }

    // Integer stress: optional leading + or -, then all-ASCII-digit body.
    // Python: `is_digit(f[1 if f[:1] in ('-','+') else 0:])`.
    let body = match raw.as_bytes()[0] {
        b'+' | b'-' => &raw[1..],
        _ => raw,
    };
    if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(n) = raw.parse::<i32>() {
            return Some(LinkFeature::Stress(n as f64));
        }
    }

    // Half stress: only the three literals Python recognises.
    match raw {
        "0.5" | "+0.5" => return Some(LinkFeature::Stress(0.5)),
        "-0.5" => return Some(LinkFeature::Stress(-0.5)),
        _ => {}
    }

    // /phonemes/ — Python strips one leading slash and all trailing slashes
    // (via `f[1:].rstrip('/')` at parse and `v.lstrip('/')` at apply time).
    if raw.len() > 1 && raw.starts_with('/') && raw.ends_with('/') {
        let inner = raw[1..].trim_end_matches('/');
        return Some(LinkFeature::Phonemes(inner.to_string()));
    }

    // #num_flags# — same lstrip/rstrip semantics as the phoneme branch.
    if raw.len() > 1 && raw.starts_with('#') && raw.ends_with('#') {
        let inner = raw[1..].trim_end_matches('#');
        return Some(LinkFeature::NumFlags(inner.to_string()));
    }

    None
}

pub struct G2P {
    pub lexicon: Lexicon,
    pub unk: String,
    subtoken_regex: FancyRegex,
    link_regex: Regex,
    spacy_tagger: crate::spacy_tagger::Tagger,
    spacy_tokenizer: crate::spacy_tokenizer::SpacyTokenizer,
    rules: Box<dyn LanguageRules>,
    fallback: Option<Box<dyn Fallback>>,
}

impl G2P {
    pub fn new(lang: Language) -> Self {
        // misaki's SUBTOKEN_REGEX (`en.py:55`), complete and in order, since
        // regex alternation takes the first match:
        //
        //   ^['‘’]+
        //   | \p{Lu}(?=\p{Lu}\p{Ll})
        //   | (?:^-)?(?:\d?[,.]?\d)+
        //   | [-_]+
        //   | ['‘’]{2,}
        //   | \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})
        //   | \p{L}+(?:['‘’]\p{L})*
        //   | [^-_\p{L}'‘’\d]
        //   | ['‘’]+$
        //
        // The two alternatives carrying lookahead are what split camel case,
        // and they were missing: `iPhone` stayed whole, missed the lexicon and
        // resolved to nothing, where misaki splits it `i` + `Phone` and gets
        // ˈIfˌOn. fancy_regex supplies the lookahead the regex crate lacks.
        //
        // Two bugs in the character classes are fixed here too. The
        // word-internal apostrophe class held three copies of the STRAIGHT
        // apostrophe where misaki has straight plus both curly, and the
        // catch-all excluded only the straight one -- so a curly right
        // apostrophe became its own token and every curly possessive and
        // contraction in real prose came apart.
        let subtoken_regex = FancyRegex::new(concat!(
            r"^['‘’]+",
            r"|\p{Lu}(?=\p{Lu}\p{Ll})",
            r"|(?:^-)?(?:\d?[,.]?\d)+",
            r"|[-_]+",
            r"|['‘’]{2,}",
            r"|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})",
            r"|\p{L}+(?:['‘’]\p{L})*",
            r"|[^-_\p{L}'‘’\d]",
            r"|['‘’]+$",
        ))
        .unwrap();

        let rules: Box<dyn LanguageRules> = match lang {
            Language::EnglishUS | Language::EnglishGB => Box::new(English),
            // Language::Italian => Box::new(Italian),
        };

        #[cfg(feature = "espeak")]
        let fallback: Option<Box<dyn Fallback>> = match EspeakFallback::new(lang == Language::EnglishGB) {
            Ok(fb) => Some(Box::new(fb)),
            Err(e) => {
                log::warn!("espeak-ng fallback unavailable: {}", e);
                None
            }
        };
        #[cfg(not(feature = "espeak"))]
        let fallback: Option<Box<dyn Fallback>> = None;

        // Mirrors `LINK_REGEX = re.compile(r'\[([^\]]+)\]\(([^\)]*)\)')`
        // from misaki/en.py.
        let link_regex = Regex::new(r"\[([^\]]+)\]\(([^\)]*)\)").unwrap();

        Self {
            lexicon: Lexicon::new(lang),
            unk: "❓".to_string(),
            subtoken_regex,
            link_regex,
            spacy_tagger: crate::spacy_tagger::Tagger::load(),
            spacy_tokenizer: crate::spacy_tokenizer::SpacyTokenizer::new(),
            rules,
            fallback,
        }
    }

    pub fn with_fallback(lang: Language, fallback: Option<Box<dyn Fallback>>) -> Self {
        let mut g2p = Self::new(lang);
        g2p.fallback = fallback;
        g2p
    }

    /// Legacy preprocess shape, kept for API compatibility.
    ///
    /// The third return value (token-index → feature) is intentionally empty
    /// — markers are now exposed via [`G2P::preprocess_links`] using byte
    /// offsets, which compose with our subtokenizer (Python misaki keys
    /// features by spaCy-tokenized index; we don't have spaCy).
    pub fn preprocess(&self, text: &str) -> (String, Vec<String>, HashMap<usize, String>) {
        let (stripped, _) = self.preprocess_links(text);
        let tokens: Vec<String> = stripped.split_whitespace().map(|s| s.to_string()).collect();
        (stripped, tokens, HashMap::new())
    }

    /// Parse `[text](feature)` markers from `text` and return the stripped
    /// text along with markers anchored to byte ranges in the stripped text.
    ///
    /// Mirrors the parsing half of `Lexicon.preprocess` in misaki/en.py.
    pub fn preprocess_links(&self, text: &str) -> (String, Vec<LinkMarker>) {
        let mut stripped = String::with_capacity(text.len());
        let mut markers: Vec<LinkMarker> = Vec::new();
        let mut last_end = 0usize;
        for caps in self.link_regex.captures_iter(text) {
            let whole = caps.get(0).unwrap();
            let inner = caps.get(1).unwrap().as_str();
            let feat_str = caps.get(2).unwrap().as_str();
            stripped.push_str(&text[last_end..whole.start()]);
            let start = stripped.len();
            stripped.push_str(inner);
            let end = stripped.len();
            if let Some(feature) = parse_link_feature(feat_str) {
                markers.push(LinkMarker {
                    byte_start: start,
                    byte_end: end,
                    feature,
                });
            }
            last_end = whole.end();
        }
        stripped.push_str(&text[last_end..]);
        (stripped, markers)
    }

    /// misaki's punctuation test (`retokenize`):
    ///
    /// ```text
    ///     elif tk.tag in PUNCT_TAGS and not all(97 <= ord(c.lower()) <= 122 for c in tk.text):
    /// ```
    ///
    /// with PUNCT_TAGS = {".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":",
    /// "$", "#", "NFP"}. Such a token gets its phonemes in `retokenize` and is
    /// therefore appended STANDALONE, never merged into a neighbouring word.
    ///
    /// This is the rule, and no character test substitutes for it: spaCy tags
    /// the `_` in `_like` as UH and the one in `_it` as NFP, so misaki merges
    /// the first and not the second from structurally identical input. Likewise
    /// `dogs'` has tag POS and merges, while `Thomas’` has `''` and does not.
    fn is_punct_token(tk: &MToken) -> bool {
        const PUNCT_TAGS: [&str; 11] = [
            ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP",
        ];
        if !PUNCT_TAGS.contains(&tk.tag.as_str()) {
            return false;
        }
        // ... and not entirely a-z, so a word that happens to carry a
        // punctuation tag is still a word.
        !tk.text.chars().all(|c| {
            let l = c.to_lowercase().next().unwrap_or(c);
            ('a'..='z').contains(&l)
        })
    }

    /// Port of misaki's module-level `merge_tokens` (`en.py`).
    ///
    /// Collapses a run of adjacent subtokens into the single word misaki would
    /// have produced. The text is rebuilt with the interior whitespace put
    /// back, the group inherits the LAST token's trailing whitespace, and the
    /// tag comes from the "heaviest" member -- misaki scores each token's text
    /// as `sum(1 if c == c.lower() else 2 for c in text)` and takes the
    /// maximum, so `five-storied` takes `storied`'s JJ (7 beats 4 and 1) and
    /// `It’s` takes `It`'s PRP (3 beats 2).
    fn merge_text_tag(group: &[MToken]) -> (String, String, String) {
        let mut text = String::new();
        for tk in &group[..group.len() - 1] {
            text.push_str(&tk.text);
            text.push_str(&tk.whitespace);
        }
        text.push_str(&group[group.len() - 1].text);

        let weight = |t: &str| -> usize {
            t.chars()
                .map(|c| if c.to_lowercase().next() == Some(c) { 1 } else { 2 })
                .sum()
        };
        // Python `max` keeps the FIRST maximum; Rust `max_by_key` keeps the last.
        let tag = {
            let mut best: Option<(&MToken, usize)> = None;
            for tk in group {
                let w = weight(&tk.text);
                match best {
                    Some((_, bw)) if w <= bw => {}
                    _ => best = Some((tk, w)),
                }
            }
            best.map(|(tk, _)| tk.tag.clone()).unwrap_or_default()
        };

        let ws = group[group.len() - 1].whitespace.clone();
        (text, tag, ws)
    }

    /// Port of the grouping half of misaki's `G2P.retokenize` (`en.py`).
    ///
    /// misaki groups subtokens into words by ADJACENCY, not by character
    /// class: a token with no trailing whitespace opens a group, and each
    /// following token joins it while the group's last member still has no
    /// trailing whitespace.
    ///
    /// ```text
    ///     elif words and isinstance(words[-1], list) and not words[-1][-1].whitespace:
    ///         tk._.is_head = False; words[-1].append(tk)
    ///     else:
    ///         words.append(tk if tk.whitespace else [tk])
    /// ```
    ///
    /// Adjacent non-punctuation runs merge into one word (`It’s`,
    /// `five-storied`). Punctuation stands alone, matching misaki: spaCy
    /// exceptions already keep `Mr.` / `Dr.` as a single token, so a period
    /// is never rejoined into the open group (that used to eat the sentence
    /// stop after forms like `AT&T.`). Returns index ranges into `tokens`.
    fn group_by_adjacency(tokens: &[MToken]) -> Vec<std::ops::Range<usize>> {
        // misaki's FIRST branch takes precedence over adjacency:
        //
        //     if tk._.alias is not None or tk.phonemes is not None:
        //         words.append(tk)
        //
        // and `retokenize` has already given punctuation its phonemes by then,
        // so a punctuation mark is always its own word. `is_punct_token` is
        // misaki's own test, on the TAG; the character approximation that used
        // to be here could not tell `_like` from `_it`.
        let is_standalone = |tk: &MToken, _is_last: bool| -> bool {
            // misaki's first branch is `if tk._.alias is not None or
            // tk.phonemes is not None` during *retokenize*, when only links,
            // currency and punctuation have phonemes. After the lexicon pass
            // almost every token has phonemes, so we key off `rating` (set for
            // those early assignments) plus alias / punct tags.
            tk.underscore().alias.is_some()
                || tk.underscore().rating.is_some()
                || Self::is_punct_token(tk)
        };

        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        let mut open = false; // is the last group still extensible?
        for i in 0..tokens.len() {
            if is_standalone(&tokens[i], i + 1 == tokens.len()) {
                groups.push(i..i + 1);
                open = false;
                continue;
            }
            if open && tokens[groups.last().unwrap().end - 1].whitespace.is_empty() {
                let g = groups.last_mut().unwrap();
                *g = g.start..i + 1;
            } else {
                groups.push(i..i + 1);
                // A token that already carries trailing whitespace is a
                // finished word: misaki appends it bare rather than as a list,
                // and only a list can be extended.
                open = tokens[i].whitespace.is_empty();
            }
        }
        groups
    }

    /// misaki's lookup-key normalisation (`Lexicon.__call__`):
    ///
    /// ```text
    /// word = (...).replace(chr(8216), "'").replace(chr(8217), "'")
    /// word = unicodedata.normalize('NFKC', word)
    /// word = ''.join(Lexicon.numeric_if_needed(c) for c in word)
    /// ```
    fn normalize_lookup(text: &str) -> String {
        use unicode_normalization::UnicodeNormalization;
        let straightened = text.replace('\u{2018}', "'").replace('\u{2019}', "'");
        let nfkc: String = straightened.nfkc().collect();
        let mut out = String::with_capacity(nfkc.len());
        for c in nfkc.chars() {
            if let Some(d) = crate::unicode_numeric_data::numeric_if_needed(c) {
                out.push_str(d);
            } else {
                out.push(c);
            }
        }
        out
    }

    /// misaki's punctuation phonemes (`en.py:598-599`): the PUNCT_TAG_PHONEMES
    /// mapping, then keep only characters in PUNCTS.
    ///
    /// ```text
    ///     tk.phonemes = PUNCT_TAG_PHONEMES.get(
    ///         tk.tag, ''.join(c for c in tk.text if c in PUNCTS))
    /// ```
    ///
    /// with PUNCTS = ';:,.!?\u2014\u2026"\u201c\u201d' and `` -> \u201c, '' -> \u201d, -LRB- -> (,
    /// -RRB- -> ). An empty result is correct for a symbol in no class (`*`).
    fn punct_phonemes(text: &str, tag: &str) -> String {
        const PUNCTS: &str = ";:,.!?\u{2014}\u{2026}\"\u{201c}\u{201d}";
        // PUNCT_TAG_PHONEMES first, by TAG. This is keyed on the tag and not on
        // the character, which is what distinguishes an opening quote from a
        // closing one: in `'Single quotes'` spaCy tags the first apostrophe ``
        // and the second '', so misaki renders “ then ” from identical
        // characters. A straight `"` is likewise “ or ” depending on side.
        match tag {
            "-LRB-" => return "(".to_string(),
            "-RRB-" => return ")".to_string(),
            "``" => return "\u{201c}".to_string(),
            "''" | "\"\"" => return "\u{201d}".to_string(),
            _ => {}
        }
        // Otherwise keep only the characters that are in PUNCTS.
        text.chars().filter(|c| PUNCTS.contains(*c)).collect()
    }

    /// Port of misaki's `G2P.token_context` (`en.py`).
    ///
    /// Decides what the token to the LEFT should assume about what follows it:
    /// whether the next sound is a vowel (`the` -> `\u00f0i` vs `\u00f0\u0259`, `a` -> `\u0250`), and
    /// whether the next word is `to` (`to` -> `t\u028a` vs `t\u0259`). Both are decided
    /// from the RESOLVED PHONEMES, never from spelling.
    fn token_context(
        ctx: &crate::lexicon::TokenContext,
        ps: Option<&str>,
        tk: &MToken,
    ) -> crate::lexicon::TokenContext {
        const VOWELS: &str = "AIOQWYaiu\u{e6}\u{251}\u{252}\u{254}\u{259}\u{25b}\u{25c}\u{26a}\u{28a}\u{28c}\u{1d7b}";
        const CONSONANTS: &str = "bdfhjklmnpstvwz\u{f0}\u{14b}\u{261}\u{279}\u{27e}\u{283}\u{292}\u{2a4}\u{2a7}\u{3b8}";
        const NON_QUOTE_PUNCTS: &str = "!,.:;?\u{2014}\u{2026}";

        // `next(..., vowel)` in Python: if no character belongs to any of the
        // three classes, the incoming value survives unchanged.
        let mut future_vowel = ctx.future_vowel;
        if let Some(ps) = ps {
            for c in ps.chars() {
                if NON_QUOTE_PUNCTS.contains(c) {
                    future_vowel = None;
                    break;
                } else if VOWELS.contains(c) {
                    future_vowel = Some(true);
                    break;
                } else if CONSONANTS.contains(c) {
                    future_vowel = Some(false);
                    break;
                }
            }
        }

        let future_to = tk.text == "to"
            || tk.text == "To"
            || (tk.text == "TO" && (tk.tag == "TO" || tk.tag == "IN"));

        crate::lexicon::TokenContext { future_vowel, future_to }
    }

    pub fn tokenize(&self, text: &str) -> Vec<MToken> {
        self.tokenize_with_offsets(text)
            .into_iter()
            .map(|(tk, _)| tk)
            .collect()
    }

    /// Tokenize, returning each subtoken alongside its byte range in `text`.
    ///
    /// Needed to map [`LinkMarker`] byte ranges (from
    /// [`G2P::preprocess_links`]) onto specific [`MToken`]s after
    /// subtokenization. The Python equivalent uses spaCy's
    /// `Alignment.from_strings` against a whitespace-split token list; we use
    /// byte offsets instead, which is more robust to subtoken splits.
    pub fn tokenize_with_offsets(
        &self,
        text: &str,
    ) -> Vec<(MToken, std::ops::Range<usize>)> {
        let word_boundary_regex = Regex::new(r"\S+").unwrap();
        let mut tokens = Vec::new();

        for mat in word_boundary_regex.find_iter(text) {
            let word = mat.as_str();
            let word_start = mat.start();
            let subtokens: Vec<fancy_regex::Match> =
                self.subtoken_regex.find_iter(word).flatten().collect();

            if subtokens.is_empty() {
                let tk = MToken::new(word.to_string(), "NN".to_string(), String::new());
                tokens.push((tk, word_start..mat.end()));
            } else {
                for sub in subtokens {
                    let s = word_start + sub.start();
                    let e = word_start + sub.end();
                    let tk =
                        MToken::new(sub.as_str().to_string(), "NN".to_string(), String::new());
                    tokens.push((tk, s..e));
                }
            }
        }

        // Fill in each token's trailing whitespace from the SOURCE TEXT.
        //
        // misaki carries spaCy's `token.whitespace_`, i.e. the actual
        // characters between this token and the next -- empty when the next
        // token is adjacent. The assembled phoneme string is
        // `''.join(phonemes + whitespace)` (`en.py:711`), so hardcoding " "
        // here produced "b\u02c8\u0251\u0279kt . " where misaki gives
        // "b\u02c8\u0251\u0279kt.", and no phoneme string could ever match.
        // Byte ranges from the regex are always char boundaries, so slicing is
        // safe.
        for i in 0..tokens.len() {
            let gap_start = tokens[i].1.end;
            let gap_end = if i + 1 < tokens.len() {
                tokens[i + 1].1.start
            } else {
                text.len()
            };
            if gap_start < gap_end {
                tokens[i].0.whitespace = text[gap_start..gap_end].to_string();
            }
        }

        tokens
    }

    pub fn g2p(&self, text: &str) -> Result<(String, Vec<MToken>), G2PError> {
        #[cfg(feature = "fixes")]
        let text = &*crate::fixes::straight_quotes(text);
        // misaki `G2P.preprocess`: `text = text.lstrip()` before tokenization.
        let text = text.trim_start();
        // Parse `[text](feature)` markers and attach them to the subtokens
        // whose byte range falls inside the marker span. Mirrors the
        // `Lexicon.tokenize` feature-application pass in misaki/en.py.
        let (processed_text, markers) = self.preprocess_links(text);
        let mut tokens_with_spans = self.tokenize_with_offsets(&processed_text);

        for marker in &markers {
            let mut first_in_span = true;
            for (tk, span) in tokens_with_spans.iter_mut() {
                if span.start >= marker.byte_start && span.end <= marker.byte_end {
                    match &marker.feature {
                        LinkFeature::Stress(s) => {
                            tk.underscore_mut().stress = Some(*s);
                        }
                        LinkFeature::Phonemes(p) => {
                            // Only the head subtoken receives the override;
                            // trailing subtokens are blanked so the main loop
                            // skips them. Matches Python's
                            // `phonemes = v.lstrip('/') if i == 0 else ''`.
                            if first_in_span {
                                tk.phonemes = Some(p.clone());
                            } else {
                                tk.phonemes = Some(String::new());
                            }
                        }
                        LinkFeature::NumFlags(flags) => {
                            tk.underscore_mut().num_flags = flags.clone();
                        }
                    }
                    first_in_span = false;
                }
            }
        }

        let mut tokens: Vec<MToken> =
            tokens_with_spans.into_iter().map(|(tk, _)| tk).collect();

        // Tag at WORD level, as misaki does. spaCy tags its own tokens and
        // misaki's subtokens inherit the parent's tag; our tokenizer splits
        // finer than spaCy's, so tagging subtokens asks the model about
        // fragments it never saw in training ("Mr" and "." separately rather
        // than "Mr."). Group by adjacency first -- the same grouping that
        // reassembles words later -- tag the group texts, and give every
        // subtoken its group's tag.
        //
        // norm is the lowercased text. spaCy also applies a norm-exceptions
        // table through its tokenizer ("n't" -> "not"), which only fires on
        // pieces its own tokenizer produces; ours does not produce them, and
        // the group text ("hadn't") is what gets tagged.
        // misaki's pipeline, now reproduced rather than approximated:
        //
        //     doc = self.nlp(text)                  # spaCy tokenize + tag
        //     mutable_tokens = [MToken(text=tk.text, tag=tk.tag_, ...) for tk in doc]
        //     ... retokenize: subtokenize(token.text), each subtoken keeping the
        //         parent's tag, the LAST one keeping the parent's whitespace
        //
        // The grouping rule downstream is `tk.tag in PUNCT_TAGS` on a spaCy
        // token, so the tags have to come from spaCy's tokenization and not
        // from ours. Both halves are now exact against spaCy (771/771
        // sentences for the tokenizer, 20623/20623 tokens for the tagger), so
        // the approximations that used to live here -- merging abbreviations
        // and apostrophe pieces to guess at spaCy's boundaries -- are gone.
        let spacy_tokens = self.spacy_tokenizer.tokenize(&processed_text);
        let tagger_input: Vec<(String, String, bool)> = spacy_tokens
            .iter()
            .map(|tok| {
                let (t, ws) = (&tok.text, &tok.whitespace);
                // The NORM the tokenizer RULE supplied wins. It has to: `you’d`
                // gives its second piece the norm `'d` while `you’d’ve` gives
                // the same orth `would`, so no table keyed on the orth alone is
                // right, and with the wrong norm the tagger called that piece
                // an MD where spaCy says '' -- which decides whether the word
                // stays joined.
                let norm = tok
                    .norm
                    .clone()
                    .unwrap_or_else(|| crate::norms::norm_of(t, crate::spacy_tagger::hash_string));
                (t.clone(), norm, !ws.is_empty())
            })
            .collect();
        let group_tags = self.spacy_tagger.tag(&tagger_input);
        // Subtokenize each spaCy token with misaki's SUBTOKEN_REGEX. Every
        // subtoken inherits the parent's tag; only the last carries the
        // parent's trailing whitespace (`tks[-1].whitespace = token.whitespace`).
        //
        // Link markers are applied here, by BYTE SPAN against the
        // preprocessed text. Carrying them across by matching token text was
        // fragile and lost `[hello](-1)`: the same word appears more than once
        // and the first match wins.
        //
        // Spans are exact because the tokens concatenate back to the text --
        // each is `token + trailing whitespace`, in order.
        let mut rebuilt: Vec<MToken> = Vec::with_capacity(tokens.len());
        let mut tags: Vec<String> = Vec::new();
        // misaki's `j+1 == len(tks) and (i+1 == len(tokens) or tokens[i+1].tag != 'CD')`:
        // the last piece of a spaCy token not followed by a number.
        let mut ends_number: Vec<bool> = Vec::with_capacity(tokens.len());
        let mut phoneme_heads = std::collections::HashSet::new();
        let mut pos = 0usize;
        for (si, st) in spacy_tokens.iter().enumerate() {
            let (stext, sws) = (&st.text, &st.whitespace);
            let tag = group_tags.get(si).copied().unwrap_or("NN").to_string();
            let next_not_cd = group_tags.get(si + 1).is_none_or(|t| *t != "CD");
            let subs: Vec<String> = self
                .subtoken_regex
                .find_iter(stext)
                .flatten()
                .map(|m| m.as_str().to_string())
                .collect();
            let subs = if subs.is_empty() { vec![stext.clone()] } else { subs };
            let n = subs.len();
            let subs_snapshot = subs.clone();
            for (j, sub) in subs.into_iter().enumerate() {
                let start = pos;
                let end = pos + sub.len();
                pos = end;
                let mut tk = MToken::new(sub, tag.clone(), String::new());
                if j + 1 == n {
                    tk.whitespace = sws.clone();
                    pos += sws.len();
                }
                // misaki's `2` -> `to` alias (`retokenize`):
                //
                //     elif 0 < j < len(tks)-1 and tk.text == '2' \
                //          and (tks[j-1].text[-1]+tks[j+1].text[0]).isalpha():
                //         tk._.alias = 'to'
                //
                // A `2` strictly inside a word with letters either side is read
                // as "to", which is what makes `G2P` ʤˈitəpˈi. The alias also
                // makes the token stand alone, since misaki's first grouping
                // branch takes any token with an alias or phonemes.
                if j > 0 && j + 1 < n && tk.text == "2" {
                    let left_ok = subs_snapshot
                        .get(j - 1)
                        .and_then(|t: &String| t.chars().last())
                        .map(|c| c.is_alphabetic())
                        .unwrap_or(false);
                    let right_ok = subs_snapshot
                        .get(j + 1)
                        .and_then(|t: &String| t.chars().next())
                        .map(|c| c.is_alphabetic())
                        .unwrap_or(false);
                    if left_ok && right_ok {
                        tk.underscore_mut().alias = Some("to".to_string());
                    }
                }

                for (mi, marker) in markers.iter().enumerate() {
                    if start >= marker.byte_start && end <= marker.byte_end {
                        match &marker.feature {
                            LinkFeature::Stress(st) => {
                                tk.underscore_mut().stress = Some(*st);
                            }
                            LinkFeature::Phonemes(ph) => {
                                // Across the whole marker span, only the first
                                // subtoken is the head (`en.py` tokenize: `i == 0`).
                                let is_head = phoneme_heads.insert(mi);
                                if is_head {
                                    tk.phonemes = Some(ph.clone());
                                    tk.underscore_mut().rating = Some(5);
                                    tk.underscore_mut().is_head = true;
                                } else {
                                    tk.phonemes = Some(String::new());
                                    tk.underscore_mut().rating = Some(5);
                                    tk.underscore_mut().is_head = false;
                                }
                            }
                            LinkFeature::NumFlags(f) => {
                                tk.underscore_mut().num_flags = f.clone();
                            }
                        }
                    }
                }
                rebuilt.push(tk);
                tags.push(tag.clone());
                ends_number.push(j + 1 == n && next_not_cd);
            }
        }
        // Align tags onto rebuilt tokens, then fold non-head link pieces.
        for (tk, tag) in rebuilt.iter_mut().zip(tags.iter()) {
            tk.tag = tag.clone();
        }
        let (folded, folded_ends) = Self::fold_left(rebuilt, ends_number);
        tokens = folded;
        ends_number = folded_ends;

        log::debug!(
            "g2p '{}' -> {} tokens",
            text,
            tokens.len(),
        );
        for (i, tk) in tokens.iter().enumerate() {
            log::debug!("token[{}]: '{}'", i, tk.text);
        }

        // Process tokens in reverse order (like Python) to build context
        let mut contexts: Vec<crate::lexicon::TokenContext> =
            vec![crate::lexicon::TokenContext::default(); tokens.len()];

        // `is_head` per subtoken. misaki sets `tk._.is_head = False` on any
        // subtoken that JOINS an open group, and `get_number` branches on it:
        // a three-digit group that is not the head reads as digit-then-pair
        // rather than hundreds, so the 555 of a phone number is "five fifty
        // five" and not "five hundred fifty five".
        let head_groups = Self::group_by_adjacency(&tokens);
        let mut is_head: Vec<bool> = vec![false; tokens.len()];
        let mut is_subtoken_group = vec![false; tokens.len()];
        for g in &head_groups {
            if g.start < is_head.len() {
                is_head[g.start] = true;
            }
            if g.len() > 1 {
                is_subtoken_group[g.clone()].fill(true);
            }
        }
        #[cfg(feature = "fixes")]
        crate::fixes::perfect_participles(&mut tokens, &head_groups);

        // misaki's currency state machine (`retokenize`):
        //
        //     elif tk.tag == '$' and tk.text in CURRENCIES:
        //         currency = tk.text; tk.phonemes = ''; tk._.rating = 4
        //     elif <a dash or punctuation>: ...
        //     elif currency is not None:
        //         if tk.tag != 'CD':
        //             currency = None
        //         elif j+1 == len(tks) and (i+1 == len(tokens) or tokens[i+1].tag != 'CD'):
        //             tk._.currency = currency
        //
        // The symbol becomes its own silent token and the unit is spoken by the
        // LAST number of the amount: `$9` is [('$', ''), ('9', 'nˈIn dˈɑləɹz')],
        // `$2 billion` is "two billion dollars", and `$9 and 5 more` leaves the
        // 5 bare.
        //
        // This has to be a separate pass because spaCy tokenizes the symbol
        // away from the digits, so the two are never in one group.
        {
            let mut currency: Option<char> = None;
            for i in 0..tokens.len() {
                if tokens[i].phonemes.is_some() {
                    continue;
                }
                let mut cs = tokens[i].text.chars();
                let symbol = match (cs.next(), cs.next()) {
                    (Some(c), None) if Self::currency_units(c).is_some() => Some(c),
                    _ => None,
                };
                if tokens[i].tag == "$" && symbol.is_some() {
                    currency = symbol;
                    tokens[i].phonemes = Some(String::new());
                    tokens[i].underscore_mut().rating = Some(4);
                } else if Self::is_punct_token(&tokens[i]) {
                    // punctuation leaves the amount open
                } else if let Some(c) = currency {
                    if tokens[i].tag != "CD" {
                        currency = None;
                    } else if ends_number[i] {
                        tokens[i].underscore_mut().currency = Some(c.to_string());
                    }
                }
            }
        }

        // misaki assigns punctuation its phonemes in `retokenize`, i.e. BEFORE
        // any lexicon lookup, and the resolution loop then skips those tokens
        // because `tk.phonemes is not None`. Order matters: our lookup
        // normalises ‘ to a straight quote, which the lexicon resolves to ”, so
        // reaching the lexicon at all gave ‘ the CLOSING glyph.
        //
        // The test is misaki's own, on the TAG. A character test was wrong in
        // both directions: it gave `[` an opening paren where spaCy tags it XX
        // rather than -LRB-, so misaki merges it into the following word and it
        // contributes NOTHING -- `[Illustration` is ˌɪləstrˈAʃən, not
        // `( ˌɪləstrˈAʃən`.
        for i in 0..tokens.len() {
            if tokens[i].phonemes.is_some() || tokens[i].text.is_empty() {
                continue;
            }
            // misaki's dash branch comes BEFORE the general punctuation one
            // (`retokenize`):
            //
            // ```text
            //     elif tk.tag == ':' and tk.text in ('-', '–'):
            //         tk.phonemes = '—'
            // ```
            //
            // A hyphen or en dash that spaCy tagged ':' is spoken as an EM
            // dash. The general path drops it instead, since '-' is not in
            // PUNCTS -- so `laws-a-me` lost its dashes and `1-25—1469-94` ran
            // its first two numbers together.
            if tokens[i].tag == ":" && (tokens[i].text == "-" || tokens[i].text == "\u{2013}") {
                tokens[i].phonemes = Some("\u{2014}".to_string());
                tokens[i].underscore_mut().rating = Some(3);
                continue;
            }
            if Self::is_punct_token(&tokens[i]) {
                let tg = tokens[i].tag.clone();
                tokens[i].phonemes = Some(Self::punct_phonemes(&tokens[i].text, &tg));
                tokens[i].underscore_mut().rating = Some(4);
            }
        }

        // Process in reverse to build context from future tokens
        for i in (0..tokens.len()).rev() {
            // misaki normalises the LOOKUP KEY before touching the lexicon
            // (`Lexicon.__call__`):
            //
            //     word = tk.text.replace(chr(8216), "'").replace(chr(8217), "'")
            //     word = unicodedata.normalize('NFKC', word)
            //
            // The lexicons are keyed with STRAIGHT apostrophes -- "it's" is in
            // us_gold as ɪts, "I'm" as ˌIm, "'s" as s -- so prose using curly
            // ’ missed every one of them and fell through to spelling the
            // pieces out. The token's own text keeps the original character.
            // misaki looks up `tk._.alias` in place of the text when one is
            // set (`word = tk.text if tk._.alias is None else tk._.alias`), and
            // it happens HERE rather than in a pre-pass so the lookup sees the
            // rolling context: `to` is tə before a consonant and tu before a
            // vowel, and in `G2P` the `P` that follows is a consonant.
            // Python normalises alias and text the same way.
            let word = match tokens[i].underscore().alias.clone() {
                Some(a) => Self::normalize_lookup(&a),
                None => Self::normalize_lookup(&tokens[i].text),
            };
            let tag = tokens[i].tag.clone();
            // Cap stress goes into `get_word`; link stress is applied AFTER
            // (`en.py` Lexicon.__call__: `apply_stress(..., tk._.stress)`).
            let cap_stress = if word == word.to_lowercase() {
                None
            } else if word == word.to_uppercase() {
                Some(self.lexicon.cap_stresses.1)
            } else {
                Some(self.lexicon.cap_stresses.0)
            };
            let link_stress = tokens[i].underscore().stress;

            // NOTE: `contexts[i]` was already filled by iteration i+1 from that
            // token's RESOLVED PHONEMES -- see the `token_context` call at the
            // end of this loop. There was an orthographic heuristic here
            // ("does the next word start with aeiou") that overwrote it on
            // every iteration, which is why `the FBI` came out `\u00f0\u0259` instead of
            // `\u00f0i`: F is not in "aeiou", but FBI's phonemes begin `\u02cc\u025b`, a vowel.
            // misaki decides this from phonemes, never from spelling.

            // Process current token
            if tokens[i].phonemes.is_none() {
                let ctx = Some(&contexts[i]);

                // Use get_word which handles special cases, lookup, and stemming
                if let Some((ps, _)) = self.lexicon.get_word(&word, &tag, cap_stress, ctx) {
                    let ps = self.append_currency(ps, Self::currency_of(&tokens[i]));
                    tokens[i].phonemes = Some(self.lexicon.apply_stress(&ps, link_stress));
                }

                if tokens[i].phonemes.is_none() {
                    // The number path comes FIRST. misaki has no
                    // hyphen-splitting branch at all -- it is an invention of
                    // this port -- and letting it win meant a signed number was
                    // split on its own sign: `-40` became "" + "forty" and the
                    // minus was simply lost, where misaki says mˈInəs fˈɔɹTi.
                    if self.is_number(&word, is_head[i]) {
                        let flags = tokens[i].underscore().num_flags.clone();
                        let cur = Self::currency_of(&tokens[i]);
                        if let Some(p) =
                            self.get_number_phonemes(&word, cur, is_head[i], &flags)
                        {
                            tokens[i].phonemes = Some(p);
                        }
                    } else if word.contains('-') && word.len() > 1 {
                        // Handle hyphenated words like "twenty-one"
                        let parts: Vec<&str> = word.split('-').filter(|s| !s.is_empty()).collect();
                        let mut sub_ps = Vec::new();
                        for part in parts {
                            let (p, _) = self.g2p(part)?;
                            sub_ps.push(p);
                        }
                        tokens[i].phonemes = Some(sub_ps.join(" "));
                    }
                }

                if tokens[i].phonemes.is_none() {
                    if let Some(ps) = self.rules.apply_rules(&word, &tag, &self.lexicon) {
                        tokens[i].phonemes = Some(ps);
                    }
                }

                if tokens[i].phonemes.is_none() && word.chars().count() == 1 {
                    // Try to normalize the character or leave unresolved.
                    let normalized: String = word
                        .chars()
                        .map(|c| match c {
                            'é' | 'è' | 'ê' | 'ë' => 'e',
                            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
                            'í' | 'ì' | 'î' | 'ï' => 'i',
                            'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
                            'ú' | 'ù' | 'û' | 'ü' => 'u',
                            'ñ' => 'n',
                            'ç' => 'c',
                            // NB: em/en dashes are NOT mapped away here.
                            // U+2014 is in misaki's PUNCTS and survives as
                            // itself; mapping it to a space sent it down the
                            // renormalise path and silently dropped it.
                            _ => c,
                        })
                        .collect();

                    if normalized != word {
                        let (p, _) = self.g2p(&normalized)?;
                        tokens[i].phonemes = Some(p);
                    } else {
                        // Punctuation, as misaki resolves it (`en.py:598-599`):
                        //
                        //     elif tk.tag in PUNCT_TAGS and not all(97 <= ord(c.lower()) <= 122 ...):
                        //         tk.phonemes = PUNCT_TAG_PHONEMES.get(
                        //             tk.tag, ''.join(c for c in tk.text if c in PUNCTS))
                        //
                        // with PUNCTS = ';:,.!?\u2014\u2026"\u201c\u201d' and PUNCT_TAG_PHONEMES
                        // mapping spaCy's quote/paren tags: `` -> \u201c, '' -> \u201d,
                        // -LRB- -> (, -RRB- -> ).
                        //
                        // The punctuation pre-pass above already handled
                        // every token misaki calls punctuation, by TAG. A
                        // non-alphanumeric token that gets here is therefore
                        // one spaCy did NOT tag as punctuation -- `[` is
                        // tagged XX, not -LRB- -- and misaki leaves it
                        // unresolved when fallback is None, so it contributes
                        // nothing when merged into the following word.
                        //
                        // SUBTOKEN_JUNKS still resolve to the empty string
                        // (resolve_tokens), which the merge pass applies.
                        tokens[i].phonemes = None;
                    }
                }

                // misaki: any still-unresolved non-group token goes to fallback
                // (`en.py`: `if w.phonemes is None and self.fallback is not None`),
                // including single-character symbols (`-`, `#`, `©`, emoji).
                // Multi-piece groups defer to the merge-pass `should_fallback`.
                if tokens[i].phonemes.is_none() && !is_subtoken_group[i] {
                    if let Some(ref fallback) = self.fallback {
                        // Python passes the original token text to fallback,
                        // not the NFKC-normalised lookup key.
                        let raw = tokens[i].text.clone();
                        match fallback.phonemize(&raw) {
                            // `Ok("")` is successful silence (Python HYPH `-` →
                            // `('', 2)`). Soft-fail is `Err(NoPhonemes)`.
                            Ok(ps) => tokens[i].phonemes = Some(ps),
                            Err(crate::fallback::FallbackError::NoPhonemes { .. }) => {}
                            Err(e) => {
                                log::error!("fallback error for '{}': {}", raw, e);
                                return Err(G2PError::Fallback(e));
                            }
                        }
                    }
                }
            }

            // Propagate context leftwards, exactly as misaki's
            // `G2P.token_context` does (`en.py`):
            //
            //     vowel = next((None if c in NON_QUOTE_PUNCTS else (c in VOWELS)
            //                   for c in ps if any(c in s for s in
            //                   (VOWELS, CONSONANTS, NON_QUOTE_PUNCTS))), vowel) if ps else vowel
            //
            // Three things the previous version got wrong and this one does not:
            // punctuation RESETS future_vowel to None rather than being skipped;
            // a token whose phonemes contain no vowel, consonant or punctuation
            // CARRIES THE PREVIOUS value rather than leaving a stale one; and
            // future_to is set here from the token, not guessed from spelling.
            if i > 0 {
                contexts[i - 1] =
                    Self::token_context(&contexts[i], tokens[i].phonemes.as_deref(), &tokens[i]);
            }
        }

        // ---- merge adjacent subtokens into words (misaki's `merge_tokens`) ----
        //
        // misaki's final token list has ONE token per adjacency group:
        //
        //     tokens = [merge_tokens(tk, unk=self.unk) if isinstance(tk, list) else tk
        //               for tk in tokens]
        //
        // and this runs BEFORE the flap pass below, which is why it is placed
        // here. Without it our tokenizer's finer splits reached the output and
        // 257 of 700 prose cases differed in token count -- every one of which
        // corrupts the word timestamps, since `pred_dur` is per token.
        //
        // Phonemes are joined with no separator unless `prespace`, which misaki
        // derives in `resolve_tokens` from the group's text:
        //
        //     prespace = ' ' in text or '/' in text or len({0 if c.isalpha()
        //                else (1 if is_digit(c) else 2) for c in text
        //                if c not in SUBTOKEN_JUNKS}) > 1
        //
        // i.e. a space, a slash, or a MIX of character classes once the junk
        // characters are ignored. `five-storied` is all-alpha once `-` is
        // dropped, so it joins tight: fˌIv + stˈɔɹid -> fˌIvstˈɔɹid.
        // Parallel to `tokens` after merge: phonemes fed to re-resolve
        // `token_context`. `None` means "do not advance rolling" (Python
        // `should_fallback` with no prior span hit). `Some(None)` means use
        // the token's own phonemes. `Some(Some(ps))` means a partial span hit
        // updated ctx before `should_fallback` overwrote the word (e.g. `15`
        // of `3:42:15` → `in` stays unstressed).
        let mut ctx_feed: Vec<Option<Option<String>>> = vec![Some(None); tokens.len()];
        {
            const SUBTOKEN_JUNKS: &str = "',-._\u{2018}\u{2019}/";
            let groups = Self::group_by_adjacency(&tokens);
            if groups.len() != tokens.len() {
                let mut merged: Vec<MToken> = Vec::with_capacity(groups.len());
                let mut feeds: Vec<Option<Option<String>>> = Vec::with_capacity(groups.len());
                for g in groups {
                    if g.len() == 1 {
                        merged.push(tokens[g.start].clone());
                        feeds.push(Some(None));
                        continue;
                    }
                    let group = &tokens[g.clone()];
                    let (text, tag, ws) = Self::merge_text_tag(group);

                    // misaki's `resolve_tokens`: a subtoken made entirely of
                    // SUBTOKEN_JUNKS contributes NOTHING.
                    //
                    //     elif all(c in SUBTOKEN_JUNKS for c in tk.text):
                    //         tk.phonemes = ''
                    //
                    // Without this the word-internal ’ in `it’s` kept the ”
                    // that the standalone-quote path gives it, producing
                    // ɪt”ˈɛs instead of ɪts.
                    let part_of = |tk: &MToken| -> Option<String> {
                        if !tk.text.is_empty() && tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                            Some(String::new())
                        } else {
                            tk.phonemes.clone()
                        }
                    };

                    let classes: std::collections::BTreeSet<u8> = text
                        .chars()
                        .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
                        .map(|c| if c.is_alphabetic() { 0u8 } else if c.is_ascii_digit() { 1 } else { 2 })
                        .collect();
                    let prespace =
                        text.contains(' ') || text.contains('/') || classes.len() > 1;

                    // misaki tries the whole word in the lexicon first and only
                    // falls back to stitching the parts together.
                    // misaki's multi-token loop (`G2P.__call__`). It does not
                    // pick one span -- it resolves the word in PIECES, right to
                    // left, and the result is their concatenation:
                    //
                    //     left, right = 0, len(w)
                    //     while left < right:
                    //         tk = merge_tokens(w[left:right])      # if none resolved yet
                    //         ps, rating = self.lexicon(tk, ctx)
                    //         if ps is not None:
                    //             w[left].phonemes = ps
                    //             for x in w[left+1:right]: x.phonemes = ''
                    //             right = left; left = 0
                    //         elif left + 1 < right: left += 1
                    //         else:
                    //             right -= 1
                    //             if all(c in SUBTOKEN_JUNKS for c in w[right].text):
                    //                 w[right].phonemes = ''
                    //             left = 0
                    //
                    // `Austen’s` needs this: "Austen's" misses, then the span
                    // "'s" HITS (us_gold holds "'s" as s), and Austen stays
                    // unresolved -- giving misaki's `s`. Trying only the whole
                    // word stitched on the letter name ˈɛs instead.
                    // misaki's `Lexicon.__call__` tries get_word, then the
                    // NUMBER path, before anything else:
                    //
                    //     ps, rating = self.get_word(...)
                    //     if ps is not None: return ...
                    //     elif Lexicon.is_number(word, tk._.is_head):
                    //         ps, rating = self.get_number(...)
                    //
                    // Our tokenizer splits `1st` into `1` + `st`, so the group
                    // text is the first point at which it looks like an ordinal
                    // at all. Without this `1st` read as "one" and `23rd` as
                    // "twenty three".
                    // Currency. misaki's `Lexicon.get_number` pairs the
                    // dot-separated parts with CURRENCIES[sym] -- $ is
                    // (dollar, cent), £ is (pound, pence), € is (euro, cent) --
                    // drops a zero part, joins the rest with "and", and uses the
                    // singular at exactly one:
                    //
                    //   $9        -> nine dollars
                    //   $1        -> one dollar
                    //   $0.50     -> fifty cents        (the zero major is dropped)
                    //   $1,250.75 -> one thousand two hundred fifty dollars
                    //                and seventy five cents
                    if self.is_number(&text, true) {
                        let cur = group.iter().find_map(Self::currency_of);
                        let flags = group
                            .iter()
                            .map(|t| t.underscore().num_flags.as_str())
                            .collect::<String>();
                        if let Some(p) = self.get_number_phonemes(&text, cur, true, &flags) {
                            let mut tk = group[0].clone();
                            tk.text = text;
                            tk.tag = tag;
                            tk.whitespace = ws;
                            tk.phonemes = Some(p);
                            merged.push(tk);
                            feeds.push(Some(None));
                            continue;
                        }
                    }

                    let n = g.len();
                    let mut res: Vec<Option<String>> = vec![None; n];
                    let mut left = 0usize;
                    let mut right = n;
                    // Last phonemes that updated ctx in the span loop (Python
                    // `ctx = token_context(ctx, ps, tk)` on each hit). Survives
                    // `should_fallback`, which does not itself update ctx.
                    let mut last_ctx_ps: Option<String> = None;
                    let mut should_fallback = false;
                    while left < right {
                        // misaki only merges a span none of whose members has
                        // resolved yet.
                        let free = res[left..right].iter().all(|r| r.is_none());
                        let hit = if free {
                            let sub = &tokens[g.start + left..g.start + right];
                            let (t, tg, _) = Self::merge_text_tag(sub);
                            let k = Self::normalize_lookup(&t);
                            let st = if k == k.to_lowercase() {
                                None
                            } else if k == k.to_uppercase() {
                                Some(self.lexicon.cap_stresses.1)
                            } else {
                                Some(self.lexicon.cap_stresses.0)
                            };
                            // the context at the SPAN's own left edge, not the
                            // group's: misaki carries a rolling ctx through its
                            // loop, so the inner `the` of `out-of-the-way` sees
                            // what follows IT (`way`, a consonant -> ðə) and not
                            // what follows the whole word.
                            self.lexicon
                                .get_word(&k, &tg, st, Some(&contexts[g.start + left]))
                                .map(|(ps, _)| {
                                    self.append_currency(ps, sub.iter().find_map(Self::currency_of))
                                })
                                .or_else(|| self.number_with_ending(&k, sub, left == 0))
                                .or_else(|| {
                                    // Non-head / ordinal / dotted pieces inside a
                                    // multi-piece group (e.g. `.5` of `No.5`).
                                    if !self.is_number(&k, left == 0) {
                                        return None;
                                    }
                                    self.get_number_phonemes(&k, None, left == 0, "")
                                })
                        } else {
                            None
                        };
                        if let Some(ps) = hit {
                            last_ctx_ps = Some(ps.clone());
                            res[left] = Some(ps);
                            for k in left + 1..right {
                                res[k] = Some(String::new());
                            }
                            right = left;
                            left = 0;
                        } else if left + 1 < right {
                            left += 1;
                        } else {
                            right -= 1;
                            if res[right].is_none() {
                                let t = &tokens[g.start + right].text;
                                // Span shrink only blanks SUBTOKEN_JUNKS (`.`, `-`,
                                // …). NON_QUOTE last-piece rule lives in
                                // resolve_tokens *after* this loop; if JUNKS
                                // already set `''`, NON_QUOTE must not override.
                                if !t.is_empty()
                                    && t.chars().all(|c| SUBTOKEN_JUNKS.contains(c))
                                {
                                    res[right] = Some(String::new());
                                } else if self.fallback.is_some()
                                    && group[right].phonemes.is_none()
                                    && !t.is_empty()
                                {
                                    // Python: unresolved non-JUNKS piece with
                                    // fallback set → should_fallback; break
                                    // without trying shorter head spans.
                                    should_fallback = true;
                                    break;
                                }
                            }
                            left = 0;
                        }
                    }
                    if should_fallback {
                        if let Some(fallback) = &self.fallback {
                            match fallback.phonemize(&text) {
                                Ok(ps) => {
                                    let mut tk = group[0].clone();
                                    tk.text = text;
                                    tk.tag = tag;
                                    tk.whitespace = ws;
                                    tk.phonemes = Some(ps);
                                    merged.push(tk);
                                    // Feed partial-hit phonemes (if any), else skip.
                                    feeds.push(last_ctx_ps.map(Some));
                                    continue;
                                }
                                Err(crate::fallback::FallbackError::NoPhonemes { .. }) => {}
                                Err(e) => return Err(G2PError::Fallback(e)),
                            }
                        }
                    } else if let Some(fallback) = &self.fallback {
                        // Residual: pieces still missing after a full resolve
                        // (e.g. per-token already had phonemes that blocked a
                        // free span). Same whole-group fallback policy.
                        let needs_fallback = res.iter().enumerate().any(|(i, part)| {
                            part.is_none()
                                && group[i].phonemes.is_none()
                                && !group[i].text.is_empty()
                                && !group[i]
                                    .text
                                    .chars()
                                    .all(|c| SUBTOKEN_JUNKS.contains(c))
                        });
                        if needs_fallback {
                            match fallback.phonemize(&text) {
                                Ok(ps) => {
                                    let mut tk = group[0].clone();
                                    tk.text = text;
                                    tk.tag = tag;
                                    tk.whitespace = ws;
                                    tk.phonemes = Some(ps);
                                    merged.push(tk);
                                    feeds.push(last_ctx_ps.map(Some));
                                    continue;
                                }
                                Err(crate::fallback::FallbackError::NoPhonemes { .. }) => {}
                                Err(e) => return Err(G2PError::Fallback(e)),
                            }
                        }
                    }
                    // Anything still unresolved stays unresolved, as misaki
                    // leaves it: merge_tokens substitutes `unk` for None.
                    // The stitched path below assembles whatever `res` holds.
                    let whole: Option<String> = None;

                    let phonemes = whole.or_else(|| {
                        // misaki's `resolve_tokens` rebalances stress across the
                        // parts of one word before they are joined:
                        //
                        //     indices = [(PRIMARY_STRESS in tk.phonemes,
                        //                 stress_weight(tk.phonemes), i) ...]
                        //     if len(indices) == 2 and len(tokens[indices[0][2]].text) == 1:
                        //         demote indices[1]
                        //     elif len(indices) < 2 or sum(b for b,_,_ in indices) <= (len(indices)+1)//2:
                        //         return
                        //     for _,_,i in sorted(indices)[:len(indices)//2]: demote i
                        //
                        // where demote is `apply_stress(ps, -0.5)` -- primary
                        // becomes secondary. Without it a compound kept a primary
                        // stress on every part: `Caucus-Race` came out
                        // kˈɔkəsɹˈAs where misaki gives kˈɔkəsɹˌAs.
                        //
                        // `stress_weight` counts diphthongs double, which is what
                        // decides WHICH part is demoted: ɹˈAs weighs 5 (A is a
                        // diphthong) against kˈɔkəs's 6, so Race loses its primary.
                        // None stays None until the join, where misaki's
                        // `merge_tokens` writes `unk` with no space before it.
                        let mut parts: Vec<Option<String>> = res
                            .iter()
                            .enumerate()
                            .map(|(i, r)| match r {
                                Some(ps) => Some(ps.clone()),
                                // Fall back to whatever the per-token pass found
                                // only for a token the span loop never reached.
                                None => part_of(&group[i]),
                            })
                            .collect();
                        // `resolve_tokens` RETURNS before rebalancing when the
                        // parts are space-separated:
                        //
                        //     if prespace:
                        //         return
                        //
                        // so a mixed-class word keeps every primary stress.
                        // That is why misaki gives `50%` as fˈɪfti pəɹsˈɛnt and not
                        // fˌɪfti pəɹsˈɛnt -- '5','0' and '%' are different classes,
                        // prespace is true, and no demotion happens.
                        if !prespace {
                            const PRIMARY: char = '\u{2c8}';
                            const DIPHTHONGS: &str = "AIOQWY\u{2a4}\u{2a7}";
                            let weight = |ps: &str| -> usize {
                                ps.chars()
                                    .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
                                    .sum()
                            };
                            let mut idx: Vec<(bool, usize, usize)> = parts
                                .iter()
                                .enumerate()
                                .filter_map(|(i, p)| p.as_deref().map(|p| (i, p)))
                                .filter(|(_, p)| !p.is_empty())
                                .map(|(i, p)| (p.contains(PRIMARY), weight(p), i))
                                .collect();
                            let n_primary = idx.iter().filter(|(b, _, _)| *b).count();
                            let demote = |p: &mut Option<String>| {
                                if let Some(ps) = p {
                                    *ps = self.lexicon.apply_stress(ps, Some(-0.5));
                                }
                            };
                            if idx.len() == 2
                                && group[idx[0].2].text.chars().count() == 1
                            {
                                demote(&mut parts[idx[1].2]);
                            } else if idx.len() >= 2 && n_primary > (idx.len() + 1) / 2 {
                                idx.sort();
                                let half = idx.len() / 2;
                                for (_, _, i) in idx.into_iter().take(half) {
                                    demote(&mut parts[i]);
                                }
                            }
                        }

                        let mut out = String::new();
                        for part in parts {
                            match part {
                                Some(part) => {
                                    if prespace
                                        && !out.is_empty()
                                        && !out.ends_with(char::is_whitespace)
                                        && !part.is_empty()
                                    {
                                        out.push(' ');
                                    }
                                    out.push_str(&part);
                                }
                                None => out.push_str(&self.unk),
                            }
                        }
                        Some(out)
                    });

                    let mut tk = group[0].clone();
                    tk.text = text;
                    tk.tag = tag;
                    tk.whitespace = ws;
                    tk.phonemes = phonemes;
                    merged.push(tk);
                    feeds.push(Some(None));
                }
                debug_assert_eq!(merged.len(), feeds.len());
                tokens = merged;
                ctx_feed = feeds;
            }
        }

        // ---- contexts from the MERGED words, then re-resolve ----
        //
        // misaki's resolution loop iterates over adjacency GROUPS, so the
        // lookahead context it carries leftward comes from the phonemes of a
        // whole word. Ours resolves subtokens first and merges afterwards, so
        // the context came from a SUBTOKEN, and a word whose merged phonemes
        // differ from its parts' gave the wrong answer:
        //
        // ```text
        //     "I belong to Tellson’s Bank."
        //       misaki  to -> tə    (merged `Tellson’s` is `s`, a consonant)
        //       ours    to -> tʊ    (the `s` SUBTOKEN is the letter name ˈɛs)
        // ```
        //
        // Rather than restructure the loop, the contexts are recomputed here
        // over the finished word list -- which is exactly what misaki's loop
        // sees -- and the words whose pronunciation depends on that context are
        // resolved again against it.
        //
        // `ctx_feed` mirrors Python's multipart ctx updates: whole-group
        // fallback does not call `token_context` (skip), but a span hit before
        // `should_fallback` does (override phonemes).
        //
        // The re-resolution is deliberately narrow. It touches a token only
        // when it is a plain alphabetic word with no alias, already resolved by
        // the lexicon, and `get_word` under the new context returns something
        // else. Numbers, currency, punctuation, aliases and out-of-vocabulary
        // words are left exactly as they were.
        {
            let n = tokens.len();
            debug_assert_eq!(n, ctx_feed.len());
            let mut ctxs = vec![crate::lexicon::TokenContext::default(); n];
            let mut rolling = crate::lexicon::TokenContext::default();
            for i in (0..n).rev() {
                ctxs[i] = rolling.clone();
                match &ctx_feed[i] {
                    None => {
                        // should_fallback with no prior span hit: leave rolling.
                    }
                    Some(None) => {
                        rolling = Self::token_context(
                            &rolling,
                            tokens[i].phonemes.as_deref(),
                            &tokens[i],
                        );
                    }
                    Some(Some(ps)) => {
                        rolling =
                            Self::token_context(&rolling, Some(ps.as_str()), &tokens[i]);
                    }
                }
            }
            for i in 0..n {
                if tokens[i].underscore().alias.is_some()
                    || tokens[i].underscore().currency.is_some()
                    || tokens[i].underscore().rating == Some(5)
                {
                    // rating 5 = link phoneme override — never clobber.
                    continue;
                }
                let text = tokens[i].text.clone();
                if text.is_empty() || !text.chars().all(|c| c.is_alphabetic()) {
                    continue;
                }
                match tokens[i].phonemes.as_deref() {
                    Some(p) if !p.is_empty() => {}
                    _ => continue,
                }
                let key = Self::normalize_lookup(&text);
                let cap_stress = if key == key.to_lowercase() {
                    None
                } else if key == key.to_uppercase() {
                    Some(self.lexicon.cap_stresses.1)
                } else {
                    Some(self.lexicon.cap_stresses.0)
                };
                let link_stress = tokens[i].underscore().stress;
                let tag = tokens[i].tag.clone();
                if let Some((ps, _)) =
                    self.lexicon.get_word(&key, &tag, cap_stress, Some(&ctxs[i]))
                {
                    tokens[i].phonemes = Some(self.lexicon.apply_stress(&ps, link_stress));
                }
            }
        }

        // misaki applies exactly this, after all resolution and merging and
        // before assembling the string (`en.py:708-710`, gated on
        // `version != '2.0'`):
        //
        //     tk.phonemes = tk.phonemes.replace('\u027e', 'T').replace('\u0294', 't')
        //
        // The lexicons store \u027e and \u0294; Kokoro's vocab wants T and t. It runs
        // AFTER resolution on purpose -- the context pass matches \u027e against its
        // consonant set, so replacing earlier would break that -- and it is
        // applied to the TOKENS, not just the assembled string, because
        // `pred_dur` has one entry per token and the word timestamps are
        // derived from the per-token phonemes.
        for tk in tokens.iter_mut() {
            if let Some(ps) = tk.phonemes.as_mut() {
                if !ps.is_empty() {
                    *ps = ps.replace('\u{27e}', "T").replace('\u{294}', "t");
                }
            }
        }

        let result = tokens
            .iter()
            .map(|tk| tk.phonemes.as_ref().unwrap_or(&self.unk).clone() + &tk.whitespace)
            .collect::<String>();

        Ok((result, tokens))
    }

    /// `G2P.fold_left`: merge tokens with `is_head == false` into the previous.
    fn fold_left(tokens: Vec<MToken>, ends_number: Vec<bool>) -> (Vec<MToken>, Vec<bool>) {
        let mut result: Vec<MToken> = Vec::with_capacity(tokens.len());
        let mut ends: Vec<bool> = Vec::with_capacity(tokens.len());
        for (i, tk) in tokens.into_iter().enumerate() {
            let end_flag = ends_number.get(i).copied().unwrap_or(false);
            if !result.is_empty() && !tk.underscore().is_head {
                let prev = result.last_mut().unwrap();
                prev.text.push_str(&tk.text);
                prev.text.push_str(&tk.whitespace);
                // Keep the trailing whitespace of the last merged piece.
                prev.whitespace = tk.whitespace.clone();
                if prev.underscore().num_flags.is_empty()
                    && !tk.underscore().num_flags.is_empty()
                {
                    prev.underscore_mut().num_flags = tk.underscore().num_flags.clone();
                }
                if prev.underscore().stress.is_none() {
                    prev.underscore_mut().stress = tk.underscore().stress;
                }
                if let Some(last) = ends.last_mut() {
                    *last = end_flag;
                }
            } else {
                result.push(tk);
                ends.push(end_flag);
            }
        }
        (result, ends)
    }

    /// Python `is_digit` in misaki/en.py: `re.match(r'^[0-9]+$', text)`.
    fn ascii_digit_str(s: &str) -> bool {
        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
    }

    /// Case-sensitive suffix peel matching `Lexicon.get_number` /
    /// `Lexicon.is_number` (`ing`, `'d`, `ed`, `'s`, ordinals, `s`).
    fn peel_number_suffix(word: &str) -> (&str, Option<&str>) {
        const SUFFIXES: &[&str] = &[
            "ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s",
        ];
        for s in SUFFIXES {
            if let Some(rest) = word.strip_suffix(s) {
                return (rest, Some(*s));
            }
        }
        (word, None)
    }

    /// Does this token go down misaki's number path?
    ///
    /// misaki's `Lexicon.is_number(word, is_head)` peels one case-sensitive
    /// suffix first (`5TH` is NOT an ordinal), then requires at least one
    /// ASCII digit and only digits / `,` / `.` / leading `-` (when head).
    fn is_number_like(&self, word: &str) -> bool {
        self.is_number(word, true)
    }

    /// misaki's `Lexicon.is_number(word, is_head)`.
    fn is_number(&self, word: &str, is_head: bool) -> bool {
        if !word.chars().any(|c| c.is_ascii_digit()) {
            return false;
        }
        let (body, _) = Self::peel_number_suffix(word);
        body.chars().enumerate().all(|(i, c)| {
            c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-')
        })
    }

    /// Reduce a num2words string to the word sequence misaki feeds the lexicon.
    ///
    /// misaki does `re.split(r'[^a-z]+', num2words(int(num)))` and looks each
    /// piece up on its own (`Lexicon.get_number`), so hyphens and commas become
    /// word boundaries -- "fifty-seven" is two lookups, not one. It also drops
    /// "and" unless '&' is in num_flags, which is why 2001 reads
    /// "two thousand one".
    ///
    /// Without this, "457" produced `f\u02c8\u0254\u0279 h\u02c8\u028cndr\u0259d f\u02c8\u026afti s\u02c8\u025bv\u0259n` with the last
    /// two words run together, because "fifty-seven" went to the lexicon whole.
    fn normalize_num_words(spoken: &str) -> String {
        spoken
            .split(|c: char| !c.is_ascii_alphabetic())
            .filter(|w| !w.is_empty() && *w != "and")
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// num2words' `to='year'` form, which misaki uses for any bare 4-digit
    /// number that is not a currency amount (`en.py`:
    /// `len(word) == 4 and currency not in CURRENCIES`).
    ///
    /// Verified against Python num2words: 1999 "nineteen ninety-nine",
    /// 1900 "nineteen hundred", 2000 "two thousand", 2001 "two thousand and
    /// one", 2005 "two thousand and five", 2010 "twenty ten", 1066 "ten
    /// sixty-six", 1905 "nineteen oh-five", 1100 "eleven hundred", 2100
    /// "twenty-one hundred", 1010 "ten ten".
    fn year_words(y: i64) -> Option<String> {
        let card = |n: i64| Num2Words::new(n).to_words().ok();
        let (hi, lo) = (y / 100, y % 100);
        if lo == 0 {
            if y % 1000 == 0 {
                card(y) // 2000 -> two thousand
            } else {
                Some(format!("{} hundred", card(hi)?)) // 1900 -> nineteen hundred
            }
        } else if hi % 10 == 0 && lo < 10 {
            card(y) // 2005 -> two thousand and five
        } else if lo < 10 {
            Some(format!("{} oh {}", card(hi)?, card(lo)?)) // 1905 -> nineteen oh-five
        } else {
            Some(format!("{} {}", card(hi)?, card(lo)?)) // 1999 -> nineteen ninety-nine
        }
    }

    /// misaki's CURRENCIES: the unit and its hundredth.
    fn currency_units(sym: char) -> Option<(&'static str, &'static str)> {
        match sym {
            '$' => Some(("dollar", "cent")),
            '\u{a3}' => Some(("pound", "pence")),
            '\u{20ac}' => Some(("euro", "cent")),
            _ => None,
        }
    }

    /// misaki's `Lexicon.is_currency`.
    fn is_currency_amount(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        let cleaned = word.replace(',', "");
        let parts: Vec<&str> = cleaned.split('.').collect();
        if parts.len() > 2 {
            return false;
        }
        // Three or more cent digits → not currency (read as a plain number).
        parts.get(1).map(|c| c.len() < 3).unwrap_or(true)
    }

    /// Thin wrapper kept for unit tests that exercise spoken forms without
    /// currency / flags. Head = true, empty num_flags.
    fn currency_words(&self, sym: char, amount: &str) -> Option<String> {
        let spoken = self.get_number_spoken(amount, Some(sym), true, "")?;
        Some(spoken)
    }

    fn currency_of(tk: &MToken) -> Option<char> {
        tk.underscore().currency.as_deref().and_then(|s| s.chars().next())
    }

    /// misaki's `append_currency`: a word that carries the currency, such as
    /// "billion" in `$2 billion`, is followed by the plural unit.
    fn append_currency(&self, ps: String, sym: Option<char>) -> String {
        let unit = sym
            .and_then(Self::currency_units)
            .and_then(|(unit, _)| self.lexicon.stem_s(&format!("{unit}s"), "", None, None));
        match unit {
            Some((unit, _)) => format!("{ps} {unit}"),
            None => ps,
        }
    }

    /// Phonemes for a spoken number phrase, looked up word by word.
    ///
    /// misaki's `get_number` does not phonemise the phrase as text -- it looks
    /// each word up on its own, and gives "point" a stress of -2:
    ///
    /// ```text
    ///     result.append(self.lookup(w, None, -2 if w == 'point' else None, None))
    /// ```
    ///
    /// Re-running the whole phrase through g2p instead left "point" stressed,
    /// so 3.5 came out θɹˈi pˈYnt fˈIv where misaki gives θɹˈi pYnt fˈIv.
    fn number_phonemes(&self, spoken: &str) -> Option<String> {
        let mut out: Vec<String> = Vec::new();
        for w in spoken.split_whitespace() {
            // misaki gives both "point" and the spoken-zero "O" a stress of -2.
            let stress = if w == "point" || w == "O" { Some(-2.0) } else { None };
            let (ps, _) = self.lexicon.get_word(w, "NN", stress, None)?;
            out.push(ps);
        }
        if out.is_empty() {
            None
        } else {
            Some(out.join(" "))
        }
    }

    /// A digit group as words (cardinal), for digit-wise paths.
    fn spell(digits: &str) -> String {
        digits
            .parse::<i64>()
            .ok()
            .and_then(|n| Num2Words::new(n).to_words().ok())
            .map(|w| Self::normalize_num_words(&w))
            .unwrap_or_else(|| digits.to_string())
    }

    /// misaki `extend_num` / `get_number` — returns the spoken word string
    /// (space-separated lexicon keys), honouring `num_flags` for `and`/`a`/`n`.
    fn get_number_spoken(
        &self,
        word: &str,
        currency: Option<char>,
        is_head: bool,
        num_flags: &str,
    ) -> Option<String> {
        const ORDINALS: &[&str] = &["st", "nd", "rd", "th"];
        let (body0, suffix) = Self::peel_number_suffix(word);
        let mut body = body0.to_string();
        let mut pieces: Vec<String> = Vec::new();

        let extend_num = |num: &str, first: bool, escape: bool, pieces: &mut Vec<String>| {
            let spoken = if escape {
                num.to_string()
            } else {
                num.parse::<i64>()
                    .ok()
                    .and_then(|n| Num2Words::new(n).to_words().ok())
                    .unwrap_or_else(|| num.to_string())
            };
            let splits: Vec<&str> = spoken
                .split(|c: char| !c.is_ascii_lowercase())
                .filter(|w| !w.is_empty())
                .collect();
            for (i, w) in splits.iter().enumerate() {
                if *w != "and" || num_flags.contains('&') {
                    if first && i == 0 && splits.len() > 1 && *w == "one" && num_flags.contains('a')
                    {
                        pieces.push("ə".to_string());
                    } else {
                        pieces.push((*w).to_string());
                    }
                } else if *w == "and" && num_flags.contains('n') && !pieces.is_empty() {
                    let last = pieces.len() - 1;
                    pieces[last].push_str("ən");
                }
            }
        };

        let mut result_started = false;
        if let Some(rest) = body.strip_prefix('-').map(|s| s.to_string()) {
            pieces.push("minus".to_string());
            body = rest;
            result_started = true;
        }

        let currency_key = currency.filter(|c| Self::currency_units(*c).is_some());

        if Self::ascii_digit_str(&body) && suffix.is_some_and(|s| ORDINALS.contains(&s)) {
            let n: i64 = body.parse().ok()?;
            let spoken = Num2Words::new(n).ordinal().to_words().ok()?;
            extend_num(&spoken, true, true, &mut pieces);
        } else if !result_started
            && body.len() == 4
            && currency_key.is_none()
            && Self::ascii_digit_str(&body)
        {
            let y: i64 = body.parse().ok()?;
            let spoken = Self::year_words(y)?;
            extend_num(&spoken, true, true, &mut pieces);
        } else if !is_head && !body.contains('.') {
            let num = body.replace(',', "");
            if num.is_empty() || !Self::ascii_digit_str(&num) {
                return None;
            }
                if num.starts_with('0') || num.len() > 3 {
                for n in num.chars() {
                    extend_num(&n.to_string(), false, false, &mut pieces);
                }
            } else if num.len() == 3 && !num.ends_with("00") {
                extend_num(&num[..1], true, false, &mut pieces);
                if num.as_bytes()[1] == b'0' {
                    pieces.push("O".to_string());
                    extend_num(&num[2..], false, false, &mut pieces);
                } else {
                    extend_num(&num[1..], false, false, &mut pieces);
                }
            } else {
                extend_num(&num, true, false, &mut pieces);
            }
        } else if body.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in body.replace(',', "").split('.') {
                if num.is_empty() {
                    continue;
                }
                if !num.chars().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                if num.starts_with('0')
                    || (num.len() != 2 && num.chars().skip(1).any(|n| n != '0'))
                {
                    for n in num.chars() {
                        extend_num(&n.to_string(), false, false, &mut pieces);
                    }
                } else {
                    extend_num(num, first, false, &mut pieces);
                }
                first = false;
            }
        } else if let Some(sym) = currency_key.filter(|_| Self::is_currency_amount(&body)) {
            let units = Self::currency_units(sym)?;
            let mut pairs: Vec<(i64, &str)> = Vec::new();
            for (i, seg) in body.replace(',', "").split('.').enumerate().take(2) {
                let n: i64 = if seg.is_empty() {
                    0
                } else {
                    seg.parse().ok()?
                };
                pairs.push((n, if i == 0 { units.0 } else { units.1 }));
            }
            if pairs.len() > 1 {
                if pairs[1].0 == 0 {
                    pairs.truncate(1);
                } else if pairs[0].0 == 0 {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0 {
                    pieces.push("and".to_string());
                }
                extend_num(&num.to_string(), i == 0, false, &mut pieces);
                if num.abs() != 1 && *unit != "pence" {
                    pieces.push(format!("{unit}s"));
                } else {
                    pieces.push((*unit).to_string());
                }
            }
        } else {
            let spoken = if Self::ascii_digit_str(&body) {
                let n: i64 = body.parse().ok()?;
                Num2Words::new(n).to_words().ok()?
            } else if !body.contains('.') {
                let n: i64 = body.replace(',', "").parse().ok()?;
                if suffix.is_some_and(|s| ORDINALS.contains(&s)) {
                    Num2Words::new(n).ordinal().to_words().ok()?
                } else {
                    Num2Words::new(n).to_words().ok()?
                }
            } else {
                let cleaned = body.replace(',', "");
                if let Some(frac) = cleaned.strip_prefix('.') {
                    let mut s = String::from("point");
                    for n in frac.chars() {
                        let w = Num2Words::new((n as u8 - b'0') as i64)
                            .to_words()
                            .ok()?;
                        s.push(' ');
                        s.push_str(&w);
                    }
                    s
                } else {
                    // float path — num2words(float) trims trailing zeros.
                    // The Rust crate omits a leading "zero" for |x|<1; Python
                    // num2words always keeps it (`0.05` → "zero point zero five").
                    let f: f64 = cleaned.parse().ok()?;
                    let mut spoken = Num2Words::new(f).to_words().ok()?;
                    if cleaned.starts_with("0.") && !spoken.starts_with("zero") {
                        spoken = format!("zero {spoken}");
                    }
                    spoken
                }
            };
            extend_num(&spoken, true, true, &mut pieces);
        }

        if pieces.is_empty() {
            return None;
        }
        Some(pieces.join(" "))
    }

    /// Phonemes for `Lexicon.get_number` (spoken lookup + optional ending).
    fn get_number_phonemes(
        &self,
        word: &str,
        currency: Option<char>,
        is_head: bool,
        num_flags: &str,
    ) -> Option<String> {
        let (_, suffix) = Self::peel_number_suffix(word);
        let spoken = self.get_number_spoken(word, currency, is_head, num_flags)?;
        let mut out: Vec<String> = Vec::new();
        for w in spoken.split_whitespace() {
            if w == "ə" {
                out.push("ə".to_string());
                continue;
            }
            let stress = if w == "point" || w == "O" {
                Some(-2.0)
            } else {
                None
            };
            let (ps, _) = self.lexicon.get_word(w, "NN", stress, None)?;
            out.push(ps);
        }
        if out.is_empty() {
            return None;
        }
        let joined = out.join(" ");
        match suffix {
            Some("ing") => self.lexicon.append_ing(&joined),
            Some("ed") | Some("'d") => Some(self.lexicon.append_ed(&joined)),
            Some("s") | Some("'s") => Some(self.lexicon.append_s(&joined)),
            _ => Some(joined),
        }
    }

    /// Span-loop helper: number with ending / ordinal as one piece.
    fn number_with_ending(&self, text: &str, _span: &[MToken], is_head: bool) -> Option<String> {
        if !self.is_number(text, is_head) {
            return None;
        }
        self.get_number_phonemes(text, None, is_head, "")
    }

    /// Legacy spoken-form helper for unit tests (head, no currency, no flags).
    fn convert_number(&self, word: &str) -> String {
        self.get_number_spoken(word, None, true, "")
            .unwrap_or_else(|| word.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestFallback;

    impl Fallback for TestFallback {
        fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
            assert_eq!(word, "qzxwv");
            Ok("test-phonemes".to_string())
        }
    }

    struct HyphenatedFallback;

    impl Fallback for HyphenatedFallback {
        fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
            assert_eq!(word, "il-lilly");
            Ok("whole-word-fallback".to_string())
        }
    }

    #[test]
    fn test_g2p_basic() {
        let _ = env_logger::try_init();
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("Hello, world!").unwrap();
        println!("Phonemes: {}", phonemes);
        assert!(!phonemes.contains("❓"));
    }

    #[test]
    fn test_dotted_number_normalization() {
        let g2p = G2P::new(Language::EnglishUS);
        assert!(g2p.is_number_like("0.12.1"));
        assert!(g2p.is_number_like(".5"));
        assert!(g2p.is_number_like("-0.12.1"));
        assert!(!g2p.is_number_like("config.toml"));
        assert_eq!(g2p.convert_number("0.12.1"), "zero twelve one");
        assert_eq!(g2p.convert_number("1.100.0"), "one one hundred zero");
        assert_eq!(g2p.convert_number("1.2"), "one point two");
        assert_eq!(g2p.convert_number("0.05"), "zero point zero five");
        assert_eq!(g2p.convert_number("1.200"), "one point two");
        assert_eq!(g2p.convert_number("1.0"), "one");
        assert_eq!(g2p.convert_number("1,234"), "one thousand two hundred thirty four");
        assert_eq!(g2p.convert_number(".5"), "point five");
        assert_eq!(g2p.convert_number("2026"), "twenty twenty six");
        assert_eq!(g2p.convert_number("-40"), "minus forty");
        assert_eq!(g2p.convert_number("23rd"), "twenty third");
        assert_eq!(g2p.currency_words('\u{a3}', "1.50").as_deref(), Some("one pound and fifty pence"));
    }

    #[test]
    fn test_dotted_number_is_normalized_in_g2p_pipeline() {
        let g2p = G2P::new(Language::EnglishUS);
        let (version, _) = g2p.g2p("Version 0.12.1").unwrap();
        let (words, _) = g2p.g2p("Version zero twelve one").unwrap();
        let normalize = |text: String| text.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(normalize(version), normalize(words));
    }

    // Expected strings are misaki 0.9.4's output.
    #[test]
    fn test_plus_sign_before_a_number_is_read() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            (
                "Call +1 555 123 4567.",
                "kˈɔl plˈʌs wˈʌn fˈIv hˈʌndɹəd fˈɪfti fˈIv wˈʌn hˈʌndɹəd twˈɛnti θɹˈi fˈɔɹTi fˈIv sˈɪksti sˈɛvən.",
            ),
            ("It is +5 degrees.", "ˌɪt ɪz plˈʌs fˈIv dəɡɹˈiz."),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want);
        }
    }

    #[test]
    fn test_is_number_matches_misaki_digit_body() {
        let g2p = G2P::new(Language::EnglishUS);
        // Digits, commas, points, leading minus — not a leading plus.
        assert!(g2p.is_number("1", true));
        assert!(g2p.is_number("1,000", true));
        assert!(g2p.is_number("-40", true));
        assert!(g2p.is_number("3.5", true));
        assert!(!g2p.is_number("+1", true));
        assert!(!g2p.is_number("+5", true));
        assert!(!g2p.is_number("", true));
        assert!(!g2p.is_number("abc", true));
        // Beyond i64: misaki still treats this as a number body.
        assert!(g2p.is_number("99999999999999999999", true));
        // Case-sensitive ordinals: `5TH` is not an ordinal ending.
        assert!(g2p.is_number("5th", true));
        assert!(!g2p.is_number("5TH", true));
    }

    #[test]
    fn test_with_fallback_uses_injected_fallback_for_oov_word() {
        let g2p = G2P::with_fallback(Language::EnglishUS, Some(Box::new(TestFallback)));

        let (phonemes, tokens) = g2p.g2p("qzxwv").unwrap();

        assert_eq!(phonemes.trim(), "test-phonemes");
        assert_eq!(tokens[0].phonemes.as_deref(), Some("test-phonemes"));
    }

    #[test]
    fn test_hyphenated_oov_preserves_unknown_marker_without_fallback() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        let (phonemes, tokens) = g2p.g2p("il-lilly").unwrap();

        assert_eq!(phonemes, "❓lˈɪli");
        assert_eq!(tokens[0].phonemes.as_deref(), Some("❓lˈɪli"));
    }

    #[test]
    fn test_hyphenated_oov_fallback_receives_merged_word() {
        let g2p = G2P::with_fallback(Language::EnglishUS, Some(Box::new(HyphenatedFallback)));
        let (phonemes, tokens) = g2p.g2p("il-lilly").unwrap();

        assert_eq!(phonemes, "whole-word-fallback");
        assert_eq!(tokens[0].phonemes.as_deref(), Some("whole-word-fallback"));
    }

    /// Records every word it is asked, and answers with a fixed marker.
    struct RecordingFallback {
        asked: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Fallback for RecordingFallback {
        fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
            self.asked.lock().unwrap().push(word.to_string());
            Ok("FALLBACK".to_string())
        }
    }

    type Pairs = Vec<(String, Option<String>)>;

    /// Phonemes, per-token pairs and the words the fallback was asked.
    fn g2p_recording(text: &str) -> (String, Pairs, Vec<String>) {
        let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let fallback = RecordingFallback {
            asked: asked.clone(),
        };
        let mut g2p = G2P::with_fallback(Language::EnglishUS, Some(Box::new(fallback)));
        g2p.unk = String::new();
        let (ps, tokens) = g2p.g2p(text).unwrap();
        let pairs = tokens
            .iter()
            .map(|t| (t.text.clone(), t.phonemes.clone()))
            .collect();
        let asked = asked.lock().unwrap().clone();
        (ps, pairs, asked)
    }

    // Expected strings in both tests are misaki 0.9.4's output.
    #[test]
    fn test_multi_piece_word_with_phonemes_skips_fallback() {
        let mut none = G2P::with_fallback(Language::EnglishUS, None);
        none.unk = String::new();
        for (text, want) in [
            (
                "The download is 37% done.",
                "ðə dˈWnlˌOd ɪz θˈɜɹTi sˈɛvən pəɹsˈɛnt dˈʌn.",
            ),
            (
                "It was modified on 11/17/2026.",
                "ˌɪt wʌz mˈɑdəfˌId ˌɔn əlˈɛvᵊn sˌɛvəntˈin twˈɛnti twˈɛnti sˈɪks.",
            ),
        ] {
            let (ps, pairs, asked) = g2p_recording(text);
            let (none_ps, none_tokens) = none.g2p(text).unwrap();
            let none_pairs: Pairs = none_tokens
                .iter()
                .map(|t| (t.text.clone(), t.phonemes.clone()))
                .collect();

            assert!(asked.is_empty(), "{text:?} asked the fallback {asked:?}");
            assert_eq!(ps, want);
            assert_eq!(ps, none_ps);
            assert_eq!(pairs, none_pairs);
        }
    }

    #[test]
    fn test_multi_piece_word_without_phonemes_reaches_fallback() {
        for (text, word, want) in [
            ("Press Ctrl+S to save.", "Ctrl+S", "pɹˈɛs FALLBACK tə sˈAv."),
            (
                "Your meeting is at 3:45 PM.",
                "3:45",
                "jˌʊɹ mˈiTɪŋ ɪz æt FALLBACK pˌiˈɛm.",
            ),
        ] {
            let (ps, _, asked) = g2p_recording(text);

            assert_eq!(asked, [word]);
            assert_eq!(ps, want);
        }
    }

    // Expected strings are misaki 0.9.4's output.
    #[test]
    fn test_currency_goes_after_the_last_number() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("It costs $2 billion.", "ˌɪt kˈɔsts tˈu bˈɪljən dˈɑləɹz."),
            ("They raised $1.5 million.", "ðˌA ɹˈAzd wˈʌn pYnt fˈIv mˈɪljᵊn dˈɑləɹz."),
            ("It costs £20 million.", "ˌɪt kˈɔsts twˈɛnti mˈɪljᵊn pˈWndz."),
            ("It costs $2billion.", "ˌɪt kˈɔsts tˈu bˈɪljən dˈɑləɹz."),
            ("It costs $3/4.", "ˌɪt kˈɔsts θɹˈi fˈɔɹ."),
            ("It costs $9 and 5 more.", "ˌɪt kˈɔsts nˈIn dˈɑləɹz ænd fˈIv mˈɔɹ."),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want);
        }
    }

    #[cfg(feature = "espeak")]
    #[test]
    fn test_hyphenated_oov_matches_espeak_reference() {
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, tokens) = g2p.g2p("il-lilly").unwrap();

        assert_eq!(phonemes, "ˈɪllˈɪli");
        assert_eq!(tokens[0].phonemes.as_deref(), Some("ˈɪllˈɪli"));
    }

    // Tests for `[text](feature)` markdown-link parsing — see
    // misaki/en.py::Lexicon.preprocess for the reference implementation.

    #[test]
    fn test_preprocess_links_strips_markup() {
        let g2p = G2P::new(Language::EnglishUS);
        let (stripped, markers) = g2p.preprocess_links("say [hello](+2) please");
        assert_eq!(stripped, "say hello please");
        assert_eq!(markers.len(), 1);
        let m = &markers[0];
        assert_eq!(&stripped[m.byte_start..m.byte_end], "hello");
        match m.feature {
            LinkFeature::Stress(n) => assert!((n - 2.0).abs() < f64::EPSILON),
            _ => panic!("expected Stress, got {:?}", m.feature),
        }
    }

    #[test]
    fn test_preprocess_links_parses_features() {
        // Integer stress — signed and unsigned.
        assert!(matches!(parse_link_feature("+2"), Some(LinkFeature::Stress(n)) if n == 2.0));
        assert!(matches!(parse_link_feature("-1"), Some(LinkFeature::Stress(n)) if n == -1.0));
        assert!(matches!(parse_link_feature("2"),  Some(LinkFeature::Stress(n)) if n == 2.0));
        // Half stress — only these three literals are accepted upstream.
        assert!(matches!(parse_link_feature("0.5"),  Some(LinkFeature::Stress(n)) if n == 0.5));
        assert!(matches!(parse_link_feature("+0.5"), Some(LinkFeature::Stress(n)) if n == 0.5));
        assert!(matches!(parse_link_feature("-0.5"), Some(LinkFeature::Stress(n)) if n == -0.5));
        // Phoneme override.
        assert!(matches!(parse_link_feature("/həˈloʊ/"), Some(LinkFeature::Phonemes(ref p)) if p == "həˈloʊ"));
        // Num flags.
        assert!(matches!(parse_link_feature("#cardinal#"), Some(LinkFeature::NumFlags(ref f)) if f == "cardinal"));
        // Anything else → None, matching the upstream `else: f = None` branch.
        assert!(parse_link_feature("").is_none());
        assert!(parse_link_feature("=").is_none());
        assert!(parse_link_feature("doctor").is_none());
        assert!(parse_link_feature("2.5").is_none());
        assert!(parse_link_feature("/foo").is_none());
        assert!(parse_link_feature("foo/").is_none());
    }

    // Expected strings are misaki 0.9.4's output.
    #[test]
    fn test_three_cent_digits_read_as_a_number() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("$1.555", "wˈʌn pYnt fˈIv fˈIv fˈIv"),
            ("It cost $1.555 each.", "ˌɪt kˈɔst wˈʌn pYnt fˈIv fˈIv fˈIv ˈiʧ."),
            ("£2.125", "tˈu pYnt wˈʌn tˈu fˈIv"),
            ("$1.55", "wˈʌn dˈɑləɹ ænd fˈɪfti fˈIv sˈɛnts"),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want);
        }
    }

    #[test]
    fn test_link_stress_promotes() {
        // `[hello](+2)` should phonemize successfully and emit primary stress.
        let g2p = G2P::new(Language::EnglishUS);
        let (boosted, _) = g2p.g2p("[hello](+2)").unwrap();
        assert!(!boosted.contains("❓"), "boosted output: '{}'", boosted);
        assert!(boosted.contains('ˈ'),
            "boosted output missing primary stress: '{}'", boosted);
    }

    #[test]
    fn test_link_stress_strips() {
        // `(-2)` strips all stress marks via Lexicon::apply_stress.
        let g2p = G2P::new(Language::EnglishUS);
        let (stripped, _) = g2p.g2p("[hello](-2)").unwrap();
        assert!(!stripped.contains("❓"));
        assert!(!stripped.contains('ˈ') && !stripped.contains('ˌ'),
            "expected no stress marks, got: '{}'", stripped);
    }

    #[test]
    fn test_link_phoneme_override() {
        // `/.../` content is emitted verbatim, bypassing the lexicon.
        let g2p = G2P::new(Language::EnglishUS);
        let (out, _) = g2p.g2p("[xyzzy](/həˈloʊ/)").unwrap();
        assert!(out.contains("həˈloʊ"), "phoneme override missing: '{}'", out);
        assert!(!out.contains("❓"), "got unknown marker: '{}'", out);
    }

    #[test]
    fn test_link_num_flags_sets_field() {
        // `#flag#` markers don't affect phonemes today (the number-spelling
        // logic that reads `num_flags` isn't ported yet), but the field must
        // be set on the token so a future port can consume it.
        let g2p = G2P::new(Language::EnglishUS);
        let (_, tokens) = g2p.g2p("[5](#cardinal#)").unwrap();
        let tagged: Vec<_> = tokens
            .iter()
            .filter(|t| !t.underscore().num_flags.is_empty())
            .collect();
        assert_eq!(tagged.len(), 1, "expected one token with num_flags set");
        assert_eq!(tagged[0].underscore().num_flags, "cardinal");
    }

    #[test]
    fn test_link_unknown_feature_is_dropped() {
        // `=` and bare words must not parse — they fall through to None and
        // the marker is silently ignored, matching upstream behaviour.
        let g2p = G2P::new(Language::EnglishUS);
        let (stripped, markers) = g2p.preprocess_links("say [hello](=) please");
        assert_eq!(stripped, "say hello please");
        assert!(markers.is_empty(), "unknown feature should yield no marker");
    }

    // Expected strings are misaki 0.9.4's output (stock oracle).
    #[cfg(not(feature = "fixes"))]
    #[test]
    fn test_full_stop_after_ampersand_word() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("Call AT&T.", "kˈɔl ˈættˈi."),
            ("AT&T. Then more.", "ˈættˈi. ðˈɛn mˈɔɹ."),
            ("Ask H&M.", "ˈæsk ˈAʧ ænd ˈɛm"),
            ("R&D. Then more.", "ˈɑɹ ænd dˈi ðˈɛn mˈɔɹ."),
            ("I met J. Smith.", "ˌI mˈɛt ʤˈA smˈɪθ."),
            ("The Q&A.", "ðə kjˈu ænd ˈA"),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want);
        }
    }

    // Expected strings are misaki 0.9.4's output.
    #[test]
    fn test_degree_unit_keeps_the_full_stop() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("It is 20 °C.", "ˌɪt ɪz twˈɛnti ❓ sˈi."),
            ("It is 72 °F.", "ˌɪt ɪz sˈɛvənti tˈu ❓ ˈɛf."),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want);
        }
    }

    /// With espeak feature: "eBook" is phonemized by espeak fallback (not in lexicon).
    #[test]
    #[cfg(feature = "espeak")]
    fn test_ebook_with_espeak() {
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("eBook").unwrap();
        assert!(
            !phonemes.contains("❓"),
            "with espeak: 'eBook' should be phonemized by fallback, got: {}",
            phonemes
        );
        // Should not be spelled out letter-by-letter (e.g. no ˈi for 'e' as letter)
        assert!(
            !phonemes.contains("ˈɛl"),
            "with espeak: 'eBook' should not spell out letters, got: {}",
            phonemes
        );
        println!("eBook (with espeak): {}", phonemes);
    }

    /// Without espeak feature: "eBook" is OOV so we get unknown marker or character spelling.
    #[test]
    #[cfg(not(feature = "espeak"))]
    fn test_ebook_without_espeak() {
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("eBook").unwrap();
        // No fallback: either ❓ for unknown or character-by-character spelling
        assert!(
            phonemes.contains("❓") || phonemes.contains("ˈi") || phonemes.contains("b"),
            "without espeak: 'eBook' should show unknown or spelled form, got: {}",
            phonemes
        );
        println!("eBook (without espeak): {}", phonemes);
    }

    // Expected strings are misaki 0.9.4's output.
    #[test]
    fn test_unknown_marker_inside_a_word() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("Read pages 5–7.", "ɹˈid pˈAʤᵻz fˈIv❓ sˈɛvən."),
            ("Set it to 60Hz.", "sˈɛt ɪt tə sˈɪksti❓."),
            ("Press ⇧⌘4.", "pɹˈɛs ❓❓ fˈɔɹ."),
            ("Dana's TikTok-ish", "❓s tˈɪk❓ˌɪʃ"),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want);
        }
    }

    // #[test]
    // fn test_g2p_italian() {
    //     let g2p = G2P::new(Language::Italian);
    //     let (phonemes, _) = g2p.g2p("Ciao, mondo!");
    //     println!("Phonemes: {}", phonemes);
    //     // "ciao" -> c+i+a+o -> tʃ+a+o -> with stress tʃˈao
    //     // "mondo" -> m+o+n+d+o -> mˈondo
    //     assert!(phonemes.contains("tʃ") && phonemes.contains("ao"));
    //     assert!(phonemes.contains("mondo"));
    // }

    // #[test]
    // fn test_convert_number_italian() {
    //     let g2p = G2P::new(Language::Italian);
    //     let (phonemes, _) = g2p.g2p("42");
    //     println!("Phonemes for 42: {}", phonemes);
    //     // 42 in Italian is "quarantadue" -> kwarantadue
    //     // We relax the check to ensure it produces phonemes and not numbers/unknowns
    //     assert!(!phonemes.contains("42"));
    //     assert!(!phonemes.contains("❓"));
    //     assert!(phonemes.contains("kwaranta") || phonemes.contains("due"));
    // }

    #[test]
    fn test_english_abbreviations() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "I'll",
            "I've",
            "it's",
            "he's",
            "she's",
            "we're",
            "they're",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "don't",
            "doesn't",
            "didn't",
            "can't",
            "couldn't",
            "shouldn't",
            "wouldn't",
            "won't",
            "hasn't",
            "haven't",
            "hadn't",
            "let's",
            "that's",
            "what's",
            "who's",
            "here's",
            "there's",
            "where's",
            "how's",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            assert!(!p.contains("❓"), "Failed for '{}'", text);
        }
    }

    #[test]
    fn test_casing_and_special_chars() {
        let g2p = G2P::new(Language::EnglishUS);

        // Test 1: All caps with suffix
        let (playing, _) = g2p.g2p("PLAYING").unwrap();
        println!("PLAYING: {}", playing);
        assert!(
            !playing.contains("❓"),
            "PLAYING should be resolved, got: {}",
            playing
        );

        // Test 2: Contractions
        let (ive, _) = g2p.g2p("I've").unwrap();
        println!("I've: {}", ive);
        assert!(!ive.contains("❓"), "I've should be resolved, got: {}", ive);

        // Test 3: Dashes
        // em-dash — (U+2014) and hyphen -
        let (dash, _) = g2p.g2p("word - word — word").unwrap();
        println!("Dash: {}", dash);
        // The two dashes are NOT alike. spaCy tags the em-dash `:`, which puts
        // it in PUNCT_TAGS and renders it —; it tags the plain hyphen HYPH,
        // which is not punctuation, so the lexicon misses it.
        //
        // misaki 0.9.4 (verified):
        //   fallback=None   -> wˈɜɹd ❓ wˈɜɹd — wˈɜɹd  (HYPH stays unresolved)
        //   EspeakFallback  -> wˈɜɹd  wˈɜɹd — wˈɜɹd   (espeak returns "")
        //
        // G2P::new enables espeak when the feature is on, so the expected
        // string depends on that.
        assert!(dash.contains('—'), "em-dash dropped: {dash}");
        #[cfg(feature = "espeak")]
        assert_eq!(
            dash, "wˈɜɹd  wˈɜɹd — wˈɜɹd",
            "does not match misaki 0.9.4 with espeak fallback"
        );
        #[cfg(not(feature = "espeak"))]
        assert_eq!(
            dash, "wˈɜɹd ❓ wˈɜɹd — wˈɜɹd",
            "does not match misaki 0.9.4 with fallback=None"
        );
    }

    #[test]
    fn test_kokoros_basic() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "hello",
            "world",
            "the quick brown fox",
            "testing phonemization",
            "Hello, world!",
            "123",
            "restriction",
            "restrictions",
            "",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            if !text.is_empty() {
                assert!(!p.is_empty(), "Failed for '{}'", text);
            }
        }
    }

    #[test]
    fn test_kokoros_numbers() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "CHAPTER XIV",
            "CHAPTER 14",
            "CHAPTER 123",
            "I have 5 apples and 42 oranges",
            "The year 2024",
            "1234567890",
            "CHAPTER I",
            "CHAPTER II",
            "CHAPTER III",
            "CHAPTER IV",
            "CHAPTER V",
            "CHAPTER X",
            "CHAPTER XX",
            "CHAPTER XXX",
            "In 2024, CHAPTER XIV had 42 pages.",
            "The price is $123.45",
            "Temperature: -5°C",
            "Score: 100%",
            "Version 2.0",
            "3.14159",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            assert!(!p.is_empty(), "Failed for '{}'", text);
        }
    }

    // Expected strings are misaki 0.9.4's output.
    #[test]
    fn test_number_with_an_ending() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("Music from the 1990s.", "mjˈuzɪk fɹʌm ðə nˌIntˈin nˈIndiz."),
            ("Music from the 1990's.", "mjˈuzɪk fɹʌm ðə nˌIntˈin nˈIndiz."),
            ("The '90s were fun.", "ðə nˈIndiz wɜɹ fˈʌn."),
            ("In your 30s.", "ɪn jʊɹ θˈɜɹTiz."),
            ("PS5s sold out.", "pˌiˈɛs fˈIvz sˈOld ˈWt."),
            ("He 86ed it.", "hˌi ˈATi sˈɪkst ɪt."),
            ("It was 100ing.", "ˌɪt wʌz wˈʌn hˈʌndɹədɪŋ."),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want);
        }
    }

    #[test]
    fn test_kokoros_utf8_and_special() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "café",
            "naïve",
            "résumé",
            "Zürich",
            "São Paulo",
            "Müller",
            "北京",
            "こんにちは",
            "Здравствуй",
            "مرحبا",
            "🎉🎊🎈",
            // Control chars
            "\x00\x01\x02",
            // Mixed scripts
            "Hello 世界",
            "123中文",
            "English123中文",
            // Zero-width characters
            "hello\u{200B}world", // zero-width space
            "hello\u{200C}world", // zero-width non-joiner
            "hello\u{200D}world", // zero-width joiner
            // Combining characters
            "caf\u{00E9}",  // é as combining character
            "na\u{00EF}ve", // ï as combining character
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            // Some might be empty/unknown depending on handling, but shouldn't crash
        }
    }

    #[test]
    fn test_kokoros_punctuation() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "Hello—world", // em dash
            "Hello–world", // en dash
            "Hello…world", // ellipsis
            "\"quoted text\"",
            "'single quotes'",
            "«French quotes»",
            "„German quotes„",
            "「Japanese quotes」",
            "Dr. Smith",
            "Mr. Jones",
            "Mrs. Brown",
            "Ms. Davis",
            "etc.",
            "U.S.A.",
            "Ph.D.",
            "A.I.",
            "NASA",
            "FBI",
            "   ",
            "\n\n",
            "\t\t",
            "\r\n",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
        }
    }

    #[test]
    fn test_kokoros_punctuation_preserved() {
        // Kokoro's vocab encodes ";:,.!?" as distinct token ids (1-6).
        // G2P must emit each character literally in the phoneme stream so
        // downstream tokenizers can map them; collapsing to space loses the
        // sentence-end / clause-breath / question-intonation cues the
        // synthesis model relies on. Other ASCII punctuation (parens,
        // dashes, ellipsis, …) keeps the prior single-space fallback.
        let g2p = G2P::new(Language::EnglishUS);

        let (phonemes, _) = g2p.g2p("Hello, world.").unwrap();
        assert!(phonemes.contains(','), "comma dropped: {phonemes:?}");
        assert!(phonemes.contains('.'), "period dropped: {phonemes:?}");

        for c in [';', ':', '!', '?'] {
            let input = format!("test {c}");
            let (out, _) = g2p.g2p(&input).unwrap();
            assert!(out.contains(c), "{c:?} dropped: {out:?}");
        }

        // Parentheses ARE preserved. This assertion used to require the
        // opposite, which was the pre-fork behaviour: misaki maps spaCy's
        // -LRB-/-RRB- tags to '(' and ')' via PUNCT_TAG_PHONEMES
        // (`en.py:598-599`), and misaki 0.9.4 gives
        //
        //     "hello (world)" -> həlˈO (wˈɜɹld)
        //
        // verified against the oracle. The gate is misaki's output, so the test
        // follows it rather than the other way round.
        let (out, _) = g2p.g2p("hello (world)").unwrap();
        assert!(out.contains('('), "( dropped: {out:?}");
        assert!(out.contains(')'), ") dropped: {out:?}");
    }

    #[test]
    fn test_kokoros_long_text() {
        let g2p = G2P::new(Language::EnglishUS);
        // Reduced to 100 to check if it crashes
        let long_text = "a".repeat(1000);
        let (p, _) = g2p.g2p(&long_text).unwrap();
        assert!(!p.is_empty());
    }
}
