use crate::fallback::{Fallback, FallbackError};
use crate::language::Language;
#[cfg(feature = "espeak")]
use crate::fallback::EspeakFallback;
use crate::languages::{LanguageRules, english::English};
use crate::lexicon::Lexicon;
use thiserror::Error;
use crate::tagger::PerceptronTagger;
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
    /// The perceptron tagger this crate shipped with. No longer used for
    /// pronunciation -- `spacy_tagger` is, because misaki's output depends on
    /// spaCy's exact tags -- but kept so the two remain comparable.
    #[allow(dead_code)]
    tagger: PerceptronTagger,
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

        let weights_json = include_str!("resources/tagger/weights.json");
        let classes_txt = include_str!("resources/tagger/classes.txt");
        let tags_json = include_str!("resources/tagger/tags.json");

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
            tagger: PerceptronTagger::new(weights_json, classes_txt, tags_json),
            // The ported spaCy tagger. Kept alongside the perceptron rather
            // than replacing it so the old one stays available for comparison.
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
        let tag = group
            .iter()
            .max_by_key(|tk| weight(&tk.text))
            .map(|tk| tk.tag.clone())
            .unwrap_or_default();

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
    /// This is what lets a finer tokenizer than spaCy's stay correct: `Mr` `.`
    /// and `It` `’` `s` and `five` `-` `storied` are each adjacent runs, so they
    /// merge back into `Mr.`, `It’s` and `five-storied`. Returns index ranges
    /// into `tokens`.
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
            // tk.phonemes is not None`, so an aliased token stands alone too.
            tk.underscore().alias.is_some() || Self::is_punct_token(tk)
        };

        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        let mut open = false; // is the last group still extensible?
        for i in 0..tokens.len() {
            if is_standalone(&tokens[i], i + 1 == tokens.len()) {
                // ... unless joining it to the open group spells an
                // abbreviation spaCy would never have split. Our tokenizer
                // splits on the period unconditionally, so `Mr` + `.` must be
                // put back together; spaCy's own exception list decides.
                let rejoin = tokens[i].text == "." && open;
                if rejoin {
                    let g = groups.last().unwrap().clone();
                    let mut candidate = String::new();
                    for tk in &tokens[g.clone()] {
                        candidate.push_str(&tk.text);
                        candidate.push_str(&tk.whitespace);
                    }
                    candidate.push('.');
                    if crate::abbrev::is_period_abbreviation(candidate.trim()) {
                        let g = groups.last_mut().unwrap();
                        *g = g.start..i + 1;
                        open = tokens[i].whitespace.is_empty();
                        continue;
                    }
                }
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

    /// misaki's lookup-key normalisation (`Lexicon.__call__`): curly single
    /// quotes become straight ones, then NFKC. The lexicons are keyed with
    /// straight apostrophes.
    ///
    /// NFKC is not applied wholesale, but the trademark sign is expanded to
    /// its compatibility form because it is a common standalone lexicon key.
    fn normalize_lookup(text: &str) -> String {
        text.replace('\u{2122}', "TM")
            .replace('\u{2018}', "'")
            .replace('\u{2019}', "'")
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
        let mut pos = 0usize;
        for (si, st) in spacy_tokens.iter().enumerate() {
            let (stext, sws) = (&st.text, &st.whitespace);
            let tag = group_tags.get(si).copied().unwrap_or("NN").to_string();
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

                let mut first_in_span = true;
                for marker in &markers {
                    if start >= marker.byte_start && end <= marker.byte_end {
                        match &marker.feature {
                            LinkFeature::Stress(st) => {
                                tk.underscore_mut().stress = Some(*st);
                            }
                            LinkFeature::Phonemes(ph) => {
                                // Only the head subtoken takes the override;
                                // the rest are blanked so the loop skips them
                                // (`phonemes = v.lstrip('/') if i == 0 else ''`).
                                tk.phonemes =
                                    Some(if first_in_span { ph.clone() } else { String::new() });
                                first_in_span = false;
                            }
                            LinkFeature::NumFlags(f) => {
                                tk.underscore_mut().num_flags = f.clone();
                            }
                        }
                    }
                }
                rebuilt.push(tk);
                tags.push(tag.clone());
            }
        }
        tokens = rebuilt;

        log::debug!(
            "g2p '{}' -> {} tokens, {} tags",
            text,
            tokens.len(),
            tags.len()
        );
        for (i, tk) in tokens.iter().enumerate() {
            log::debug!("token[{}]: '{}'", i, tk.text);
        }

        // Process tokens in reverse order (like Python) to build context
        let mut contexts: Vec<crate::lexicon::TokenContext> =
            vec![crate::lexicon::TokenContext::default(); tokens.len()];

        // First, set tags
        for (tk, tag) in tokens.iter_mut().zip(tags.iter()) {
            tk.tag = tag.clone();
        }

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

        // misaki's currency state machine (`retokenize`):
        //
        //     elif tk.tag == '$' and tk.text in CURRENCIES:
        //         currency = tk.text; tk.phonemes = ''; tk._.rating = 4
        //     ...
        //     elif j+1 == len(tks) and (i+1 == len(tokens) or tokens[i+1].tag != 'CD'):
        //         tk._.currency = currency
        //
        // The symbol becomes its own silent token and the unit is spoken by the
        // number after it: `$9` is [('$', ''), ('9', 'nˈIn dˈɑləɹz')]. Measured,
        // it reaches only the IMMEDIATELY following number -- `$9 and 5 more`
        // leaves the 5 bare.
        //
        // This has to be a separate pass because spaCy tokenizes the symbol
        // away from the digits, so the two are never in one group.
        let mut currency_of: Vec<Option<char>> = vec![None; tokens.len()];
        {
            let mut pending: Option<char> = None;
            for i in 0..tokens.len() {
                let c1 = {
                    let t = &tokens[i].text;
                    if t.chars().count() == 1 { t.chars().next() } else { None }
                };
                if tokens[i].tag == "$" {
                    if let Some(c) = c1 {
                        if matches!(c, '$' | '\u{a3}' | '\u{20ac}') {
                            pending = Some(c);
                            tokens[i].phonemes = Some(String::new());
                            continue;
                        }
                    }
                }
                if let Some(c) = pending.take() {
                    currency_of[i] = Some(c);
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
                continue;
            }
            if Self::is_punct_token(&tokens[i]) {
                let tg = tokens[i].tag.clone();
                tokens[i].phonemes = Some(Self::punct_phonemes(&tokens[i].text, &tg));
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
            let word = match tokens[i].underscore().alias.clone() {
                Some(a) => a,
                None => Self::normalize_lookup(&tokens[i].text),
            };
            let tag = tokens[i].tag.clone();
            // An explicit `[word](+N)` marker takes precedence over
            // capitalization-derived stress, matching Python misaki which
            // writes `tk._.stress = v` before the lexicon lookup runs.
            let stress = tokens[i].underscore().stress.or_else(|| {
                if word == word.to_lowercase() {
                    None
                } else if word == word.to_uppercase() {
                    Some(self.lexicon.cap_stresses.1)
                } else {
                    Some(self.lexicon.cap_stresses.0)
                }
            });

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
                if let Some((ps, _)) = self.lexicon.get_word(&word, &tag, stress, ctx) {
                    tokens[i].phonemes = Some(ps);
                }

                if tokens[i].phonemes.is_none() {
                    // The number path comes FIRST. misaki has no
                    // hyphen-splitting branch at all -- it is an invention of
                    // this port -- and letting it win meant a signed number was
                    // split on its own sign: `-40` became "" + "forty" and the
                    // minus was simply lost, where misaki says mˈInəs fˈɔɹTi.
                    if self.is_number_like(&word) && word.starts_with('-') {
                        let spoken = self.convert_number(&word);
                        if spoken != word {
                            if let Some(p) = self.number_phonemes(&spoken) {
                                tokens[i].phonemes = Some(p);
                            } else if let Ok((p, _)) = self.g2p(&spoken) {
                                tokens[i].phonemes = Some(p.trim_end().to_string());
                            }
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
                    } else if let Some(spoken) = currency_of[i]
                        .filter(|_| {
                            let c = word.replace(',', "");
                            !c.is_empty() && c.chars().all(|ch| ch.is_ascii_digit() || ch == '.')
                        })
                        .and_then(|sym| self.currency_words(sym, &word))
                    {
                        // A number carrying a currency from the symbol before
                        // it. This is handled here as well as in the merge pass
                        // because spaCy puts the symbol in its own token, so a
                        // bare `$9` never forms a multi-token group at all.
                        if let Some(p) = self.number_phonemes(&spoken) {
                            tokens[i].phonemes = Some(p);
                        } else if let Ok((p, _)) = self.g2p(&spoken) {
                            tokens[i].phonemes = Some(p.trim_end().to_string());
                        }
                    } else if let Some(spoken) = (!is_head[i])
                        .then(|| self.nonhead_number(&word))
                        .flatten()
                    {
                        // A numeric group that is not the head of its word.
                        if let Some(p) = self.number_phonemes(&spoken) {
                            tokens[i].phonemes = Some(p);
                        } else if let Ok((p, _)) = self.g2p(&spoken) {
                            tokens[i].phonemes = Some(p.trim_end().to_string());
                        }
                    } else if self.is_number_like(&word) {
                        let spoken = self.convert_number(&word);
                        if spoken != word {
                            // Word by word, so "point" keeps its -2 stress.
                            if let Some(p) = self.number_phonemes(&spoken) {
                                tokens[i].phonemes = Some(p);
                            } else {
                                let (p, _) = self.g2p(&spoken)?;
                                tokens[i].phonemes = Some(p.trim_end().to_string());
                            }
                        }
                    }
                }

                if tokens[i].phonemes.is_none() {
                    if let Some(ps) = self.rules.apply_rules(&word, &tag, &self.lexicon) {
                        tokens[i].phonemes = Some(ps);
                    }
                }

                if tokens[i].phonemes.is_none() {
                    if word.chars().count() > 1 {
                        // Python applies fallback after resolving the whole
                        // adjacency group, not to an unresolved subtoken.
                        if !is_subtoken_group[i] {
                            if let Some(ref fallback) = self.fallback {
                                match fallback.phonemize(&word) {
                                    Ok(ps) => tokens[i].phonemes = Some(ps),
                                    Err(e) => {
                                        log::error!("fallback error for '{}': {}", word, e);
                                        return Err(G2PError::Fallback(e));
                                    }
                                }
                            }
                        }

                        // Leave it unresolved for the group merge or final
                        // unknown marker. Misaki does not spell OOV words out.
                    } else {
                        // Try to normalize the character or return unknown
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
                            // unresolved, so it contributes nothing when merged
                            // into the following word. `[Illustration` is
                            // ˌɪləstrˈAʃən, not `( ˌɪləstrˈAʃən`.
                            //
                            // SUBTOKEN_JUNKS still resolve to the empty string
                            // (resolve_tokens), which the merge pass applies.
                            tokens[i].phonemes = None;
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
        {
            const SUBTOKEN_JUNKS: &str = "',-._\u{2018}\u{2019}/";
            let groups = Self::group_by_adjacency(&tokens);
            if groups.len() != tokens.len() {
                let mut merged: Vec<MToken> = Vec::with_capacity(groups.len());
                for g in groups {
                    if g.len() == 1 {
                        merged.push(tokens[g.start].clone());
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
                    let part_of = |tk: &MToken| -> String {
                        if !tk.text.is_empty() && tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                            String::new()
                        } else {
                            tk.phonemes.clone().unwrap_or_else(|| self.unk.clone())
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
                    if let Some(spoken) = currency_of[g.start]
                        .filter(|_| {
                            let c = text.replace(',', "");
                            !c.is_empty() && c.chars().all(|ch| ch.is_ascii_digit() || ch == '.')
                        })
                        .and_then(|sym| self.currency_words(sym, &text))
                    {
                        if let Ok((p, _)) = self.g2p(&spoken) {
                            // The symbol is already its own token, silenced by
                            // the pass above, so only the number is emitted here.
                            let mut tk = group[0].clone();
                            tk.text = text;
                            tk.tag = tag;
                            tk.whitespace = ws;
                            tk.phonemes = Some(p);
                            merged.push(tk);
                            continue;
                        }
                    }

                    if self.is_number_like(&text) {
                        let spoken = self.convert_number(&text);
                        if spoken != text {
                            // Word by word, so "point" keeps its -2 stress; the
                            // phrase path is only a fallback.
                            if let Some(p) = self.number_phonemes(&spoken).or_else(|| {
                                self.g2p(&spoken).ok().map(|(p, _)| p.trim_end().to_string())
                            }) {
                                let mut tk = group[0].clone();
                                tk.text = text;
                                tk.tag = tag;
                                tk.whitespace = ws;
                                tk.phonemes = Some(p);
                                merged.push(tk);
                                continue;
                            }
                        }
                    }

                    let n = g.len();
                    let mut res: Vec<Option<String>> = vec![None; n];
                    let mut left = 0usize;
                    let mut right = n;
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
                                .map(|(ps, _)| ps)
                        } else {
                            None
                        };
                        if let Some(ps) = hit {
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
                                if !t.is_empty() && t.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                                    res[right] = Some(String::new());
                                }
                            }
                            left = 0;
                        }
                    }
                    if let Some(fallback) = &self.fallback {
                        let needs_fallback = res.iter().enumerate().any(|(i, part)| {
                            part.is_none()
                                && !group[i].text.is_empty()
                                && !group[i]
                                    .text
                                    .chars()
                                    .all(|c| SUBTOKEN_JUNKS.contains(c))
                        });
                        if needs_fallback {
                            let ps = fallback.phonemize(&text)?;
                            let mut tk = group[0].clone();
                            tk.text = text;
                            tk.tag = tag;
                            tk.whitespace = ws;
                            tk.phonemes = Some(ps);
                            merged.push(tk);
                            continue;
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
                        let mut parts: Vec<String> = res
                            .iter()
                            .enumerate()
                            .map(|(i, r)| match r {
                                Some(ps) => ps.clone(),
                                // Unresolved: misaki writes `unk`, which our
                                // pipeline sets to the empty string. Fall back to
                                // whatever the per-token pass found only for a
                                // token the span loop never reached.
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
                                .filter(|(_, p)| !p.is_empty())
                                .map(|(i, p)| (p.contains(PRIMARY), weight(p), i))
                                .collect();
                            let n_primary = idx.iter().filter(|(b, _, _)| *b).count();
                            if idx.len() == 2
                                && group[idx[0].2].text.chars().count() == 1
                            {
                                let i = idx[1].2;
                                parts[i] = self.lexicon.apply_stress(&parts[i], Some(-0.5));
                            } else if idx.len() >= 2 && n_primary > (idx.len() + 1) / 2 {
                                idx.sort();
                                let half = idx.len() / 2;
                                for (_, _, i) in idx.into_iter().take(half) {
                                    parts[i] = self.lexicon.apply_stress(&parts[i], Some(-0.5));
                                }
                            }
                        }

                        let mut out = String::new();
                        for part in parts {
                            if prespace
                                && !out.is_empty()
                                && !out.ends_with(char::is_whitespace)
                                && !part.is_empty()
                            {
                                out.push(' ');
                            }
                            out.push_str(&part);
                        }
                        Some(out)
                    });

                    let mut tk = group[0].clone();
                    tk.text = text;
                    tk.tag = tag;
                    tk.whitespace = ws;
                    tk.phonemes = phonemes;
                    merged.push(tk);
                }
                tokens = merged;
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
        // The re-resolution is deliberately narrow. It touches a token only
        // when it is a plain alphabetic word with no alias, already resolved by
        // the lexicon, and `get_word` under the new context returns something
        // else. Numbers, currency, punctuation, aliases and out-of-vocabulary
        // words are left exactly as they were.
        {
            let n = tokens.len();
            let mut ctxs = vec![crate::lexicon::TokenContext::default(); n];
            let mut rolling = crate::lexicon::TokenContext::default();
            for i in (0..n).rev() {
                ctxs[i] = rolling.clone();
                rolling =
                    Self::token_context(&rolling, tokens[i].phonemes.as_deref(), &tokens[i]);
            }
            for i in 0..n {
                if tokens[i].underscore().alias.is_some() {
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
                let stress = tokens[i].underscore().stress.or_else(|| {
                    if key == key.to_lowercase() {
                        None
                    } else if key == key.to_uppercase() {
                        Some(self.lexicon.cap_stresses.1)
                    } else {
                        Some(self.lexicon.cap_stresses.0)
                    }
                });
                let tag = tokens[i].tag.clone();
                if let Some((ps, _)) =
                    self.lexicon.get_word(&key, &tag, stress, Some(&ctxs[i]))
                {
                    tokens[i].phonemes = Some(ps);
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

    /// Does this token go down misaki's number path?
    ///
    /// misaki's `Lexicon.is_number` peels an alphabetic suffix first, so `1st`,
    /// `23rd` and `2nd` are numbers, not words. Testing only for all-digits
    /// meant an ordinal never reached `get_number` and `1st` read as "one".
    fn is_number_like(&self, word: &str) -> bool {
        let clean = word.replace(',', "");
        let lower = clean.to_lowercase();
        for suf in ["st", "nd", "rd", "th"] {
            if let Some(d) = lower.strip_suffix(suf) {
                if !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()) {
                    return true;
                }
            }
        }
        if let Some(rest) = clean.strip_prefix('-') {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
        let unsigned = clean.strip_prefix('-').unwrap_or(&clean);
        if unsigned.matches('.').count() > 1
            && unsigned.chars().any(|c| c.is_ascii_digit())
            && unsigned
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.')
        {
            return true;
        }
        // A single decimal point, with an optional integer part.
        if unsigned.matches('.').count() == 1 {
            if let Some((a, b)) = unsigned.split_once('.') {
                if (a.is_empty() || a.chars().all(|c| c.is_ascii_digit()))
                    && !b.is_empty()
                    && b.chars().all(|c| c.is_ascii_digit())
                {
                    return true;
                }
            }
        }
        self.is_number(word)
    }

    fn is_number(&self, word: &str) -> bool {
        let clean = word.replace(",", "");
        clean.parse::<i64>().is_ok()
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

    /// A currency amount spelled out, or None if `word` is not one.
    ///
    /// See the call site for misaki's rules; the units come from its CURRENCIES
    /// table.
    fn currency_words(&self, sym: char, amount: &str) -> Option<String> {
        let units: (&str, &str) = match sym {
            '$' => ("dollar", "cent"),
            '\u{a3}' => ("pound", "pence"),
            '\u{20ac}' => ("euro", "cent"),
            _ => return None,
        };
        let rest: String = amount.replace(',', "");
        if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return None;
        }
        let mut parts: Vec<(i64, &str)> = Vec::new();
        for (i, seg) in rest.split('.').enumerate().take(2) {
            let n: i64 = if seg.is_empty() { 0 } else { seg.parse().ok()? };
            parts.push((n, if i == 0 { units.0 } else { units.1 }));
        }
        // misaki drops a zero part rather than saying "zero dollars".
        if parts.len() > 1 {
            if parts[1].0 == 0 {
                parts.truncate(1);
            } else if parts[0].0 == 0 {
                parts.remove(0);
            }
        }
        let mut out: Vec<String> = Vec::new();
        for (n, unit) in parts {
            if !out.is_empty() {
                out.push("and".to_string());
            }
            out.push(Self::normalize_num_words(
                &Num2Words::new(n).to_words().ok()?,
            ));
            out.push(if n == 1 || unit == "pence" {
                unit.to_string()
            } else {
                format!("{unit}s")
            });
        }
        Some(out.join(" "))
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

    /// A digit group as words, for the number paths that need a piece spelled
    /// without going back through the full `convert_number`.
    fn spell(digits: &str) -> String {
        digits
            .parse::<i64>()
            .ok()
            .and_then(|n| Num2Words::new(n).to_words().ok())
            .map(|w| Self::normalize_num_words(&w))
            .unwrap_or_else(|| digits.to_string())
    }

    /// misaki's non-head reading of a digit group (`Lexicon.get_number`):
    ///
    /// ```text
    ///     elif not is_head and '.' not in word:
    ///         if num[0] == '0' or len(num) > 3: digit by digit
    ///         elif len(num) == 3 and not num.endswith('00'):
    ///             num[0], then 'O' + num[2] if num[1]=='0' else num[1:]
    ///         else: the whole as a cardinal
    /// ```
    ///
    /// Returns None when the ordinary cardinal reading applies, including for
    /// four digits -- those are caught earlier by the YEAR rule, which is why
    /// 0199 reads "one ninety nine" and 1234 "twelve thirty four".
    fn nonhead_number(&self, word: &str) -> Option<String> {
        let num = word.replace(',', "");
        if num.is_empty() || !num.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        if num.len() == 4 {
            return None; // the year rule owns this
        }
        if num.starts_with('0') || num.len() > 3 {
            return Some(
                num.chars()
                    .map(|c| Self::spell(&c.to_string()))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        if num.len() == 3 && !num.ends_with("00") {
            let first = Self::spell(&num[..1]);
            let rest = &num[1..];
            return Some(if rest.starts_with('0') {
                format!("{first} O {}", Self::spell(&rest[1..]))
            } else {
                format!("{first} {}", Self::spell(rest))
            });
        }
        None
    }

    fn number_component(&self, part: &str) -> String {
        let digit_wise =
            part.starts_with('0') || (part.len() != 2 && part[1..].chars().any(|c| c != '0'));
        if digit_wise {
            part.chars()
                .map(|digit| Self::spell(&digit.to_string()))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            Self::spell(part)
        }
    }

    fn convert_number(&self, word: &str) -> String {
        let clean = word.replace(",", "");

        // misaki prefixes "minus" for a leading '-' (`Lexicon.get_number`):
        //
        //     if word.startswith('-'):
        //         result.append(self.lookup('minus', ...))
        //         word = word[1:]
        //
        // We were dropping the sign, so -40 read as "forty".
        if let Some(rest) = clean.strip_prefix('-') {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit() || c == '.') {
                return format!("minus {}", self.convert_number(rest));
            }
        }

        // Ordinals. misaki (`Lexicon.get_number`) peels an alphabetic suffix
        // and, when it is one of ORDINALS = {st, nd, rd, th} and the rest is
        // digits, reads the ORDINAL form:
        //
        //     if is_digit(word) and suffix in ORDINALS:
        //         extend_num(num2words(int(word), to='ordinal'), escape=True)
        //
        // Without it "23rd" read as "twenty three" and "1st" as "one".
        {
            let lower = clean.to_lowercase();
            for suf in ["st", "nd", "rd", "th"] {
                if let Some(digits) = lower.strip_suffix(suf) {
                    if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                        if let Ok(n) = digits.parse::<i64>() {
                            if let Ok(spoken) = Num2Words::new(n).ordinal().to_words() {
                                return Self::normalize_num_words(&spoken);
                            }
                        }
                    }
                }
            }
        }

        // Bare four-digit numbers read as years, per misaki's get_number.
        if word.len() == 4 && word.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(y) = clean.parse::<i64>() {
                if let Some(spoken) = Self::year_words(y) {
                    return Self::normalize_num_words(&spoken);
                }
            }
        }

        if let Ok(val) = clean.parse::<i64>() {
            let n2w = match self.lexicon.lang {
                Language::EnglishUS | Language::EnglishGB => Num2Words::new(val),
                // Language::Italian => Num2Words::new(val).lang(num2words::Lang::English),
            };
            if let Ok(spoken) = n2w.to_words() {
                return Self::normalize_num_words(&spoken);
            }
        }

        // Dotted versions read each component independently.
        if clean.matches('.').count() > 1 {
            return clean
                .split('.')
                .filter(|part| !part.is_empty())
                .map(|part| self.number_component(part))
                .collect::<Vec<_>>()
                .join(" ");
        }

        // A single decimal point: the integer part as a cardinal, then "point",
        // then each fractional DIGIT on its own. Verified against misaki:
        //
        //   3.5    -> three point five        3.75   -> three point seven five
        //   1.25   -> one point two five      100.01 -> one hundred point zero one
        if clean.matches('.').count() == 1 {
            let (int_part, frac) = clean.split_once('.').unwrap();
            let int_ok = int_part.is_empty() || int_part.chars().all(|c| c.is_ascii_digit());
            let frac_ok = !frac.is_empty() && frac.chars().all(|c| c.is_ascii_digit());
            if int_ok && frac_ok {
                let head = if int_part.is_empty() {
                    Some(String::new())
                } else {
                    int_part
                        .parse::<i64>()
                        .ok()
                        .and_then(|iv| Num2Words::new(iv).to_words().ok())
                };
                if let Some(head) = head {
                    let significant_frac = frac.trim_end_matches('0');
                    if significant_frac.is_empty() {
                        return if head.is_empty() {
                            "zero".to_string()
                        } else {
                            Self::normalize_num_words(&head)
                        };
                    }
                    let mut out = Vec::new();
                    if !head.is_empty() {
                        out.push(Self::normalize_num_words(&head));
                    }
                    out.push("point".to_string());
                    for d in significant_frac.chars() {
                        let dv = (d as u8 - b'0') as i64;
                        if let Ok(w) = Num2Words::new(dv).to_words() {
                            out.push(Self::normalize_num_words(&w));
                        }
                    }
                    return out.join(" ");
                }
            }
        }

        word.to_string()
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
        // misaki 0.9.4 gives exactly this, verified against the oracle:
        //
        //     "word - word — word" -> wˈɜɹd ❓ wˈɜɹd — wˈɜɹd
        //
        // The two dashes are NOT alike. spaCy tags the em-dash `:`, which puts
        // it in PUNCT_TAGS and renders it —; it tags the plain hyphen HYPH,
        // which is not punctuation, so the lexicon misses it and it comes back
        // unresolved and prints as `unk`. This assertion used to forbid `unk`
        // altogether, which was the pre-fork behaviour.
        assert!(dash.contains('—'), "em-dash dropped: {dash}");
        assert_eq!(
            dash, "wˈɜɹd ❓ wˈɜɹd — wˈɜɹd",
            "does not match misaki 0.9.4"
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
