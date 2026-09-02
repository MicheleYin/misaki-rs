use crate::fallback::{Fallback, FallbackError};
use crate::language::Language;
#[cfg(feature = "espeak")]
use crate::fallback::EspeakFallback;
use crate::languages::{LanguageRules, english::English};
use crate::lexicon::Lexicon;
use thiserror::Error;
use crate::tagger::PerceptronTagger;
use crate::token::MToken;
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
    subtoken_regex: Regex,
    link_regex: Regex,
    tagger: PerceptronTagger,
    rules: Box<dyn LanguageRules>,
    fallback: Option<Box<dyn Fallback>>,
}

impl G2P {
    pub fn new(lang: Language) -> Self {
        // Regex for subtokenization with better UTF-8 support using Unicode properties
        let subtoken_regex = Regex::new(
            r"(?x)
            ^['‘’]+ |
            (?:^-)?(?:\d?[,.]?\d)+ |
            [\-_]+ |
            ['‘’]{2,} |
            \p{L}+(?:[''']\p{L}+)* |
            [^\s\-_0-9\p{L}''] |
            ['‘’]+$
        ",
        )
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
            let subtokens: Vec<regex::Match> =
                self.subtoken_regex.find_iter(word).collect();

            if subtokens.is_empty() {
                let tk = MToken::new(word.to_string(), "NN".to_string(), " ".to_string());
                tokens.push((tk, word_start..mat.end()));
            } else {
                for sub in subtokens {
                    let s = word_start + sub.start();
                    let e = word_start + sub.end();
                    let tk =
                        MToken::new(sub.as_str().to_string(), "NN".to_string(), " ".to_string());
                    tokens.push((tk, s..e));
                }
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

        // Collect words for tagging
        let words_owned: Vec<String> = tokens.iter().map(|tk| tk.text.clone()).collect();
        let words: Vec<&str> = words_owned.iter().map(|s| s.as_str()).collect();
        let tags = self.tagger.tag(&words);

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
            tk.tag = tag.tag.clone();
        }

        // Process in reverse to build context from future tokens
        for i in (0..tokens.len()).rev() {
            let word = tokens[i].text.clone();
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

            // Determine context from next token
            if i < tokens.len() - 1 {
                let next_word = &tokens[i + 1].text;
                // Check if next word starts with vowel (simple heuristic)
                if let Some(first_char) = next_word.chars().next() {
                    let first_lower = first_char.to_lowercase().next().unwrap();
                    if "aeiou".contains(first_lower) {
                        contexts[i].future_vowel = Some(true);
                    } else if first_char.is_alphabetic() {
                        contexts[i].future_vowel = Some(false);
                    }
                }

                if next_word.to_lowercase() == "to" {
                    contexts[i].future_to = true;
                }
            }

            // Process current token
            if tokens[i].phonemes.is_none() {
                let ctx = Some(&contexts[i]);

                // Use get_word which handles special cases, lookup, and stemming
                if let Some((ps, _)) = self.lexicon.get_word(&word, &tag, stress, ctx) {
                    tokens[i].phonemes = Some(ps);
                }

                if tokens[i].phonemes.is_none() {
                    if word.contains('-') && word.len() > 1 {
                        // Handle hyphenated words like "twenty-one"
                        let parts: Vec<&str> = word.split('-').filter(|s| !s.is_empty()).collect();
                        let mut sub_ps = Vec::new();
                        for part in parts {
                            let (p, _) = self.g2p(part)?;
                            sub_ps.push(p);
                        }
                        tokens[i].phonemes = Some(sub_ps.join(" "));
                    } else if self.is_number(&word) {
                        let spoken = self.convert_number(&word);
                        if spoken != word {
                            let (p, _) = self.g2p(&spoken)?;
                            tokens[i].phonemes = Some(p);
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
                        // Unknown multi-character word - use fallback
                        let mut handled = false;
                        if let Some(ref fallback) = self.fallback {
                            match fallback.phonemize(&word) {
                                Ok(ps) => {
                                    tokens[i].phonemes = Some(ps);
                                    handled = true;
                                }
                                Err(e) => {
                                    log::error!("fallback error for '{}': {}", word, e);
                                    return Err(G2PError::Fallback(e));
                                }
                            }
                        }

                        if !handled {
                            // No fallback available or failed, try character-by-character
                            let mut char_ps = Vec::new();
                            for c in word.chars() {
                                let (p, _) = self.g2p(&c.to_string())?;
                                char_ps.push(p);
                            }
                            tokens[i].phonemes = Some(char_ps.join(" "));
                        }
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
                                '—' | '–' => ' ', // map dashes to spaces
                                _ => c,
                            })
                            .collect();

                        if normalized != word {
                            let (p, _) = self.g2p(&normalized)?;
                            tokens[i].phonemes = Some(p);
                        } else {
                            // Handle standard punctuation and symbols gracefully
                            if word.chars().count() == 1 {
                                let c = word.chars().next().unwrap();
                                if c.is_ascii_punctuation() || "—–…".contains(c) {
                                    // Preserve the six punctuation marks Kokoro's vocab
                                    // encodes explicitly (vocab ids 1-6: ';:,.!?').
                                    // Downstream tokenizers map these to their own ids
                                    // and the synthesis model uses them as prosody cues
                                    // (sentence-end pause, clause breath, question
                                    // intonation). Anything else (em-dash, ellipsis,
                                    // parens, …) still collapses to a single space,
                                    // matching the prior whitespace fallback.
                                    if matches!(c, ';' | ':' | ',' | '.' | '!' | '?') {
                                        tokens[i].phonemes = Some(c.to_string());
                                    } else {
                                        tokens[i].phonemes = Some(" ".to_string());
                                    }
                                } else {
                                    tokens[i].phonemes = Some(self.unk.clone());
                                }
                            } else {
                                tokens[i].phonemes = Some(self.unk.clone());
                            }
                        }
                    }
                }
            }

            // Update context for previous tokens based on current phonemes
            if i > 0 && tokens[i].phonemes.is_some() {
                let vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
                let consonants = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
                let phonemes = tokens[i].phonemes.as_ref().unwrap();
                for c in phonemes.chars() {
                    if vowels.contains(c) {
                        contexts[i - 1].future_vowel = Some(true);
                        break;
                    } else if consonants.contains(c) {
                        contexts[i - 1].future_vowel = Some(false);
                        break;
                    }
                }
            }
        }

        let result = tokens
            .iter()
            .map(|tk| tk.phonemes.as_ref().unwrap_or(&self.unk).clone() + &tk.whitespace)
            .collect::<String>();

        Ok((result, tokens))
    }

    fn is_number(&self, word: &str) -> bool {
        let clean = word.replace(",", "");
        clean.parse::<i64>().is_ok()
    }

    fn convert_number(&self, word: &str) -> String {
        let clean = word.replace(",", "");
        if let Ok(val) = clean.parse::<i64>() {
            let n2w = match self.lexicon.lang {
                Language::EnglishUS | Language::EnglishGB => Num2Words::new(val),
                // Language::Italian => Num2Words::new(val).lang(num2words::Lang::English),
            };
            if let Ok(spoken) = n2w.to_words() {
                return spoken;
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

    #[test]
    fn test_g2p_basic() {
        let _ = env_logger::try_init();
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("Hello, world!").unwrap();
        println!("Phonemes: {}", phonemes);
        assert!(!phonemes.contains("❓"));
    }

    #[test]
    fn test_with_fallback_uses_injected_fallback_for_oov_word() {
        let g2p = G2P::with_fallback(Language::EnglishUS, Some(Box::new(TestFallback)));

        let (phonemes, tokens) = g2p.g2p("qzxwv").unwrap();

        assert_eq!(phonemes.trim(), "test-phonemes");
        assert_eq!(tokens[0].phonemes.as_deref(), Some("test-phonemes"));
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
        assert!(
            !dash.contains("❓"),
            "Dashes should be handled gracefully, got: {}",
            dash
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

        // Non-vocab ASCII punctuation should still collapse to space.
        let (out, _) = g2p.g2p("hello (world)").unwrap();
        assert!(!out.contains('('), "( unexpectedly preserved: {out:?}");
        assert!(!out.contains(')'), ") unexpectedly preserved: {out:?}");
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
