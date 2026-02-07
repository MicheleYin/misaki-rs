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

pub struct G2P {
    pub lexicon: Lexicon,
    pub unk: String,
    subtoken_regex: Regex,
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

        Self {
            lexicon: Lexicon::new(lang),
            unk: "❓".to_string(),
            subtoken_regex,
            tagger: PerceptronTagger::new(weights_json, classes_txt, tags_json),
            rules,
            fallback,
        }
    }

    pub fn preprocess(&self, text: &str) -> (String, Vec<String>, HashMap<usize, String>) {
        // Simplified preprocess: just return the text and tokens for now
        // Python handles links like [text](phonemes), we'll skip that for simplicity unless needed
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        (text.to_string(), tokens, HashMap::new())
    }

    pub fn tokenize(&self, text: &str) -> Vec<MToken> {
        // Use language-tokenizer for word boundary detection (like spaCy in Python)
        // However, language-tokenizer with snowball does stemming, so we need to extract original text
        // Strategy: Use a simple word splitter that handles contractions, then apply subtokenization
        // Python uses spaCy which preserves original text, then applies subtokenize later

        // Simple word splitting that handles contractions: split on whitespace and punctuation
        // but keep contractions together
        let word_boundary_regex = Regex::new(r"\S+").unwrap();
        let mut tokens = Vec::new();

        for mat in word_boundary_regex.find_iter(text) {
            let word = mat.as_str();
            // Apply subtokenization regex to each word (like Python's subtokenize in retokenize)
            // This handles abbreviations, numbers, etc. but preserves contractions
            let subtokens: Vec<&str> = self
                .subtoken_regex
                .find_iter(word)
                .map(|m| m.as_str())
                .collect();

            if subtokens.is_empty() {
                // If regex doesn't match, use the word as-is
                let tk = MToken::new(word.to_string(), "NN".to_string(), " ".to_string());
                tokens.push(tk);
            } else {
                for sub in subtokens {
                    let tk = MToken::new(sub.to_string(), "NN".to_string(), " ".to_string());
                    tokens.push(tk);
                }
            }
        }

        tokens
    }

    pub fn g2p(&self, text: &str) -> Result<(String, Vec<MToken>), G2PError> {
        let (processed_text, _, _) = self.preprocess(text);
        let mut tokens = self.tokenize(&processed_text);

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
            let stress = if word == word.to_lowercase() {
                None
            } else {
                Some(if word == word.to_uppercase() {
                    self.lexicon.cap_stresses.1
                } else {
                    self.lexicon.cap_stresses.0
                })
            };

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
                                    tokens[i].phonemes = Some(" ".to_string());
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

    #[test]
    fn test_g2p_basic() {
        let _ = env_logger::try_init();
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("Hello, world!").unwrap();
        println!("Phonemes: {}", phonemes);
        assert!(!phonemes.contains("❓"));
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
    fn test_kokoros_long_text() {
        let g2p = G2P::new(Language::EnglishUS);
        // Reduced to 100 to check if it crashes
        let long_text = "a".repeat(1000);
        let (p, _) = g2p.g2p(&long_text).unwrap();
        assert!(!p.is_empty());
    }
}
