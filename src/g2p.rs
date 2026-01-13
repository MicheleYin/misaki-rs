use regex::Regex;
use num2words::Num2Words;
use crate::token::MToken;
use crate::lexicon::Lexicon;
use std::collections::HashMap;
use crate::language::Language;
use crate::languages::{LanguageRules, english::English};
use crate::tagger::PerceptronTagger;

pub struct G2P {
    pub lexicon: Lexicon,
    pub unk: String,
    subtoken_regex: Regex,
    tagger: PerceptronTagger,
    rules: Box<dyn LanguageRules>,
}

impl G2P {
    pub fn new(lang: Language) -> Self {
        // Regex for subtokenization with better UTF-8 support using Unicode properties
        let subtoken_regex = Regex::new(r"(?x)
            ^['‘’]+ |
            (?:^-)?(?:\d?[,.]?\d)+ |
            [\-_]+ |
            ['‘’]{2,} |
            \p{L}+(?:[\-'‘’]\p{L})* |
            [^\s\-_0-9\p{L}'‘’] |
            ['‘’]+$
        ").unwrap();

        let weights_json = include_str!("resources/tagger/weights.json");
        let classes_txt = include_str!("resources/tagger/classes.txt");
        let tags_json = include_str!("resources/tagger/tags.json");

        let rules: Box<dyn LanguageRules> = match lang {
            Language::EnglishUS | Language::EnglishGB => Box::new(English),
            // Language::Italian => Box::new(Italian),
        };

        Self {
            lexicon: Lexicon::new(lang),
            unk: "❓".to_string(),
            subtoken_regex,
            tagger: PerceptronTagger::new(weights_json, classes_txt, tags_json),
            rules,
        }
    }

    pub fn preprocess(&self, text: &str) -> (String, Vec<String>, HashMap<usize, String>) {
        // Simplified preprocess: just return the text and tokens for now
        // Python handles links like [text](phonemes), we'll skip that for simplicity unless needed
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        (text.to_string(), tokens, HashMap::new())
    }

    pub fn tokenize(&self, text: &str) -> Vec<MToken> {
        let mut tokens = Vec::new();
        for mat in self.subtoken_regex.find_iter(text) {
            let sub = mat.as_str();
            let tk = MToken::new(sub.to_string(), "NN".to_string(), " ".to_string());
            tokens.push(tk);
        }
        tokens
    }


    pub fn g2p(&self, text: &str) -> (String, Vec<MToken>) {
        let (processed_text, _, _) = self.preprocess(text);
        let mut tokens = self.tokenize(&processed_text);

        // Collect words for tagging
        let words_owned: Vec<String> = tokens.iter().map(|tk| tk.text.clone()).collect();
        let words: Vec<&str> = words_owned.iter().map(|s| s.as_str()).collect();
        let tags = self.tagger.tag(&words);

        eprintln!("DEBUG: g2p '{}' -> {} tokens, {} tags", text, tokens.len(), tags.len());

        for (tk, tag) in tokens.iter_mut().zip(tags.into_iter()) {
            tk.tag = tag.tag;
            if tk.phonemes.is_none() {
                let word = tk.text.clone();
                let tag = tk.tag.clone();
                eprintln!("DEBUG: processing token '{}' with tag '{}'", word, tag);
                
                // Try dictionary lookup
                // Try dictionary lookup
                if let Some((ps, _)) = self.lexicon.lookup(&word, &tag, None) {
                    tk.phonemes = Some(ps);
                }

                if tk.phonemes.is_none() {
                     if word.contains('-') && word.len() > 1 {
                        // Handle hyphenated words like "twenty-one"
                        let parts: Vec<&str> = word.split('-').filter(|s| !s.is_empty()).collect();
                        let mut sub_ps = Vec::new();
                        for part in parts {
                            let (p, _) = self.g2p(part);
                            sub_ps.push(p);
                        }
                        tk.phonemes = Some(sub_ps.join(" "));
                    } else if self.is_number(&word) {
                        let spoken = self.convert_number(&word);
                         if spoken != word {
                             let (p, _) = self.g2p(&spoken);
                             tk.phonemes = Some(p);
                         }
                    }
                }

                if tk.phonemes.is_none() {
                    if let Some(ps) = self.rules.apply_rules(&word, &tag, &self.lexicon) {
                        tk.phonemes = Some(ps);
                    }
                }

                if tk.phonemes.is_none() {
                    let lower = word.to_lowercase();
                    if lower != word {
                        if let Some((ps, _)) = self.lexicon.lookup(&lower, &tag, None) {
                            tk.phonemes = Some(ps);
                        }
                    }
                }

                if tk.phonemes.is_none() {
                    if word.chars().count() > 1 {
                        // Try character-by-character if the whole word is unknown
                        let mut char_ps = Vec::new();
                        for c in word.chars() {
                            let (p, _) = self.g2p(&c.to_string());
                            char_ps.push(p);
                        }
                        tk.phonemes = Some(char_ps.join(" "));
                    } else {
                        // Try to normalize the character or return unknown
                        let normalized: String = word.chars()
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
                            let (p, _) = self.g2p(&normalized);
                            tk.phonemes = Some(p);
                        } else {
                            // Handle standard punctuation and symbols gracefully
                            if word.chars().count() == 1 {
                                let c = word.chars().next().unwrap();
                                if c.is_ascii_punctuation() || "—–…".contains(c) {
                                    tk.phonemes = Some(" ".to_string());
                                } else {
                                    tk.phonemes = Some(self.unk.clone());
                                }
                            } else {
                                tk.phonemes = Some(self.unk.clone());
                            }
                        }
                    }
                }
            }
        }

        let result = tokens.iter()
            .map(|tk| tk.phonemes.as_ref().unwrap_or(&self.unk).clone() + &tk.whitespace)
            .collect::<String>();

        (result, tokens)
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
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("Hello, world!");
        println!("Phonemes: {}", phonemes);
        assert!(!phonemes.contains("❓"));
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
            "I'll", "I've", "it's", "he's", "she's", "we're", "they're",
            "isn't", "aren't", "wasn't", "weren't",
            "don't", "doesn't", "didn't",
            "can't", "couldn't", "shouldn't", "wouldn't", "won't",
            "hasn't", "haven't", "hadn't",
            "let's", "that's", "what's", "who's", "here's", "there's", "where's",
            "how's",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text);
            println!("'{}' -> '{}'", text, p);
            assert!(!p.contains("❓"), "Failed for '{}'", text);
        }
    }

    #[test]
    fn test_casing_and_special_chars() {
        let g2p = G2P::new(Language::EnglishUS);
        
        // Test 1: All caps with suffix
        let (playing, _) = g2p.g2p("PLAYING");
        println!("PLAYING: {}", playing);
        assert!(!playing.contains("❓"), "PLAYING should be resolved, got: {}", playing);

        // Test 2: Contractions
        let (ive, _) = g2p.g2p("I've");
        println!("I've: {}", ive);
        assert!(!ive.contains("❓"), "I've should be resolved, got: {}", ive);

        // Test 3: Dashes
        // em-dash — (U+2014) and hyphen -
        let (dash, _) = g2p.g2p("word - word — word");
        println!("Dash: {}", dash);
        assert!(!dash.contains("❓"), "Dashes should be handled gracefully, got: {}", dash);
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
            let (p, _) = g2p.g2p(text);
            println!("'{}' -> '{}'", text, p);
            if !text.is_empty() { assert!(!p.is_empty(), "Failed for '{}'", text); }
        }
    }

    #[test]
    fn test_kokoros_numbers() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "CHAPTER XIV", "CHAPTER 14", "CHAPTER 123",
            "I have 5 apples and 42 oranges", "The year 2024", "1234567890",
            "CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV", "CHAPTER V",
            "CHAPTER X", "CHAPTER XX", "CHAPTER XXX",
             "In 2024, CHAPTER XIV had 42 pages.",
            "The price is $123.45",
            "Temperature: -5°C",
            "Score: 100%",
            "Version 2.0",
            "3.14159",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text);
            println!("'{}' -> '{}'", text, p);
            assert!(!p.is_empty(), "Failed for '{}'", text);
        }
    }

    #[test]
    fn test_kokoros_utf8_and_special() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "café", "naïve", "résumé", "Zürich", "São Paulo", "Müller",
            "北京", "こんにちは", "Здравствуй", "مرحبا", "🎉🎊🎈",
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
            let (p, _) = g2p.g2p(text);
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
            "Dr. Smith", "Mr. Jones", "Mrs. Brown", "Ms. Davis", "etc.",
            "U.S.A.", "Ph.D.", "A.I.", "NASA", "FBI",
             "   ", "\n\n", "\t\t", "\r\n",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text);
            println!("'{}' -> '{}'", text, p);
        }
    }

    #[test]
    fn test_kokoros_long_text() {
        let g2p = G2P::new(Language::EnglishUS);
        // Reduced to 100 to check if it crashes
        let long_text = "a".repeat(1000); 
        let (p, _) = g2p.g2p(&long_text);
        assert!(!p.is_empty());
    }
}
