use thiserror::Error;

#[cfg(feature = "espeak")]
use std::sync::Mutex;
#[cfg(feature = "espeak")]
use espeak_rs::text_to_phonemes;

#[cfg(feature = "espeak")]
static ESPEAK_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Error, Debug)]
pub enum FallbackError {
    #[error("espeak error for '{word}': {error}")]
    Espeak {
        word: String,
        error: String,
    },
    #[error("no phonemes matched for '{word}'")]
    NoPhonemes {
        word: String,
    },
    #[error("mutex poisoned: {0}")]
    MutexPoisoned(String),
}

/// Trait for OOV (out-of-vocabulary) word fallback mechanisms
pub trait Fallback: Send + Sync {
    /// Convert unknown word to phonemes
    /// Returns phonemes
    /// Note: espeak-ng is rule-based and always produces output
    fn phonemize(&self, word: &str) -> Result<String, FallbackError>;
}

/// espeak-ng based fallback (requires `espeak` feature)
#[cfg(feature = "espeak")]
pub struct EspeakFallback {
    british: bool,
}

#[cfg(feature = "espeak")]
impl EspeakFallback {
    pub fn new(british: bool) -> Result<Self, String> {
        Ok(Self { british })
    }

    /// Port of `misaki.espeak.EspeakFallback.__call__` conversion (E2M +
    /// post-passes). Tie character is `^` when the backend emits ties; we also
    /// accept already-tied digraphs from backends that merge them.
    fn convert_espeak_to_misaki(&self, espeak_ipa: &str) -> String {
        let mut ps = espeak_ipa.trim().to_string();

        // Longer keys first — mirrors `sorted(..., key=lambda kv: -len(kv[0]))`.
        const E2M: &[(&str, &str)] = &[
            ("ʔˌn\u{0329}", "ʔn"),
            ("ʔn\u{0329}", "ʔn"),
            ("a^ɪ", "I"),
            ("aɪ", "I"),
            ("a^ʊ", "W"),
            ("aʊ", "W"),
            ("d^ʒ", "ʤ"),
            ("dʒ", "ʤ"),
            ("e^ɪ", "A"),
            ("eɪ", "A"),
            ("t^ʃ", "ʧ"),
            ("tʃ", "ʧ"),
            ("ɔ^ɪ", "Y"),
            ("ɔɪ", "Y"),
            ("ə^l", "ᵊl"),
            ("ʲo", "jo"),
            ("ʲə", "jə"),
            ("ʲ", ""),
            ("ɚ", "əɹ"),
            ("r", "ɹ"),
            ("x", "k"),
            ("ç", "k"),
            ("ɐ", "ə"),
            ("ɬ", "l"),
            ("\u{0303}", ""),
        ];
        for (old, new) in E2M {
            ps = ps.replace(old, new);
        }

        // `re.sub(r'(\S)\u0329', r'ᵊ\1', ps).replace(chr(809), '')`
        {
            let mut out = String::with_capacity(ps.len());
            let chars: Vec<char> = ps.chars().collect();
            let mut i = 0usize;
            while i < chars.len() {
                if i + 1 < chars.len() && chars[i + 1] == '\u{0329}' && !chars[i].is_whitespace()
                {
                    out.push('ᵊ');
                    out.push(chars[i]);
                    i += 2;
                } else if chars[i] == '\u{0329}' {
                    i += 1; // chr(809) bare
                } else {
                    out.push(chars[i]);
                    i += 1;
                }
            }
            ps = out;
        }

        if self.british {
            ps = ps
                .replace("e^ə", "ɛː")
                .replace("eə", "ɛː")
                .replace("iə", "ɪə")
                .replace("ə^ʊ", "Q")
                .replace("əʊ", "Q");
        } else {
            ps = ps
                .replace("o^ʊ", "O")
                .replace("oʊ", "O")
                .replace("ɜːɹ", "ɜɹ")
                .replace("ɜː", "ɜɹ")
                .replace("ɪə", "iə")
                .replace('ː', "");
        }

        ps = ps.replace('o', "ɔ"); // espeak < 1.52
        ps = ps.replace('ɾ', "T").replace('ʔ', "t");
        ps = ps.replace('^', "");
        ps
    }
}

#[cfg(feature = "espeak")]
impl Fallback for EspeakFallback {
    fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
        let sanitized_word: String = word.chars().filter(|c| !c.is_control()).collect();
        if sanitized_word.is_empty() {
            // Soft-fail like Python `(None, None)` when there is nothing to say.
            return Err(FallbackError::NoPhonemes {
                word: word.to_string(),
            });
        }

        let _lock = ESPEAK_MUTEX
            .lock()
            .map_err(|e| FallbackError::MutexPoisoned(format!("{:?}", e)))?;
        let voice = if self.british { "en" } else { "en-us" };

        // Use the portable espeak-rs call (used in kokoros).
        //
        // espeak-rs 0.1.9 always appends '.' after every utterance: its
        // `_text_to_phonemes` hardcodes `terminator = 0`, and
        // `CLAUSE_INTONATION_FULL_STOP` is also 0, so the full-stop branch
        // always fires. phonemizer (stock misaki) does not — it only keeps
        // clause punctuation that came from the input (`ilili.` → ….,
        // `il-lilly` → no trailing dot; `-` → empty). Strip that spurious
        // trailing '.' unless the orth itself ended with one.
        //
        // `Ok("")` is successful silence (HYPH `-` → `('', 2)` in Python).
        // `Err(NoPhonemes)` is the soft-fail when the backend returns nothing.
        match text_to_phonemes(&sanitized_word, voice, None, true, false) {
            Ok(phonemes) => {
                if phonemes.is_empty() {
                    return Err(FallbackError::NoPhonemes {
                        word: word.to_string(),
                    });
                }
                let mut raw = phonemes.join("");
                if !sanitized_word.ends_with('.') {
                    while raw.ends_with('.') {
                        raw.pop();
                    }
                }
                Ok(self.convert_espeak_to_misaki(&raw))
            }
            Err(e) => Err(FallbackError::Espeak {
                word: word.to_string(),
                error: format!("{:?}", e),
            }),
        }
    }
}

#[cfg(all(test, feature = "espeak"))]
mod tests {
    use super::*;

    #[test]
    fn test_espeak_fallback() {
        let fallback = EspeakFallback::new(false).expect("espeak should initialize");

        // Test unknown word - espeak ALWAYS returns something
        let phonemes = fallback.phonemize("ilili").unwrap();
        assert!(!phonemes.is_empty());

        // Verify it doesn't spell out character-by-character
        assert!(!phonemes.contains("ˈɛl")); // Should not have spelled-out 'L'
    }

    #[test]
    fn test_espeak_fallback_ignores_control_only_input() {
        let fallback = EspeakFallback::new(false).expect("espeak should initialize");

        assert!(matches!(
            fallback.phonemize("\0\x01\x02"),
            Err(FallbackError::NoPhonemes { .. })
        ));
        assert!(!ESPEAK_MUTEX.is_poisoned());
    }

    #[test]
    fn test_espeak_conversion_matches_python_reference_forms() {
        let us = EspeakFallback::new(false).expect("espeak should initialize");
        let gb = EspeakFallback::new(true).expect("espeak should initialize");

        // Stock misaki keeps trailing punctuation (preserve_punctuation=True).
        assert_eq!(us.convert_espeak_to_misaki("ˈɪllˈɪli."), "ˈɪllˈɪli.");
        // US: ɜː → ɜɹ; keep trailing punct; ə^l / əl via syllabic / tied forms.
        assert_eq!(us.convert_espeak_to_misaki("ɜː."), "ɜɹ.");
        assert_eq!(us.convert_espeak_to_misaki("ɚ"), "əɹ");
        assert_eq!(us.convert_espeak_to_misaki("o^ʊ"), "O");
        // GB keeps length mark on ɜː and maps ə^ʊ → Q.
        assert_eq!(gb.convert_espeak_to_misaki("ɜː"), "ɜː");
        assert_eq!(gb.convert_espeak_to_misaki("ə^ʊ"), "Q");
    }

    #[test]
    fn test_espeak_strips_spurious_clause_full_stop() {
        let us = EspeakFallback::new(false).expect("espeak should initialize");
        // espeak-rs always appends '.'; phonemizer/misaki do not for bare words.
        let ill = us.phonemize("il-lilly").unwrap();
        assert_eq!(ill, "ˈɪllˈɪli", "got {ill:?}");
        assert_eq!(us.phonemize("-").unwrap(), "");
        // Input orth with a real period keeps it.
        let with_dot = us.phonemize("ilili.").unwrap();
        assert!(
            with_dot.ends_with('.'),
            "expected trailing period from orth, got {with_dot:?}"
        );
    }

    #[test]
    fn test_espeak_nonsense_word() {
        let fallback = EspeakFallback::new(false).unwrap();

        // espeak handles even nonsense words
        let phonemes = fallback.phonemize("xyzqwop").unwrap();
        assert!(
            !phonemes.is_empty(),
            "espeak should phonemize nonsense words"
        );
    }

    #[test]
    fn test_espeak_phonemes_beat() {
        let fallback = EspeakFallback::new(false).unwrap();
        let phonemes = fallback.phonemize("beat").unwrap();
        // Misaki for beat should probably be bˈit or similar
        assert!(
            phonemes.contains("ˈi"),
            "Should contain stressed i, got: {}",
            phonemes
        );
    }

    #[test]
    fn test_espeak_american_vs_british() {
        let us = EspeakFallback::new(false).unwrap();
        let gb = EspeakFallback::new(true).unwrap();

        // Test word with different pronunciations
        let us_phonemes = us.phonemize("schedule").unwrap();
        let gb_phonemes = gb.phonemize("schedule").unwrap();

        assert!(!us_phonemes.is_empty());
        assert!(!gb_phonemes.is_empty());
        // Pronunciations should differ (US: sked-, GB: shed-)
        // In IPA: US often starts with 'sk', GB often starts with 'ʃ'
        assert!(
            us_phonemes.contains("sk"),
            "US schedule usually has 'sk', got: {}",
            us_phonemes
        );
    }
}
