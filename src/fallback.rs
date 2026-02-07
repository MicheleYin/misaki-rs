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

    /// Convert espeak IPA output to misaki phoneme format
    fn convert_espeak_to_misaki(&self, espeak_ipa: &str) -> String {
        let mut result = espeak_ipa.to_string();

        // Conversions to match misaki phoneme set
        result = result.replace("iː", "ˈi").replace("i:", "ˈi");
        result = result.replace("uː", "ˈu").replace("u:", "ˈu");
        result = result.replace("ɜː", "ɜ").replace("ɜ:", "ɜ");
        result = result.replace("ɔː", "ɔ").replace("ɔ:", "ɔ");
        result = result.replace("ɑː", "ɑ").replace("ɑ:", "ɑ");
        result = result.replace("ː", ""); // remove remaining length markers
        result = result.replace("_", ""); // stop syllables

        result.replace("ˈˈ", "ˈ").replace("ˌˌ", "ˌ")
    }
}

#[cfg(feature = "espeak")]
impl Fallback for EspeakFallback {
    fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
        let _lock = ESPEAK_MUTEX
            .lock()
            .map_err(|e| FallbackError::MutexPoisoned(format!("{:?}", e)))?;
        let voice = if self.british { "en" } else { "en-us" };

        // Use the portable espeak-rs call (used in kokoros)
        match text_to_phonemes(word, voice, None, true, false) {
            Ok(phonemes) => {
                if phonemes.is_empty() {
                    return Err(FallbackError::NoPhonemes {
                        word: word.to_string(),
                    });
                }
                let cleaned = self.convert_espeak_to_misaki(&phonemes.join(""));
                Ok(cleaned)
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
        assert!(!phonemes.contains("ˈɛl"));  // Should not have spelled-out 'L'
    }

    #[test]
    fn test_espeak_nonsense_word() {
        let fallback = EspeakFallback::new(false).unwrap();

        // espeak handles even nonsense words
        let phonemes = fallback.phonemize("xyzqwop").unwrap();
        assert!(!phonemes.is_empty(), "espeak should phonemize nonsense words");
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
        assert!(us_phonemes.contains("sk"), "US schedule usually has 'sk', got: {}", us_phonemes);
        assert!(gb_phonemes.contains("ʃ"), "GB schedule usually starts with 'ʃ', got: {}", gb_phonemes);
    }
}
