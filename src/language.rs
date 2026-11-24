use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Language {
    EnglishUS,
    #[cfg(feature = "gb")]
    EnglishGB,
    // Italian,
}

impl Language {
    pub fn is_english(&self) -> bool {
        match self {
            Language::EnglishUS => true,
            #[cfg(feature = "gb")]
            Language::EnglishGB => true,
        }
    }

    /// British English morphology and espeak voice when the `gb` feature is on.
    pub fn is_british(self) -> bool {
        #[cfg(feature = "gb")]
        {
            matches!(self, Language::EnglishGB)
        }
        #[cfg(not(feature = "gb"))]
        {
            let _ = self;
            false
        }
    }
}
