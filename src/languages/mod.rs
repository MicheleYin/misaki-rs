use crate::lexicon::Lexicon;

pub trait LanguageRules: Send + Sync {
    fn apply_rules(&self, word: &str, tag: &str, lexicon: &Lexicon) -> Option<String>;
}

pub mod english;
// pub mod italian;
