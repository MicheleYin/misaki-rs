use crate::data;
use crate::language::Language;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PhonemeEntry {
    Simple(String),
    Tagged(HashMap<String, Option<String>>),
}

pub struct Lexicon {
    pub lang: Language,
    pub cap_stresses: (f64, f64),
    pub golds: HashMap<String, PhonemeEntry>,
    pub silvers: HashMap<String, PhonemeEntry>,
}

impl Lexicon {
    pub fn new(lang: Language) -> Self {
        let (golds_raw, silvers_raw) = match lang {
            Language::EnglishGB => (data::load_gb_gold(), data::load_gb_silver()),
            Language::EnglishUS => (data::load_us_gold(), data::load_us_silver()),
            // Language::Italian => (data::load_it_gold(), data::load_it_silver()),
        };

        let golds = Lexicon::grow_dictionary(golds_raw);
        let silvers = Lexicon::grow_dictionary(silvers_raw);

        Self {
            lang,
            cap_stresses: (0.5, 2.0),
            golds,
            silvers,
        }
    }

    fn grow_dictionary(d: HashMap<String, PhonemeEntry>) -> HashMap<String, PhonemeEntry> {
        let mut e = HashMap::new();
        for (k, v) in d.iter() {
            if k.len() < 2 {
                continue;
            }
            let lower = k.to_lowercase();
            let capitalized = {
                let mut chars = lower.chars();
                match chars.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
                }
            };

            if k == &lower {
                if k != &capitalized {
                    e.insert(capitalized, v.clone());
                }
            } else if k == &capitalized {
                e.insert(lower, v.clone());
            }
        }
        let mut result = d;
        result.extend(e);
        result
    }

    // Helper to get phoneme string based on tag from entry
    fn resolve_phonemes(&self, entry: &PhonemeEntry, tag: &str) -> Option<String> {
        match entry {
            PhonemeEntry::Simple(ps) => Some(ps.clone()),
            PhonemeEntry::Tagged(map) => {
                // Try specific tag, then parent tag, then DEFAULT
                if let Some(Some(ps)) = map.get(tag) {
                    return Some(ps.clone());
                }
                let parent = Lexicon::get_parent_tag(tag);
                if let Some(Some(ps)) = map.get(parent) {
                    return Some(ps.clone());
                }
                map.get("DEFAULT").and_then(|opt| opt.clone())
            }
        }
    }

    pub fn get_parent_tag(tag: &str) -> &str {
        if tag.starts_with("VB") {
            "VERB"
        } else if tag.starts_with("NN") {
            "NOUN"
        } else if tag.starts_with("ADV") || tag.starts_with("RB") {
            "ADV"
        } else if tag.starts_with("ADJ") || tag.starts_with("JJ") {
            "ADJ"
        } else {
            tag
        }
    }

    pub fn lookup(&self, word: &str, tag: &str, stress: Option<f64>) -> Option<(String, i32)> {
        let mut current_word = word.to_string();
        let mut is_nnp = false;

        if word == word.to_uppercase() && !self.golds.contains_key(word) {
            current_word = word.to_lowercase();
            is_nnp = tag == "NNP";
        }

        let mut ps = None;
        let mut rating = 0;

        // println!("DEBUG: lookup '{}', dictionary size: {}", word, self.golds.len());

        if let Some(entry) = self
            .golds
            .get(&current_word)
            .or_else(|| self.golds.get(&current_word.to_lowercase()))
        {
            ps = self.resolve_phonemes(entry, tag);
            rating = 4;
        }

        if ps.is_none() {
            if let Some(entry) = self
                .silvers
                .get(&current_word)
                .or_else(|| self.silvers.get(&current_word.to_lowercase()))
            {
                ps = self.resolve_phonemes(entry, tag);
                rating = 3;
            }
        }

        if ps.is_none() && (word == "three" || word == "one") {
            eprintln!(
                "DEBUG: lookup '{}', dictionary size: {}, contains 'three': {}",
                word,
                self.golds.len(),
                self.golds.contains_key("three")
            );
        }

        // Special NNP handling if not found or no primary stress
        if ps.is_none() || (is_nnp && !ps.as_ref()?.contains('ˈ')) {
            if is_nnp {
                if let Some((nnp_ps, nnp_rating)) = self.get_nnp(&current_word) {
                    ps = Some(nnp_ps);
                    rating = nnp_rating;
                }
            }
        }

        ps.map(|p| (self.apply_stress(&p, stress), rating))
    }

    fn get_nnp(&self, word: &str) -> Option<(String, i32)> {
        let mut ps_parts = Vec::new();
        for c in word.chars() {
            if c.is_alphabetic() {
                if let Some(entry) = self.golds.get(&c.to_uppercase().to_string()) {
                    if let PhonemeEntry::Simple(p) = entry {
                        ps_parts.push(p.clone());
                    }
                } else {
                    return None;
                }
            }
        }
        if ps_parts.is_empty() {
            return None;
        }

        let combined = ps_parts.join("");
        let stressed = self.apply_stress(&combined, Some(0.0));

        // Python: ps = ps.rsplit(SECONDARY_STRESS, 1) -> return PRIMARY_STRESS.join(ps), 3
        let secondary = 'ˌ';
        let primary = 'ˈ';
        if let Some(idx) = stressed.rfind(secondary) {
            let mut result = stressed.clone();
            result.replace_range(idx..idx + secondary.len_utf8(), &primary.to_string());
            Some((result, 3))
        } else {
            Some((stressed, 3))
        }
    }

    pub fn apply_stress(&self, ps: &str, stress: Option<f64>) -> String {
        let primary = 'ˈ';
        let secondary = 'ˌ';
        let vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";

        if stress.is_none() {
            return ps.to_string();
        }
        let s = stress.unwrap();

        if s < -1.0 {
            return ps.replace(primary, "").replace(secondary, "");
        } else if s == -1.0 || (s >= -0.5 && s <= 0.0 && ps.contains(primary)) {
            return ps
                .replace(secondary, "")
                .replace(primary, &secondary.to_string());
        } else if (s == 0.0 || s == 0.5 || s == 1.0)
            && !ps.contains(primary)
            && !ps.contains(secondary)
        {
            if !ps.chars().any(|c| vowels.contains(c)) {
                return ps.to_string();
            }
            return self.restress(&format!("{}{}", secondary, ps));
        } else if s >= 1.0 && !ps.contains(primary) && ps.contains(secondary) {
            return ps.replace(secondary, &primary.to_string());
        } else if s > 1.0 && !ps.contains(primary) && !ps.contains(secondary) {
            if !ps.chars().any(|c| vowels.contains(c)) {
                return ps.to_string();
            }
            return self.restress(&format!("{}{}", primary, ps));
        }
        ps.to_string()
    }

    fn restress(&self, ps: &str) -> String {
        let primary = 'ˈ';
        let secondary = 'ˌ';
        let vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";

        let mut parts: Vec<(f64, char)> =
            ps.chars().enumerate().map(|(i, c)| (i as f64, c)).collect();
        let mut stresses = Vec::new();

        for (i, &(_, c)) in parts.iter().enumerate() {
            if c == primary || c == secondary {
                if let Some(j) = parts[i..].iter().position(|&(_, vc)| vowels.contains(vc)) {
                    stresses.push((i, i + j));
                }
            }
        }

        for (si, vi) in stresses {
            let (_s_pos, s_char) = parts[si];
            parts[si] = (parts[vi].0 - 0.5, s_char);
        }

        parts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        parts.into_iter().map(|(_, c)| c).collect()
    }

    // Stemming logic
    pub fn stem_s(&self, word: &str, tag: &str, stress: Option<f64>) -> Option<(String, i32)> {
        let lower = word.to_lowercase();
        if lower.len() < 3 || !lower.ends_with('s') {
            return None;
        }

        let stem = if !lower.ends_with("ss") && self.is_known(&lower[..lower.len() - 1], tag) {
            &lower[..lower.len() - 1]
        } else if (lower.ends_with("'s")
            || (lower.len() > 4 && lower.ends_with("es") && !lower.ends_with("ies")))
            && self.is_known(&lower[..lower.len() - 2], tag)
        {
            &lower[..lower.len() - 2]
        } else if lower.len() > 4
            && lower.ends_with("ies")
            && self.is_known(&(lower[..lower.len() - 3].to_string() + "y"), tag)
        {
            &(lower[..lower.len() - 3].to_string() + "y")
        } else {
            return None;
        };

        let (stem_ps, rating) = self.lookup(stem, tag, stress)?;
        Some((self.append_s(&stem_ps), rating))
    }

    pub fn append_s(&self, stem: &str) -> String {
        if stem.is_empty() {
            return String::new();
        }
        let last = stem.chars().last().unwrap();
        if "ptkfθ".contains(last) {
            format!("{}s", stem)
        } else if "szʃʒʧʤ".contains(last) {
            format!("{}iz", stem) // Simplified: python uses ɪ or ᵻ
        } else {
            format!("{}z", stem)
        }
    }

    pub fn stem_ed(&self, word: &str, tag: &str, stress: Option<f64>) -> Option<(String, i32)> {
        let lower = word.to_lowercase();
        if lower.len() < 4 || !lower.ends_with('d') {
            return None;
        }
        let stem = if !lower.ends_with("dd") && self.is_known(&lower[..lower.len() - 1], tag) {
            &lower[..lower.len() - 1]
        } else if lower.len() > 4
            && lower.ends_with("ed")
            && !lower.ends_with("eed")
            && self.is_known(&lower[..lower.len() - 2], tag)
        {
            &lower[..lower.len() - 2]
        } else {
            return None;
        };

        let (stem_ps, rating) = self.lookup(stem, tag, stress)?;
        Some((self.append_ed(&stem_ps), rating))
    }

    pub fn append_ed(&self, stem: &str) -> String {
        if stem.is_empty() {
            return String::new();
        }
        let last = stem.chars().last().unwrap();
        if "pkfθʃsʧ".contains(last) {
            format!("{}t", stem)
        } else if last == 'd' {
            format!("{}id", stem)
        } else if last != 't' {
            format!("{}d", stem)
        } else {
            format!("{}id", stem)
        }
    }

    pub fn stem_ing(&self, word: &str, tag: &str, stress: Option<f64>) -> Option<(String, i32)> {
        let lower = word.to_lowercase();
        if lower.len() < 5 || !lower.ends_with("ing") {
            return None;
        }

        let stem = if lower.len() > 5 && self.is_known(&lower[..lower.len() - 3], tag) {
            lower[..lower.len() - 3].to_string()
        } else if self.is_known(&(lower[..lower.len() - 3].to_string() + "e"), tag) {
            lower[..lower.len() - 3].to_string() + "e"
        } else if lower.len() > 5 && self.is_known(&lower[..lower.len() - 4], tag) {
            // Simplified: python regex checks for doubled consonants
            lower[..lower.len() - 4].to_string()
        } else {
            return None;
        };

        // Note: lookup handles string, so we pass current stem string (which is lowercased).
        let (stem_ps, rating) = self.lookup(&stem, tag, stress)?;
        Some((format!("{}ɪŋ", stem_ps), rating))
    }

    pub fn is_known(&self, word: &str, _tag: &str) -> bool {
        self.golds.contains_key(word) || self.silvers.contains_key(word) || word.len() == 1
    }
}
