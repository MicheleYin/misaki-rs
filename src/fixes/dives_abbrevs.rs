//! Lexicon overlays for `dives` and missing titles (hexgrad/misaki#110, #86).

use std::collections::HashMap;

use crate::language::Language;
use crate::lexicon::PhonemeEntry;

/// misaki reads "dives" as the name "Dives"; it is the plural of "dive"
/// (misaki pull request 110).
const DIVES: &str = "dˈIvz";

/// Titles and abbreviations misaki lacks, with US and GB sounds (misaki pull
/// request 86). Left out, with or without a full stop: "Ft", "Col", "Rev" and
/// "Gen", which also mean feet, a column, a revision and a generation, and
/// "St", which is a street as often as a saint.
const TITLES: &[(&str, &str, &str)] = &[
    ("Capt", "kˈæptᵊn", "kˈaptɪn"),
    ("Capt.", "kˈæptᵊn", "kˈaptɪn"),
    ("Co.", "kˈʌmpᵊni", "kˈʌmpᵊni"),
    ("Esq", "ˈɛskwˌIəɹ", "ɪskwˈIə"),
    ("Hon", "ˈɑnəɹəbᵊl", "ˈɒnəɹəbᵊl"),
    ("Hon.", "ˈɑnəɹəbᵊl", "ˈɒnəɹəbᵊl"),
    ("Jr", "ʤˈunjəɹ", "ʤˈuːnɪə"),
    ("Jr.", "ʤˈunjəɹ", "ʤˈuːnɪə"),
    ("Lt", "lutˈɛnənt", "lɛftˈɛnənt"),
    ("Lt.", "lutˈɛnənt", "lɛftˈɛnənt"),
    ("Ltd", "lˈɪməɾᵻd", "lˈɪmɪtɪd"),
    ("Ltd.", "lˈɪməɾᵻd", "lˈɪmɪtɪd"),
    ("Maj", "mˈAʤəɹ", "mˈAʤə"),
    ("Maj.", "mˈAʤəɹ", "mˈAʤə"),
    ("Sgt", "sˈɑɹʤənt", "sˈɑːʤᵊnt"),
    ("Sgt.", "sˈɑɹʤənt", "sˈɑːʤᵊnt"),
];

/// Adds the entries above: "dives" to gold and silver, in place of gold's
/// "Dives", and the titles to gold.
pub(crate) fn dictionary(
    golds: &mut HashMap<String, PhonemeEntry>,
    silvers: &mut HashMap<String, PhonemeEntry>,
    lang: Language,
) {
    golds.remove("Dives");
    for table in [&mut *golds, &mut *silvers] {
        table.insert("dives".to_owned(), PhonemeEntry::Simple(DIVES.to_owned()));
    }
    for &(key, us, gb) in TITLES {
        let ps = match lang {
            Language::EnglishUS => us,
            Language::EnglishGB => gb,
        };
        golds.insert(key.to_owned(), PhonemeEntry::Simple(ps.to_owned()));
    }
}
