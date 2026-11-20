use std::collections::HashMap;

use crate::language::Language;
use crate::lexicon::PhonemeEntry;

/// Added to spaCy's suffix pattern: a full stop after "&" and one capital splits off, so
/// "AT&T." reads by its letters and "Ask H&M." keeps its full stop. spaCy keeps one capital
/// and its full stop as one token, for initials like "J.".
pub(crate) const SUFFIX: &str = r"|(?<=&\p{Lu})\.$";

/// misaki reads "dives" as the name "Dives"; it is the plural of "dive" (misaki pull request 110).
const DIVES: &str = "dˈIvz";

/// Titles and abbreviations misaki lacks, with US and GB sounds (misaki pull request 86). Left out,
/// with or without a full stop: "Ft", "Col", "Rev" and "Gen", which also mean feet, a column,
/// a revision and a generation, and "St", which is a street as often as a saint.
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

/// Adds the entries above: "dives" to gold and silver, in place of gold's "Dives", and the
/// titles to gold.
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

#[cfg(test)]
mod tests {
    use crate::{G2P, Language};

    // Expected strings are misaki 0.9.4's output with this file's rules added to it.
    #[test]
    fn a_full_stop_after_and_and_one_capital_splits_off() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("Call AT&T.", "kˈɔl ˌAtˈi ænd tˈi."),
            ("AT&T. Then more.", "ˌAtˈi ænd tˈi. ðˈɛn mˈɔɹ."),
            ("Ask H&M.", "ˈæsk ˈAʧ ænd ˈɛm."),
            ("R&D. Then more.", "ˈɑɹ ænd dˈi. ðˈɛn mˈɔɹ."),
            ("I met J. Smith.", "ˌI mˈɛt ʤˈA smˈɪθ."),
            ("The Q&A.", "ðə kjˈu ænd ˈA."),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want, "{text}");
        }
    }

    #[test]
    fn dives_is_the_plural_of_dive_and_titles_are_read_in_full() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("Take a few deep dives.", "tˈAk ɐ fjˈu dˈip dˈIvz."),
            ("Dives start at nine.", "dˈIvz stˈɑɹt æt nˈIn."),
            (
                "Acme Ltd. sent the invoice.",
                "ˈækmi lˈɪməTᵻd sˈɛnt ði ˈɪnvˌYs.",
            ),
            (
                "John Smith Jr. is calling.",
                "ʤˈɑn smˈɪθ ʤˈunjəɹ ɪz kˈɔlɪŋ.",
            ),
            ("Call Acme Co. tomorrow.", "kˈɔl ˈækmi kˈʌmpᵊni təmˈɑɹO."),
            (
                "Capt. Kirk is on the line.",
                "kˈæptᵊn kˈɜɹk ɪz ˌɔn ðə lˈIn.",
            ),
            ("Lt. Dan is here.", "lutˈɛnənt dˈæn ɪz hˈɪɹ."),
            ("Jane Doe, Esq.", "ʤˈAn dˈO, ˈɛskwˌIəɹ."),
            // spaCy splits the full stop off "Sgt.", as before.
            ("Sgt. Pepper is playing.", "sˈɑɹʤənt. pˈɛpəɹ ɪz plˈAɪŋ."),
            // Left out: a column and a revision, with or without a full stop, and a generation.
            ("Go to Col 3.", "ɡˌO tə kˈɑl θɹˈi."),
            ("Find it in col. 3.", "fˈInd ɪt ɪn kˈɑl θɹˈi."),
            ("Rev 2 is out.", "ɹˈɛv tˈu ɪz ˈWt."),
            ("See Rev. 3 for details.", "sˈi ɹˈɛv θɹˈi fɔɹ dətˈAlz."),
            ("12th Gen. Intel", "twˈɛlfθ ʤˈɛn ˈɪntˌɛl"),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want, "{text}");
        }
        let gb = G2P::with_fallback(Language::EnglishGB, None);
        for (text, want) in [
            (
                "Acme Ltd. sent the invoice.",
                "ˈakmi lˈɪmɪtɪd sˈɛnt ði ˈɪnvYs.",
            ),
            ("Take a few deep dives.", "tˈAk ɐ fjˈuː dˈiːp dˈIvz."),
        ] {
            assert_eq!(gb.g2p(text).unwrap().0, want, "{text}");
        }
    }
}
