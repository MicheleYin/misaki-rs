use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;

use crate::language::Language;
use crate::lexicon::PhonemeEntry;
use crate::token::MToken;

/// Added to spaCy's suffix pattern: a full stop after "&" and one capital splits off, so
/// "AT&T." reads by its letters and "Ask H&M." keeps its full stop. spaCy keeps one capital
/// and its full stop as one token, for initials like "J.".
pub(crate) const SUFFIX: &str = r"|(?<=&\p{Lu})\.$";

/// misaki tags a curly "n’t" as punctuation and drops the "not", and it makes quotes straight
/// only after tagging, so curly single quotes are made straight first (misaki pull request 101).
pub(crate) fn straight_quotes(text: &str) -> Cow<'_, str> {
    if text.contains(['\u{2018}', '\u{2019}']) {
        Cow::Owned(text.replace(['\u{2018}', '\u{2019}'], "'"))
    } else {
        Cow::Borrowed(text)
    }
}

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

/// In a question that opens with have, has or had, a base-form verb (`VB`) in the next three
/// words, before any "to", is retagged as the participle (`VBN`). misaki's tagger calls "read"
/// in "Have you read it?" the base form, as in "Do you read it?". A longer subject ("Have the
/// new users read it?") is out of reach.
pub(crate) fn perfect_participles(tokens: &mut [MToken], words: &[Range<usize>]) {
    let texts: Vec<String> = words
        .iter()
        .map(|w| {
            tokens[w.clone()]
                .iter()
                .map(|t| t.text.to_lowercase())
                .collect()
        })
        .collect();
    let opens = ["have", "has", "had", "haven't", "hasn't", "hadn't"];
    let mut start = 0;
    for (end, text) in texts.iter().enumerate() {
        if text.is_empty() || !text.chars().all(|c| matches!(c, '.' | '!' | '?')) {
            continue;
        }
        if text.contains('?') && opens.contains(&texts[start].as_str()) {
            for k in start + 1..end.min(start + 4) {
                if texts[k] == "to" {
                    break;
                }
                let word = &mut tokens[words[k].clone()];
                if word.iter().all(|t| t.tag == "VB") {
                    word.iter_mut().for_each(|t| t.tag = "VBN".to_string());
                }
            }
        }
        start = end + 1;
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
    fn a_curly_apostrophe_reads_as_a_straight_one() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("No, it wasn\u{2019}t.", "nˈO, ɪt wˈʌzᵊnt."),
            ("No, it wasn't.", "nˈO, ɪt wˈʌzᵊnt."),
            ("It isn\u{2019}t.", "ˌɪt ˈɪzᵊnt."),
            ("They aren\u{2019}t.", "ðˌA ˌɑɹnt."),
            ("Sam doesn\u{2019}t yet.", "sˈæm dˈʌzᵊnt jˈɛt."),
            // A curly quote around a word still reads as quotes.
            ("Click \u{2018}Save\u{2019}.", "klˈɪk “sˈAv”."),
            (
                "Choose \u{2018}Don\u{2019}t Save\u{2019}.",
                "ʧˈuz “dˈOnt sˈAv”.",
            ),
            // The cost: with straight quotes, a quoted plural before a noun reads as a possessive
            // and loses its closing mark.
            (
                "Open the \u{2018}Photos\u{2019} app.",
                "ˈOpᵊn ðə “fˈOTOz ˈæp.",
            ),
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

    #[test]
    fn a_question_that_opens_with_have_reads_the_participle() {
        let g2p = G2P::with_fallback(Language::EnglishUS, None);
        for (text, want) in [
            ("Have you read it?", "hˌæv ju ɹˈɛd ɪt?"),
            ("Has Sam read it?", "hˌæz sˈæm ɹˈɛd ɪt?"),
            ("Haven't you read it?", "hˈævᵊnt ju ɹˈɛd ɪt?"),
            ("Had she read the note?", "hˌæd ʃi ɹˈɛd ðə nˈOt?"),
            ("Has the team read it?", "hˌæz ðə tˈim ɹˈɛd ɪt?"),
            // The rule counts words: "Hasn't" is one word, though misaki holds it as two pieces.
            ("Hasn't the team read it?", "hˈæzᵊnt ðə tˈim ɹˈɛd ɪt?"),
            ("Haven't the users read it?", "hˈævᵊnt ðə jˈuzəɹz ɹˈɛd ɪt?"),
            (
                "Have you read it? Then click Next.",
                "hˌæv ju ɹˈɛd ɪt? ðˈɛn klˈɪk nˈɛkst.",
            ),
            ("Done. Have you read it?", "dˈʌn. hˌæv ju ɹˈɛd ɪt?"),
            // The rule does not fire: misaki's reading stays.
            ("Do you have to read it?", "dˈu ju hæv tə ɹˈid ɪt?"),
            ("Can you have Sam read it?", "kˌæn ju hæv sˈæm ɹˈid ɪt?"),
            ("Have you got to read it?", "hˌæv ju ɡɑt tə ɹˈid ɪt?"),
            ("Had to read it twice?", "hˌæd tə ɹˈid ɪt twˈIs?"),
            ("Have Sam read it.", "hˌæv sˈæm ɹˈid ɪt."),
            ("Did you read it?", "dˈɪd ju ɹˈid ɪt?"),
            ("You have read it.", "jˌu hæv ɹˈɛd ɪt."),
            (
                "I read the message yesterday.",
                "ˌI ɹˈɛd ðə mˈɛsɪʤ jˈɛstəɹdˌA.",
            ),
            // The costs: a subject of three words is out of reach, and a causative question takes
            // the participle too.
            (
                "Have the new users read it?",
                "hˌæv ðə nˈu jˈuzəɹz ɹˈid ɪt?",
            ),
            ("Have them read it first?", "hˌæv ðˌɛm ɹˈɛd ɪt fˈɜɹst?"),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want, "{text}");
        }
    }
}
