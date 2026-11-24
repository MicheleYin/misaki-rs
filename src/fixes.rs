use std::ops::Range;

use crate::token::MToken;

/// Added to spaCy's suffix pattern: a full stop after "&" and one capital splits off, so
/// "AT&T." reads by its letters and "Ask H&M." keeps its full stop. spaCy keeps one capital
/// and its full stop as one token, for initials like "J.".
pub(crate) const SUFFIX: &str = r"|(?<=&\p{Lu})\.$";

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
