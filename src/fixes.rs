use std::borrow::Cow;

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
}
