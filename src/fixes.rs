/// Added to spaCy's suffix pattern: a full stop after "&" and one capital splits off, so
/// "AT&T." reads by its letters and "Ask H&M." keeps its full stop. spaCy keeps one capital
/// and its full stop as one token, for initials like "J.".
pub(crate) const SUFFIX: &str = r"|(?<=&\p{Lu})\.$";

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
}
