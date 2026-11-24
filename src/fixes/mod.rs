//! Opt-in corrections to stock misaki 0.9.4 mistakes.
//!
//! Gated by Cargo feature `fixes` (off by default). Default builds keep the
//! byte-identical 0.9.4 oracle; enable `fixes` for the quality pack.
//!
//! Logic landed from @homeygravity (#30–#33); this module layout is for
//! maintainability. Re-exports keep the same hook names the pipeline calls.
//!
//! Add a new correction as a sibling module, register it in [`catalog`],
//! document it in `FIXES.md`, and re-export / call it from the hooks below.

mod ampersand_stop;
#[allow(dead_code)] // registry for FIXES.md / future tooling; read by unit tests
mod catalog;
mod curly_quotes;
mod dives_abbrevs;
mod perfect_question;

pub(crate) use ampersand_stop::SUFFIX;
pub(crate) use curly_quotes::straight_quotes;
pub(crate) use dives_abbrevs::dictionary;
pub(crate) use perfect_question::perfect_participles;

#[cfg(test)]
mod tests {
    use crate::{G2P, Language};

    // Expected strings are misaki 0.9.4's output with this pack's rules added.

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
            ("Click \u{2018}Save\u{2019}.", "klˈɪk “sˈAv”."),
            (
                "Choose \u{2018}Don\u{2019}t Save\u{2019}.",
                "ʧˈuz “dˈOnt sˈAv”.",
            ),
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
            ("Sgt. Pepper is playing.", "sˈɑɹʤənt. pˈɛpəɹ ɪz plˈAɪŋ."),
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
            ("Hasn't the team read it?", "hˈæzᵊnt ðə tˈim ɹˈɛd ɪt?"),
            ("Haven't the users read it?", "hˈævᵊnt ðə jˈuzəɹz ɹˈɛd ɪt?"),
            (
                "Have you read it? Then click Next.",
                "hˌæv ju ɹˈɛd ɪt? ðˈɛn klˈɪk nˈɛkst.",
            ),
            ("Done. Have you read it?", "dˈʌn. hˌæv ju ɹˈɛd ɪt?"),
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
            (
                "Have the new users read it?",
                "hˌæv ðə nˈu jˈuzəɹz ɹˈid ɪt?",
            ),
            ("Have them read it first?", "hˌæv ðˌɛm ɹˈɛd ɪt fˈɜɹst?"),
        ] {
            assert_eq!(g2p.g2p(text).unwrap().0, want, "{text}");
        }
    }

    #[test]
    fn catalog_lists_every_opt_in_fix() {
        assert!(
            super::catalog::FIXES
                .iter()
                .all(|f| f.status == super::catalog::FixStatus::OptIn)
        );
        let ids: Vec<_> = super::catalog::FIXES.iter().map(|f| f.id).collect();
        assert_eq!(
            ids,
            [
                super::catalog::FixId::AmpersandStop,
                super::catalog::FixId::CurlyQuotes,
                super::catalog::FixId::DivesAbbrevs,
                super::catalog::FixId::PerfectQuestion,
            ]
        );
        for entry in super::catalog::FIXES {
            assert!(!entry.name.is_empty());
            assert!(!entry.behavior.is_empty());
            assert!(!entry.upstream.is_empty());
        }
    }
}
