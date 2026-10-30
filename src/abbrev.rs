//! spaCy's English tokenizer exceptions that end in a period.
//!
//! misaki tokenizes with spaCy, which keeps `Mr.` and `a.m.` as single tokens
//! while splitting `etc.` into `etc` + `.`. Our tokenizer splits on the period
//! unconditionally, so without this list `Mr.` arrived as two tokens and
//! `Mr.`/`Mrs.` alone accounted for 39 of the token-count divergences against
//! misaki 0.9.4.
//!
//! Extracted verbatim from `spacy.load("en_core_web_sm").tokenizer.rules`
//! (the `.`-terminated, alphabetic subset), so this is data, not a guess.
//! Matching is CASE-SENSITIVE, as spaCy's rules are: the list carries `E.g.`
//! and `E.G.` as separate entries.

pub const PERIOD_ABBREVIATIONS: &[&str] = &[
    "10a.m.",
    "10p.m.",
    "11a.m.",
    "11p.m.",
    "12a.m.",
    "12p.m.",
    "1a.m.",
    "1p.m.",
    "2a.m.",
    "2p.m.",
    "3a.m.",
    "3p.m.",
    "4a.m.",
    "4p.m.",
    "5a.m.",
    "5p.m.",
    "6a.m.",
    "6p.m.",
    "7a.m.",
    "7p.m.",
    "8a.m.",
    "8p.m.",
    "9a.m.",
    "9p.m.",
    "Adm.",
    "Ak.",
    "Ala.",
    "Apr.",
    "Ariz.",
    "Ark.",
    "Aug.",
    "Bros.",
    "Calif.",
    "Co.",
    "Colo.",
    "Conn.",
    "Corp.",
    "D.C.",
    "Dec.",
    "Del.",
    "Dr.",
    "E.G.",
    "E.g.",
    "Feb.",
    "Fla.",
    "Ga.",
    "Gen.",
    "Gov.",
    "I.E.",
    "I.e.",
    "Ia.",
    "Id.",
    "Ill.",
    "Inc.",
    "Ind.",
    "Jan.",
    "Jr.",
    "Jul.",
    "Jun.",
    "Kan.",
    "Kans.",
    "Ky.",
    "La.",
    "Ltd.",
    "Mar.",
    "Mass.",
    "Md.",
    "Messrs.",
    "Mich.",
    "Minn.",
    "Miss.",
    "Mo.",
    "Mont.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Mt.",
    "N.C.",
    "N.D.",
    "N.H.",
    "N.J.",
    "N.M.",
    "N.Y.",
    "Neb.",
    "Nebr.",
    "Nev.",
    "Nov.",
    "Oct.",
    "Okla.",
    "Ore.",
    "Pa.",
    "Ph.D.",
    "Prof.",
    "Rep.",
    "Rev.",
    "S.C.",
    "Sen.",
    "Sep.",
    "Sept.",
    "St.",
    "Tenn.",
    "Va.",
    "Wash.",
    "Wis.",
    "a.",
    "a.m.",
    "b.",
    "c.",
    "co.",
    "d.",
    "e.",
    "e.g.",
    "f.",
    "g.",
    "h.",
    "i.",
    "i.e.",
    "j.",
    "k.",
    "l.",
    "m.",
    "n.",
    "o.",
    "p.",
    "p.m.",
    "q.",
    "r.",
    "s.",
    "t.",
    "u.",
    "v.",
    "v.s.",
    "vs.",
    "w.",
    "x.",
    "y.",
    "z.",
    "°C.",
    "°F.",
    "°K.",
    "°c.",
    "°f.",
    "°k.",
    "ä.",
    "ö.",
    "ü.",
];

/// Is `text` a token spaCy would keep whole rather than splitting its period?
///
/// Two rules, both spaCy's:
///   * it appears in `PERIOD_ABBREVIATIONS` (an explicit tokenizer exception);
///   * it is a single uppercase letter followed by a period, i.e. an initial
/// ```text
///     such as `K.` or `V.`. spaCy keeps these via a SUFFIX rule rather than
///     an exception, so they are absent from the list above.
/// ```
pub fn is_period_abbreviation(text: &str) -> bool {
    if PERIOD_ABBREVIATIONS.contains(&text) {
        return true;
    }
    let mut cs = text.chars();
    match (cs.next(), cs.next(), cs.next()) {
        (Some(c), Some('.'), None) => c.is_uppercase(),
        _ => false,
    }
}

/// Apostrophe-initial pieces spaCy's tokenizer emits as single tokens, from
/// `nlp.tokenizer.rules` (the multi-character subset; the bare `'` and `’` need
/// no merging because our tokenizer already emits them alone).
///
/// This matters for GROUPING, not pronunciation. spaCy tokenizes `Lydia’s` as
/// `Lydia` + `’s` and tags that piece POS, which is not a PUNCT_TAG, so misaki
/// merges it back into one word. Our subtokenizer emits `Lydia`, `’`, `s`; left
/// alone, the tagger sees a lone `’`, calls it `''`, and the word splits.
pub const CONTRACTION_PIECES: &[&str] = &[
    "''",
    "'Cause",
    "'Cos",
    "'Coz",
    "'Cuz",
    "'S",
    "'bout",
    "'cause",
    "'cos",
    "'coz",
    "'cuz",
    "'d",
    "'em",
    "'ll",
    "'m",
    "'nuff",
    "'re",
    "'s",
    "'ve",
    "'y",
    "n't",
    "n’t",
    "‘S",
    "‘s",
    "’Cause",
    "’Cos",
    "’Coz",
    "’Cuz",
    "’S",
    "’bout",
    "’cause",
    "’cos",
    "’coz",
    "’cuz",
    "’d",
    "’em",
    "’ll",
    "’m",
    "’nuff",
    "’re",
    "’s",
    "’ve",
    "’y",
    "’’",
];

/// Is `text` one of spaCy's apostrophe-initial tokenizer pieces?
pub fn is_contraction_piece(text: &str) -> bool {
    CONTRACTION_PIECES.contains(&text)
}
