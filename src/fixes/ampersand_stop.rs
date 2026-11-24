//! Peel a full stop after `&` and one capital (`AT&T.`).

/// Added to spaCy's suffix pattern: a full stop after "&" and one capital splits
/// off, so "AT&T." reads by its letters and "Ask H&M." keeps its full stop.
/// spaCy keeps one capital and its full stop as one token, for initials like "J.".
pub(crate) const SUFFIX: &str = r"|(?<=&\p{Lu})\.$";
