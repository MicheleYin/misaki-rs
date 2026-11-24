//! Straighten curly apostrophes before tagging (hexgrad/misaki#101).

use std::borrow::Cow;

/// misaki tags a curly "n’t" as punctuation and drops the "not", and it makes
/// quotes straight only after tagging, so curly single quotes are made straight
/// first (misaki pull request 101).
pub(crate) fn straight_quotes(text: &str) -> Cow<'_, str> {
    if text.contains(['\u{2018}', '\u{2019}']) {
        Cow::Owned(text.replace(['\u{2018}', '\u{2019}'], "'"))
    } else {
        Cow::Borrowed(text)
    }
}
