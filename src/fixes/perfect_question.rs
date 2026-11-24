//! Retag VB→VBN in short perfect questions (`Have you read it?`).

use std::ops::Range;

use crate::token::MToken;

/// In a question that opens with have, has or had, a base-form verb (`VB`) in
/// the next three words, before any "to", is retagged as the participle
/// (`VBN`). misaki's tagger calls "read" in "Have you read it?" the base form,
/// as in "Do you read it?". A longer subject ("Have the new users read it?")
/// is out of reach.
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
