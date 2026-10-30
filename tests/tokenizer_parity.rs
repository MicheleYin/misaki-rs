//! Does the ported spaCy tokenizer reproduce spaCy's tokenization?
//!
//! misaki tokenizes with spaCy and groups subtokens by the TAG of a spaCy
//! token, so this has to be exact for the grouping to be exact.
use misaki_rs::spacy_tokenizer::SpacyTokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    tokens: Vec<(String, String)>,
}
#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[test]
fn tokenizer_matches_spacy() {
    let raw = std::fs::read_to_string("tests/fixtures/golden_tokens.json").unwrap();
    let g: Golden = serde_json::from_str(&raw).unwrap();
    let tk = SpacyTokenizer::new();

    let (mut tok_ok, mut tok_tot, mut sent_ok) = (0usize, 0usize, 0usize);
    let mut shown = 0;
    for c in &g.cases {
        let got = tk.tokenize(&c.text);
        let ok = got.len() == c.tokens.len()
            && got
                .iter()
                .zip(&c.tokens)
                .all(|(a, b)| a.text == b.0 && a.whitespace == b.1);
        if ok {
            sent_ok += 1;
        } else if shown < 10 {
            shown += 1;
            println!("  {:?}", c.text.chars().take(64).collect::<String>());
            println!("    spacy {:?}", c.tokens.iter().map(|t| t.0.as_str()).collect::<Vec<_>>());
            println!("    rs    {:?}", got.iter().map(|t| t.text.as_str()).collect::<Vec<_>>());
        }
        let n = got.len().max(c.tokens.len());
        for i in 0..n {
            tok_tot += 1;
            if got.get(i).map(|t| (&t.text, &t.whitespace)) == c.tokens.get(i).map(|t| (&t.0, &t.1)) {
                tok_ok += 1;
            }
        }
    }
    println!(
        "\ntokenizer vs spaCy: {}/{} tokens ({:.3}%), {}/{} sentences",
        tok_ok, tok_tot, 100.0 * tok_ok as f64 / tok_tot as f64, sent_ok, g.cases.len()
    );
    assert_eq!(sent_ok, g.cases.len(), "ported tokenizer does not match spaCy");
}
