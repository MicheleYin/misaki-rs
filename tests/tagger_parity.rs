//! Does the ported spaCy tagger reproduce spaCy's tags?
//!
//! Run on spaCy's OWN tokens (fixtures carry them), so this isolates the model
//! from the tokenizer. Anything less than 100% here is a bug in the port;
//! disagreement inside the G2P pipeline on top of 100% here is a tokenization
//! difference instead.
use misaki_rs::spacy_tagger::Tagger;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    tokens: Vec<(String, String, bool)>,
    tags: Vec<String>,
    feats: Vec<Vec<u64>>,
}
#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

/// Do the six feature columns we compute match spaCy's own `extract_features`?
/// Checked separately so a tag mismatch can be blamed on the right half.
#[test]
fn features_match_spacy() {
    let raw = std::fs::read_to_string("tests/fixtures/golden_tags.json").unwrap();
    let g: Golden = serde_json::from_str(&raw).unwrap();
    const NAMES: [&str; 6] = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"];
    let mut bad = [0usize; 6];
    let mut tot = 0usize;
    let mut shown = 0;
    for c in &g.cases {
        if c.feats.len() != c.tokens.len() {
            continue;
        }
        for (tk, want) in c.tokens.iter().zip(&c.feats) {
            let got = misaki_rs::spacy_tagger::debug_features(&tk.0, &tk.1, tk.2);
            tot += 1;
            for k in 0..6 {
                if got[k] != want[k] {
                    bad[k] += 1;
                    if shown < 14 {
                        shown += 1;
                        println!("  {:<14?} {:<9} got {} want {}", tk.0, NAMES[k], got[k], want[k]);
                    }
                }
            }
        }
    }
    println!("\nfeature columns over {tot} tokens:");
    for k in 0..6 {
        println!("  {:<9} mismatches {}", NAMES[k], bad[k]);
    }
    assert_eq!(bad.iter().sum::<usize>(), 0, "feature computation differs from spaCy");
}

#[test]
fn tagger_matches_spacy() {
    let raw = std::fs::read_to_string("tests/fixtures/golden_tags.json").unwrap();
    let g: Golden = serde_json::from_str(&raw).unwrap();
    let t = Tagger::load();

    let (mut tok_ok, mut tok_tot, mut sent_ok) = (0usize, 0usize, 0usize);
    let mut shown = 0;
    for c in &g.cases {
        let got = t.tag(&c.tokens);
        let ok = got.len() == c.tags.len() && got.iter().zip(&c.tags).all(|(a, b)| *a == b.as_str());
        if ok {
            sent_ok += 1;
        }
        for (a, b) in got.iter().zip(&c.tags) {
            tok_tot += 1;
            if *a == b.as_str() {
                tok_ok += 1;
            } else if shown < 12 {
                shown += 1;
                println!("  {:?}  got {} want {}", c.text.chars().take(50).collect::<String>(), a, b);
            }
        }
    }
    println!(
        "\ntagger vs spaCy: {}/{} tokens ({:.3}%), {}/{} sentences",
        tok_ok,
        tok_tot,
        100.0 * tok_ok as f64 / tok_tot as f64,
        sent_ok,
        g.cases.len()
    );
    assert_eq!(tok_ok, tok_tot, "ported tagger does not match spaCy");
}
