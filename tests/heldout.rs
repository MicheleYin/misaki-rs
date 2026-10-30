//! A HELD-OUT check against misaki 0.9.4.
//!
//! The 771 cases in `tests/parity.rs` were also what this port was developed
//! against, so they cannot measure overfitting. This corpus -- 900 sentences
//! from five books never used while fixing anything -- can, and on first run it
//! found 8 divergences the committed fixtures did not contain.
//!
//! It now asserts EXACTNESS, like the committed corpora. It did not at first:
//! the initial run found 8 divergences the fixtures did not contain, which is
//! the whole reason it exists. All eight are fixed.
//!
//! The last two were the subtlest bugs in the port, and neither could have been
//! found on the development corpora:
//!
//!   * spaCy runs `_apply_special_cases` AFTER affix splitting, re-joining a
//!     run of tokens whose text spells a special case. Peeling takes `'nuff!"`
//!     apart correctly at every step, and only that later pass reunites the
//!     quote with the word.
//!   * a tokenizer rule's NORM must travel with the RULE. `you'd` gives its
//!     second piece the norm `'d` while `you'd've` gives the same orth `would`,
//!     so no table keyed on the orth alone can be right -- and with the wrong
//!     norm the tagger called that piece an MD where spaCy says '', which
//!     decides whether the word stays joined.
use misaki_rs::{G2P, Language};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ps: String,
    tokens: Vec<(String, Option<String>)>,
}
#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[test]
fn heldout_matches_misaki() {
    let raw = std::fs::read_to_string("tests/fixtures/golden_heldout_us.json").unwrap();
    let g: Golden = serde_json::from_str(&raw).unwrap();
    let mut e = G2P::new(Language::EnglishUS);
    e.unk = String::new();

    let mut ok = 0usize;
    let mut shown = 0;
    for c in &g.cases {
        let Ok((ps, toks)) = e.g2p(&c.text) else { continue };
        let got: Vec<(String, Option<String>)> =
            toks.iter().map(|t| (t.text.clone(), t.phonemes.clone())).collect();
        if ps == c.ps && got == c.tokens {
            ok += 1;
        } else if shown < 8 {
            shown += 1;
            println!("  {}", c.text.chars().take(72).collect::<String>());
        }
    }
    println!(
        "\nheld-out: {}/{} exact ({:.1}%)",
        ok,
        g.cases.len(),
        100.0 * ok as f64 / g.cases.len() as f64
    );
    assert_eq!(ok, g.cases.len(), "held-out parity is not exact");
}
