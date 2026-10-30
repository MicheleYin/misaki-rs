//! Differential parity against misaki 0.9.4.
//!
//! misaki has no test suite of its own -- not in the pip package, not in
//! hexgrad/misaki. Its only published expected-output string, in
//! `examples/usage.py`, is stale (it documents `ə` for "a" where 0.9.4 emits
//! `ɐ`). So parity is established the only way it can be: misaki 0.9.4 is run
//! over a corpus, its output frozen into `tests/fixtures/golden_*.json` by
//! `tools/gen_fixtures.py`, and asserted here. No Python at test time.
//!
//! Two things are checked per case, and the second is not redundant:
//!   * the phoneme STRING -- what the model is fed.
//!   * the per-token (text, phonemes) pairs -- Kokoro's `pred_dur` has one
//!     entry per input token, so word->phoneme attribution is what the WORD
//!     TIMESTAMPS are derived from. A suite checking only the string could pass
//!     with the timestamps silently wrong.
//!
//! Fixtures pin the LEXICON layer: they were generated with `fallback=None`,
//! so these tests must run with `--no-default-features` (espeak off) to
//! compare like with like.
//!
//!   cargo test --no-default-features --test parity -- --nocapture

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
    misaki_version: String,
    lang: String,
    n: usize,
    cases: Vec<Case>,
}

struct Report {
    n: usize,
    ps_ok: usize,
    tok_ok: usize,
    both_ok: usize,
    first: Vec<(String, String, String)>,
}

fn run(path: &str) -> Report {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let g: Golden = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("cannot parse {path}: {e}"));
    assert_eq!(g.misaki_version, "0.9.4", "fixtures are not from misaki 0.9.4");
    assert_eq!(g.cases.len(), g.n, "fixture case count disagrees with itself");

    let lang = if g.lang == "b" { Language::EnglishGB } else { Language::EnglishUS };
    let mut engine = G2P::with_fallback(lang, None);
    // Fixtures were generated with `unk=''` (see tools/gen_fixtures.py), the
    // same value our Kokoro pipeline constructs misaki with. The crate defaults
    // to '?' which would differ on every out-of-vocabulary word.
    engine.unk = String::new();

    let mut r = Report { n: g.cases.len(), ps_ok: 0, tok_ok: 0, both_ok: 0, first: Vec::new() };
    for c in &g.cases {
        let (ps, toks) = match engine.g2p(&c.text) {
            Ok(v) => v,
            Err(e) => {
                if r.first.len() < 10 {
                    r.first.push((c.text.clone(), c.ps.clone(), format!("<error: {e:?}>")));
                }
                continue;
            }
        };
        let ps_match = ps == c.ps;
        let got: Vec<(String, Option<String>)> =
            toks.iter().map(|t| (t.text.clone(), t.phonemes.clone())).collect();
        let tok_match = got == c.tokens;

        if ps_match { r.ps_ok += 1; }
        if tok_match { r.tok_ok += 1; }
        if ps_match && tok_match { r.both_ok += 1; }
        if !tok_match {
            println!("\n  token mismatch for {:?}:\n    misaki: {:?}\n    Rust:   {:?}", c.text, c.tokens, got);
        }
        if !ps_match && r.first.len() < 30 {
            r.first.push((c.text.clone(), c.ps.clone(), ps.clone()));
        }
    }
    r
}

fn report(name: &str, r: &Report) {
    let pct = |k: usize| 100.0 * k as f64 / r.n.max(1) as f64;
    println!("\n=== {name}: {} cases ===", r.n);
    println!("  phoneme string identical : {:4}/{} ({:5.1}%)", r.ps_ok, r.n, pct(r.ps_ok));
    println!("  token pairs identical    : {:4}/{} ({:5.1}%)", r.tok_ok, r.n, pct(r.tok_ok));
    println!("  BOTH (the gate)          : {:4}/{} ({:5.1}%)", r.both_ok, r.n, pct(r.both_ok));
    if !r.first.is_empty() {
        println!("\n  first mismatches:");
        for (text, want, got) in &r.first {
            let t: String = text.chars().take(72).collect();
            let w: String = want.chars().take(88).collect();
            let g: String = got.chars().take(88).collect();
            println!("    {t}");
            println!("      misaki 0.9.4 {w}");
            println!("      misaki-rs    {g}");
        }
    }
}

#[test]
fn parity_adversarial() {
    let r = run("tests/fixtures/golden_adversarial_us.json");
    report("adversarial (heteronyms, acronyms, numbers, punctuation)", &r);
    assert_eq!(r.both_ok, r.n, "adversarial parity is not exact");
}

#[test]
fn parity_prose() {
    let r = run("tests/fixtures/golden_prose_us.json");
    report("real prose (Gutenberg)", &r);
    assert_eq!(r.both_ok, r.n, "prose parity is not exact");
}
