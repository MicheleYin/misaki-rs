//! Print the full diff for every remaining parity failure.
use misaki_rs::{G2P, Language};
use serde::Deserialize;
#[derive(Deserialize)]
struct Case { text: String, ps: String, tokens: Vec<(String, Option<String>)> }
#[derive(Deserialize)]
struct Golden { cases: Vec<Case> }
fn main() {
    let mut e = G2P::new(Language::EnglishUS); e.unk = String::new();
    for path in std::env::args().skip(1) {
        let g: Golden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for c in &g.cases {
            let Ok((ps, toks)) = e.g2p(&c.text) else { continue };
            let got: Vec<(String, Option<String>)> =
                toks.iter().map(|t| (t.text.clone(), t.phonemes.clone())).collect();
            if ps == c.ps && got == c.tokens { continue; }
            println!("TEXT  {}", c.text);
            if ps != c.ps {
                println!("  py  {}", c.ps);
                println!("  rs  {}", ps);
            }
            if got != c.tokens {
                let mut i = 0;
                while i < got.len() && i < c.tokens.len() && got[i] == c.tokens[i] { i += 1; }
                println!("  token divergence at {i}: py {:?} rs {:?}",
                    c.tokens.get(i), got.get(i));
            }
            println!();
        }
    }
}
