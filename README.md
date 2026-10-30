# Misaki-RS

[![Crates.io](https://img.shields.io/crates/v/misaki-rs)](https://crates.io/crates/misaki-rs)


**misaki-rs** is a self-contained Rust port of the [Misaki](https://github.com/hexgrad/misaki) G2P (Grapheme-to-Phoneme) engine, with its English pipeline brought to feature parity with Misaki 0.9.4.

It is designed for TTS models like **Kokoro**, providing Part-of-Speech-aware phonemization for US and British English.

## Features

- **Self-Contained**: All lexicons, dictionaries, and Part-of-Speech tagger weights are embedded directly into the binary at compile time. No external resource files are required at runtime.
- **Misaki Compatibility**: Ports Misaki 0.9.4's English phonemization behavior, including its spaCy-compatible tokenizer, normalization, and POS tagger.
- **POS-Aware Phonemization**: Handles heteronyms (words with different pronunciations based on context, e.g., *object* as a noun vs. verb).
- **Multi-Dialect Support**: Supports both **US English** (en-us) and **British English** (en-gb).
- **Morphological Stemming**: Intelligent handling of suffixes (plurals, past tense, continuous tense). Other rules may be added in the future. Currently those are:
    -  s plural stemming
    - ed past tense stemming
    - ing continuous tense stemming
- **Number Conversion**: Automatically converts numeric values into their spoken word equivalents.
- **Optional espeak fallback** (feature `espeak`, enabled by default): Uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) for out-of-vocabulary words. Disable with `default-features = false` for a build without the espeak dependency; unresolved words are marked with `G2P::unk` (❓ by default), not spelled letter-by-letter.


## Unknown words and fallback

With the default `espeak` feature enabled, an out-of-vocabulary word is sent to espeak-ng and its IPA output is converted to the engine's phoneme set. If espeak is disabled or unavailable, unresolved words keep `None` phonemes and the assembled phoneme string uses the configured unknown marker (`G2P::unk`, ❓ by default). You can provide a different fallback with `G2P::with_fallback`.


## Testing espeak

To check that espeak fallback is working, phonemize an out-of-vocabulary word like `"eBook"` and assert it does not contain the unknown marker and is not spelled letter-by-letter:

```bash
cargo test test_ebook_with_espeak -- --nocapture
```

You should see output like `eBook (with espeak): ˈi bˈʊk.` (word-like). Without the espeak feature, the same word is spelled out: `cargo test test_ebook_without_espeak --no-default-features -- --nocapture` gives e.g. `eBook (without espeak): ˈiː  bˈi  ˈo‍ʊ  ˈo‍ʊ  kˈe‍ɪ` (E, B, O, O, K as letter names).


## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
misaki-rs = "0.7.0"
```

**Optional: disable espeak fallback** (smaller build, no espeak-ng dependency):

```toml
[dependencies]
misaki-rs = { version = "0.7.0", default-features = false }
```

To depend on misaki-rs without default features but still use espeak:

```toml
misaki-rs = { version = "0.7.0", default-features = false, features = ["espeak"] }
```

## Quick Start

```rust
use misaki_rs::{G2P, Language};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let g2p = G2P::new(Language::EnglishUS);
    let (phonemes, tokens) = g2p.g2p("Hello, world! 123")?;
    println!("US Phonemes: {}", phonemes);

    let g2p_gb = G2P::new(Language::EnglishGB);
    let (phonemes_gb, _) = g2p_gb.g2p("The schedule is full.")?;
    println!("GB Phonemes: {}", phonemes_gb);

    Ok(())
}
```

## Custom OOV fallback

If you want to override the default out-of-vocabulary behavior, inject your own fallback implementation by implementing the `Fallback` trait and passing it to `G2P::with_fallback`.

```rust
use misaki_rs::{fallback::FallbackError, Fallback, G2P, Language};

struct CustomFallback;

impl Fallback for CustomFallback {
    fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
        // Replace this with your own phoneme generation logic.
        Ok(format!("custom:{word}"))
    }
}

fn main() {
    let g2p = G2P::with_fallback(Language::EnglishUS, Some(Box::new(CustomFallback)));
    let (phonemes, _) = g2p.g2p("qzxwv");
    println!("{}", phonemes);
}
```

This is useful when you want a custom pronunciation source, a dictionary lookup, or a second-pass rule engine for unknown words instead of the built-in `espeak` fallback.

## Pronunciations

The original misaki project had very few words and some were not pronunced correctly. Here I updated the original pronunciation dict to include more words and correct pronunciations using eSpeak.

## Misaki parity

The English pipeline is tested against Misaki 0.9.4 using frozen differential fixtures. The US English phoneme output and per-token phoneme attribution match on 71 adversarial examples, 700 prose examples, and 900 held-out examples. The spaCy-compatible tokenizer and POS tagger are also checked against 20,623 tokens across 771 sentences.

Run the parity suite with:

```bash
cargo test --no-default-features
```

See [PARITY.md](PARITY.md) for the test methodology and coverage details.

## Scope

This repository aims to provide a lightweight and efficient alternative to ONNX-based phonemizers for Rust applications. It eliminates the need for external C++ dependencies or large model files by porting the logic and data into native Rust.

## License

This project is based on the original Misaki library. See the original repository for licensing details regarding the underlying dictionary data.
