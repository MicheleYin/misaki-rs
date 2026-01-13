# Misaki-RS

**misaki-rs** is a self-contained, high-performance Rust port of the [Misaki](https://github.com/hexgrad/misaki) G2P (Grapheme-to-Phoneme) engine. 

It is specifically designed for use with TTS models like **Kokoro**, providing accurate Part-of-Speech aware phonemization for English text.

## Features

- **Self-Contained**: All lexicons, dictionaries, and Part-of-Speech tagger weights are embedded directly into the binary at compile time. No external resource files are required at runtime.
- **POS-Aware Phonemization**: Uses an averaged perceptron tagger to handle heteronyms (words with different pronunciations based on context, e.g., *object* as a noun vs. verb).
- **Multi-Dialect Support**: Supports both **US English** (en-us) and **British English** (en-gb).
- **Morphological Stemming**: Intelligent handling of suffixes (plurals, past tense, continuous tense). Other rules may be added in the future. Currently those are:
    -  s plural stemming
    - ed past tense stemming
    - ing continuous tense stemming
- **Number Conversion**: Automatically converts numeric values into their spoken word equivalents.


## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
misaki-rs = "0.1.1"
```

## Quick Start

```rust
use misaki_rs::G2P;

fn main() {
    // Initialize for US English (false = US, true = GB)
    let g2p = G2P::new(false); 
    
    let (phonemes, tokens) = g2p.g2p("Hello, world! 123");
    println!("US Phonemes: {}", phonemes);
    
    // Initialize for British English
    let g2p_gb = G2P::new(true);
    let (phonemes_gb, _) = g2p_gb.g2p("The schedule is full.");
    println!("GB Phonemes: {}", phonemes_gb);
}
```

## Pronunciations

The original misaki project had very few words and some were not pronunced correctly. Here I updated the original pronunciation dict to include more words and correct pronunciations using eSpeak.

## Scope

This repository aims to provide a lightweight and efficient alternative to ONNX-based phonemizers for Rust applications. It eliminates the need for external C++ dependencies or large model files by porting the logic and data into native Rust.

## License

This project is based on the original Misaki library. See the original repository for licensing details regarding the underlying dictionary data.
