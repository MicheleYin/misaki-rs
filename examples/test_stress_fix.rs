use misaki_rs::{G2P, Language};

fn main() {
    // Test that unknown words use espeak fallback
    let test_unknown = "ilili xyzabc fantabulous";
    let g2p = G2P::new(Language::EnglishUS);
    let (phonemes, _) = g2p.g2p(test_unknown).unwrap();

    println!("Unknown words test: {}", phonemes);

    // Verify no character-by-character spelling
    // US English G2P spelling for 'ilili' would include 'L' -> 'ˈɛl'
    assert!(!phonemes.contains("ˈɛl "), "Should not spell out letters");
    // 'xyzabc' would include 'Y' -> 'ˈwaɪ'
    assert!(!phonemes.contains("ˈwaɪ "), "Should not spell out letters");

    // Verify NO unknown markers (espeak handles everything)
    assert!(!phonemes.contains("❓"), "espeak should phonemize all words");

    println!("Success!");
}
