use crate::lexicon::PhonemeEntry;
use serde_json;
use std::collections::HashMap;

pub fn load_us_gold() -> HashMap<String, PhonemeEntry> {
    let data = include_str!("../data/us_gold.json");
    serde_json::from_str(data).expect("Failed to parse us_gold.json")
}

#[cfg(feature = "silver")]
pub fn load_us_silver() -> HashMap<String, PhonemeEntry> {
    let data = include_str!("../data/us_silver.json");
    serde_json::from_str(data).expect("Failed to parse us_silver.json")
}

#[cfg(not(feature = "silver"))]
pub fn load_us_silver() -> HashMap<String, PhonemeEntry> {
    HashMap::new()
}

#[cfg(feature = "gb")]
pub fn load_gb_gold() -> HashMap<String, PhonemeEntry> {
    let data = include_str!("../data/gb_gold.json");
    serde_json::from_str(data).expect("Failed to parse gb_gold.json")
}

#[cfg(all(feature = "gb", feature = "silver"))]
pub fn load_gb_silver() -> HashMap<String, PhonemeEntry> {
    let data = include_str!("../data/gb_silver.json");
    serde_json::from_str(data).expect("Failed to parse gb_silver.json")
}

#[cfg(all(feature = "gb", not(feature = "silver")))]
pub fn load_gb_silver() -> HashMap<String, PhonemeEntry> {
    HashMap::new()
}

// pub fn load_it_gold() -> HashMap<String, PhonemeEntry> {
//     let data = include_str!("../data/it_gold.json");
//     serde_json::from_str(data).expect("Failed to parse it_gold.json")
// }

// pub fn load_it_silver() -> HashMap<String, PhonemeEntry> {
//     let data = include_str!("../data/it_silver.json");
//     serde_json::from_str(data).expect("Failed to parse it_silver.json")
// }
