//! Machine-readable registry of opt-in phonemizer corrections.
//! Keep in sync with `FIXES.md`.

/// Stable id for one correction in the `fixes` pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixId {
    AmpersandStop,
    CurlyQuotes,
    DivesAbbrevs,
    PerfectQuestion,
}

/// Lifecycle status for a correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixStatus {
    /// Available only with Cargo feature `fixes`.
    OptIn,
    /// Moved into the stock path after the parity oracle advanced.
    Promoted,
    /// Removed / superseded.
    Removed,
}

/// One row in the fixes catalog.
#[derive(Debug, Clone, Copy)]
pub struct FixEntry {
    pub id: FixId,
    pub name: &'static str,
    pub behavior: &'static str,
    pub upstream: &'static str,
    pub status: FixStatus,
}

/// All corrections currently tracked. Order is stable for docs and tests.
pub const FIXES: &[FixEntry] = &[
    FixEntry {
        id: FixId::AmpersandStop,
        name: "ampersand_stop",
        behavior: "peel `.` after `&`+capital so `AT&T.` letter-reads and keeps the stop",
        upstream: "none (local; misaki-rs #30)",
        status: FixStatus::OptIn,
    },
    FixEntry {
        id: FixId::CurlyQuotes,
        name: "curly_quotes",
        behavior: "straighten curly single quotes before POS tagging",
        upstream: "https://github.com/hexgrad/misaki/pull/101 (misaki-rs #31)",
        status: FixStatus::OptIn,
    },
    FixEntry {
        id: FixId::DivesAbbrevs,
        name: "dives_abbrevs",
        behavior: "read `dives` as plural of dive; add selected title abbreviations",
        upstream: "https://github.com/hexgrad/misaki/pull/110 , https://github.com/hexgrad/misaki/pull/86 (misaki-rs #32)",
        status: FixStatus::OptIn,
    },
    FixEntry {
        id: FixId::PerfectQuestion,
        name: "perfect_question",
        behavior: "retag VB→VBN in short Have/Has/Had…? questions",
        upstream: "none (local; misaki-rs #33)",
        status: FixStatus::OptIn,
    },
];
