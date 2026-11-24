//! spaCy's English tokenizer, ported so misaki's word grouping can read the
//! tag of a spaCy token.
//!
//! misaki tokenizes with spaCy and subtokenizes within each spaCy token, and
//! its grouping rule is `tk.tag in PUNCT_TAGS` on those tokens. Everything else
//! in this crate now matches; what remained were punctuation runs where spaCy's
//! prefix/suffix peeling decides whether a mark is its own token:
//!
//! ```text
//!     "17—."            one token        "17."           17 + .
//!     "loathsome!--and"  one token        "else—but"      else + — + but
//!     "can;—but"         one token        "five-storied"  five + - + storied
//! ```
//!
//! The difference between the first two is a lookbehind in spaCy's suffix
//! pattern: the `.` rule only fires after a character in a large class that
//! includes digits and letters but not the em dash. Porting the patterns
//! verbatim (`src/spacy_tokenizer_data.rs`, taken from the live tokenizer's
//! compiled searchers) gets all of this for free; reverse-engineering the
//! behaviour case by case does not.
//!
//! The algorithm is `Tokenizer._split_affixes`, `._attach_tokens`,
//! `._tokenize_affixes`, and `._apply_special_cases` (`spacy/tokenizer.pyx`).

use std::collections::HashMap;

use fancy_regex::Regex;

use crate::spacy_tokenizer_data as data;

/// One spaCy token: its text, the whitespace that follows it, and the NORM the
/// tokenizer rule gave it, if any.
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub whitespace: String,
    pub norm: Option<String>,
}

/// A PhraseMatcher-style special-case rule: only keys that have a prefix,
/// infix, suffix, or an embedded space (spaCy `faster_heuristics=True`).
struct MatcherRule {
    /// Affix-tokenized form of the key with specials disabled.
    pattern: Vec<String>,
    pieces: &'static [(&'static str, Option<&'static str>)],
}

pub struct SpacyTokenizer {
    prefix: Regex,
    suffix: Regex,
    infix: Regex,
    url: Option<Regex>,
    specials: HashMap<&'static str, &'static [(&'static str, Option<&'static str>)]>,
    matcher: Vec<MatcherRule>,
}

impl Default for SpacyTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SpacyTokenizer {
    pub fn new() -> Self {
        let prefix = Regex::new(data::PREFIX_PATTERN).expect("prefix pattern");
        #[cfg(not(feature = "fixes"))]
        let suffix = Regex::new(data::SUFFIX_PATTERN);
        #[cfg(feature = "fixes")]
        let suffix = Regex::new(&[data::SUFFIX_PATTERN, crate::fixes::SUFFIX].concat());
        let suffix = suffix.expect("suffix pattern");
        let infix = Regex::new(data::INFIX_PATTERN).expect("infix pattern");
        let url = Regex::new(data::URL_PATTERN).ok();
        let specials: HashMap<&'static str, &'static [(&'static str, Option<&'static str>)]> =
            data::EXCEPTIONS.iter().copied().collect();

        // Build the PhraseMatcher equivalent. Keys without an affix hit or
        // space never enter the second pass (so `I'm` after `-` stays whole).
        let mut stub = SpacyTokenizer {
            prefix,
            suffix,
            infix,
            url,
            specials: specials.clone(),
            matcher: Vec::new(),
        };
        let mut matcher = Vec::new();
        for &(key, pieces) in data::EXCEPTIONS {
            let has_affix = stub.find_prefix(key) != 0
                || stub.find_suffix(key) != 0
                || stub.has_infix(key)
                || key.contains(' ');
            if !has_affix {
                continue;
            }
            let pattern = stub.affix_tokenize_key(key);
            if pattern.is_empty() {
                continue;
            }
            matcher.push(MatcherRule { pattern, pieces });
        }
        // Longer patterns first helps the scan; spaCy filters by length later.
        matcher.sort_by(|a, b| b.pattern.len().cmp(&a.pattern.len()));
        stub.matcher = matcher;
        stub
    }

    /// Byte length of the prefix to peel, or 0.
    fn find_prefix(&self, s: &str) -> usize {
        match self.prefix.find(s) {
            Ok(Some(m)) => m.end() - m.start(),
            _ => 0,
        }
    }

    /// Byte length of the suffix to peel, or 0.
    fn find_suffix(&self, s: &str) -> usize {
        match self.suffix.find(s) {
            Ok(Some(m)) => m.end() - m.start(),
            _ => 0,
        }
    }

    fn has_infix(&self, s: &str) -> bool {
        matches!(self.infix.find(s), Ok(Some(m)) if m.start() > 0)
    }

    fn is_special(&self, s: &str) -> bool {
        self.specials.contains_key(s)
    }

    fn is_url(&self, s: &str) -> bool {
        self.url
            .as_ref()
            .map(|r| matches!(r.is_match(s), Ok(true)))
            .unwrap_or(false)
    }

    /// Affix-tokenize a special-case key with specials disabled (spaCy
    /// `_tokenize_affixes(string, False)` used as the PhraseMatcher pattern).
    fn affix_tokenize_key(&self, key: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        // Whitespace inside a key is rare; split like tokenize_affixes without
        // expanding specials on the chunks.
        let mut i = 0usize;
        let chars: Vec<(usize, char)> = key.char_indices().collect();
        if chars.is_empty() {
            return out;
        }
        // For exception keys we almost always have a single non-space chunk.
        while i < key.len() {
            let ch = key[i..].chars().next().unwrap();
            if ch.is_whitespace() {
                let start = i;
                i += ch.len_utf8();
                while i < key.len() {
                    let c = key[i..].chars().next().unwrap();
                    if !c.is_whitespace() {
                        break;
                    }
                    i += c.len_utf8();
                }
                out.push(key[start..i].to_string());
                continue;
            }
            let start = i;
            while i < key.len() {
                let c = key[i..].chars().next().unwrap();
                if c.is_whitespace() {
                    break;
                }
                i += c.len_utf8();
            }
            let chunk = &key[start..i];
            let mut prefixes: Vec<&str> = Vec::new();
            let mut suffixes: Vec<&str> = Vec::new();
            let mid = self.split_affixes(chunk, &mut prefixes, &mut suffixes, false);
            self.attach_tokens(mid, &prefixes, &suffixes, &mut out, false);
        }
        out
    }

    /// `Tokenizer._split_affixes`: peel prefixes and suffixes alternately,
    /// stopping as soon as what is left is a special case (when allowed), and
    /// return the middle. Prefixes come back in order; suffixes come back
    /// OUTERMOST FIRST and are emitted reversed, as spaCy does.
    fn split_affixes<'a>(
        &self,
        mut s: &'a str,
        prefixes: &mut Vec<&'a str>,
        suffixes: &mut Vec<&'a str>,
        with_specials: bool,
    ) -> &'a str {
        let mut last_size = usize::MAX;
        while !s.is_empty() && s.len() != last_size {
            if with_specials && self.is_special(s) {
                break;
            }
            last_size = s.len();

            let pre_len = self.find_prefix(s);
            let mut minus_pre = "";
            if pre_len != 0 {
                minus_pre = &s[pre_len..];
                if with_specials && !minus_pre.is_empty() && self.is_special(minus_pre) {
                    prefixes.push(&s[..pre_len]);
                    s = minus_pre;
                    break;
                }
            }

            // spaCy searches the suffix in the string AFTER the prefix, but
            // slices it off the FULL string; identical, since it is at the end.
            let suf_len = self.find_suffix(&s[pre_len..]);
            let mut minus_suf = "";
            if suf_len != 0 {
                minus_suf = &s[..s.len() - suf_len];
                if with_specials && !minus_suf.is_empty() && self.is_special(minus_suf) {
                    suffixes.push(&s[s.len() - suf_len..]);
                    s = minus_suf;
                    break;
                }
            }

            if pre_len != 0 && suf_len != 0 && pre_len + suf_len <= s.len() {
                prefixes.push(&s[..pre_len]);
                suffixes.push(&s[s.len() - suf_len..]);
                s = &s[pre_len..s.len() - suf_len];
            } else if pre_len != 0 {
                prefixes.push(&s[..pre_len]);
                s = minus_pre;
            } else if suf_len != 0 {
                suffixes.push(&s[s.len() - suf_len..]);
                s = minus_suf;
            }
        }
        s
    }

    /// `Tokenizer._attach_tokens`: prefixes, then the middle (expanded as a
    /// special case when allowed, kept whole as a URL, or split on infixes),
    /// then the suffixes in reverse.
    fn attach_tokens(
        &self,
        mid: &str,
        prefixes: &[&str],
        suffixes: &[&str],
        out: &mut Vec<String>,
        with_specials: bool,
    ) {
        for p in prefixes {
            out.push((*p).to_string());
        }
        if !mid.is_empty() {
            if with_specials {
                if let Some(pieces) = self.specials.get(mid) {
                    for (o, _) in *pieces {
                        out.push((*o).to_string());
                    }
                    for s in suffixes.iter().rev() {
                        out.push((*s).to_string());
                    }
                    return;
                }
            }
            if self.is_url(mid) {
                out.push(mid.to_string());
            } else {
                let mut start = 0usize;
                let mut any = false;
                let mut it = self.infix.find_iter(mid);
                let mut matches: Vec<(usize, usize)> = Vec::new();
                while let Some(Ok(m)) = it.next() {
                    matches.push((m.start(), m.end()));
                }
                for (ms, me) in matches {
                    // spaCy skips a match flush against the start, so a split
                    // never produces an empty leading token.
                    if ms == 0 {
                        continue;
                    }
                    any = true;
                    if ms != start {
                        out.push(mid[start..ms].to_string());
                    }
                    if ms != me {
                        out.push(mid[ms..me].to_string());
                    }
                    start = me;
                }
                if !any {
                    out.push(mid.to_string());
                } else if start < mid.len() {
                    out.push(mid[start..].to_string());
                }
            }
        }
        for s in suffixes.iter().rev() {
            out.push((*s).to_string());
        }
    }

    /// `Tokenizer._apply_special_cases` with spaCy's PhraseMatcher +
    /// `_filter_special_spans` (longest wins; ties prefer earlier start).
    ///
    /// Input triples are `(orth, whitespace, first_pass_norm)`. First-pass
    /// norms from whole-chunk specials are kept when the matcher does not
    /// rewrite the span.
    fn apply_special_cases(
        &self,
        toks: Vec<(String, String, Option<String>)>,
    ) -> Vec<Token> {
        if toks.is_empty() {
            return Vec::new();
        }
        let texts: Vec<&str> = toks.iter().map(|(t, _, _)| t.as_str()).collect();

        // Collect matches: (start, end, rule_index). end exclusive.
        let mut matches: Vec<(usize, usize, usize)> = Vec::new();
        for (ri, rule) in self.matcher.iter().enumerate() {
            let plen = rule.pattern.len();
            if plen == 0 || plen > texts.len() {
                continue;
            }
            for start in 0..=texts.len() - plen {
                // spaCy PhraseMatcher matches adjacent tokens; whitespace
                // between them must be empty (except space tokens themselves).
                if plen > 1
                    && toks[start..start + plen - 1]
                        .iter()
                        .any(|(_, ws, _)| !ws.is_empty())
                {
                    continue;
                }
                if texts[start..start + plen]
                    .iter()
                    .zip(rule.pattern.iter())
                    .all(|(a, b)| *a == b.as_str())
                {
                    matches.push((start, start + plen, ri));
                }
            }
        }

        // `_filter_special_spans`: sort by length asc, then start desc; walk
        // from the end so longest wins; drop if start or end-1 already seen.
        matches.sort_by(|a, b| {
            let la = a.1 - a.0;
            let lb = b.1 - b.0;
            la.cmp(&lb).then_with(|| b.0.cmp(&a.0))
        });
        let mut seen = vec![false; toks.len()];
        let mut filtered: Vec<(usize, usize, usize)> = Vec::new();
        for &(start, end, ri) in matches.iter().rev() {
            if seen[start] || seen[end - 1] {
                for i in start..end {
                    seen[i] = true;
                }
                continue;
            }
            filtered.push((start, end, ri));
            for i in start..end {
                seen[i] = true;
            }
        }
        filtered.sort_by_key(|(s, _, _)| *s);

        let mut out: Vec<Token> = Vec::with_capacity(toks.len());
        let mut i = 0usize;
        let mut fi = 0usize;
        while i < toks.len() {
            if fi < filtered.len() && filtered[fi].0 == i {
                let (_start, end, ri) = filtered[fi];
                fi += 1;
                let pieces = self.matcher[ri].pieces;
                let tail_ws = toks[end - 1].1.clone();
                let last = pieces.len() - 1;
                for (k, (o, nm)) in pieces.iter().enumerate() {
                    out.push(Token {
                        text: (*o).to_string(),
                        whitespace: if k == last {
                            tail_ws.clone()
                        } else {
                            String::new()
                        },
                        norm: nm.map(|x| x.to_string()),
                    });
                }
                i = end;
            } else {
                // Prefer the first-pass special norm; else a one-piece rule
                // keyed on this orth alone.
                let norm = toks[i].2.clone().or_else(|| {
                    self.specials
                        .get(toks[i].0.as_str())
                        .filter(|p| p.len() == 1)
                        .and_then(|p| p[0].1)
                        .map(|x| x.to_string())
                });
                out.push(Token {
                    text: toks[i].0.clone(),
                    whitespace: toks[i].1.clone(),
                    norm,
                });
                i += 1;
            }
        }
        out
    }

    /// Tokenize one whitespace-delimited chunk (specials enabled).
    ///
    /// Each piece is `(orth, norm)` — first-pass specials must keep the
    /// exception norm (`’d` → `'d`), because that token never re-enters the
    /// PhraseMatcher when the key has no affix.
    fn tokenize_chunk(&self, chunk: &str, out: &mut Vec<(String, Option<String>)>) {
        if let Some(pieces) = self.specials.get(chunk) {
            for (o, nm) in *pieces {
                out.push(((*o).to_string(), nm.map(|x| x.to_string())));
            }
            return;
        }
        let mut prefixes: Vec<&str> = Vec::new();
        let mut suffixes: Vec<&str> = Vec::new();
        let mid = self.split_affixes(chunk, &mut prefixes, &mut suffixes, true);
        let mut orths: Vec<String> = Vec::new();
        self.attach_tokens(mid, &prefixes, &suffixes, &mut orths, true);
        for o in orths {
            out.push((o, None));
        }
    }

    /// Tokenize `text` the way spaCy does: affix splitting, then
    /// `_apply_special_cases`.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let raw = self.tokenize_affixes(text);
        self.apply_special_cases(raw)
    }

    /// Affix splitting only (`Tokenizer._tokenize_affixes`).
    ///
    /// Whitespace: a single ASCII space attaches to the previous token
    /// (`spacy=True`); any other whitespace run (tab, newline, NBSP, extra
    /// spaces) becomes its own token — matching spaCy's `isspace` state
    /// machine.
    fn tokenize_affixes(&self, text: &str) -> Vec<(String, String, Option<String>)> {
        let mut out: Vec<(String, String, Option<String>)> = Vec::new();
        if text.is_empty() {
            return out;
        }

        let mut start = 0usize;
        let mut in_ws = text.chars().next().map(|c| c.is_whitespace()).unwrap_or(false);
        let mut i = 0usize;

        for (idx, uc) in text.char_indices() {
            let is_ws = uc.is_whitespace();
            if is_ws != in_ws {
                if start < idx {
                    self.emit_span(&text[start..idx], &mut out);
                }
                if uc == ' ' {
                    if let Some(last) = out.last_mut() {
                        last.1.push(' ');
                    } else {
                        // Leading space with no prior token: become a space token.
                        out.push((" ".to_string(), String::new(), None));
                    }
                    start = idx + 1;
                } else {
                    start = idx;
                }
                in_ws = !in_ws;
            }
            i = idx + uc.len_utf8();
        }
        if start < text.len() {
            self.emit_span(&text[start..], &mut out);
            if let Some(last) = out.last_mut() {
                // spaCy: last.spacy = string[-1]==" " and not in_ws
                if text.ends_with(' ') && !in_ws {
                    last.1.push(' ');
                }
            }
        }
        let _ = i;
        out
    }

    fn emit_span(&self, span: &str, out: &mut Vec<(String, String, Option<String>)>) {
        if span.is_empty() {
            return;
        }
        // A pure whitespace span (tab, newline, NBSP, multi-space remainder)
        // is a single SPACE token.
        if span.chars().all(|c| c.is_whitespace()) {
            out.push((span.to_string(), String::new(), None));
            return;
        }
        let mut pieces: Vec<(String, Option<String>)> = Vec::new();
        self.tokenize_chunk(span, &mut pieces);
        for (p, nm) in pieces {
            out.push((p, String::new(), nm));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(s: &str) -> Vec<String> {
        SpacyTokenizer::new()
            .tokenize(s)
            .into_iter()
            .map(|t| t.text)
            .collect()
    }

    #[test]
    fn wed_period_merges_via_matcher() {
        assert_eq!(texts("Wed."), vec!["We", "d."]);
        assert_eq!(texts("It's Wed. now."), vec!["It", "'s", "We", "d.", "now", "."]);
    }

    #[test]
    fn contraction_after_infix_stays_whole() {
        assert_eq!(
            texts("well-I'm sure."),
            vec!["well", "-", "I'm", "sure", "."]
        );
        assert_eq!(texts("yes-don't."), vec!["yes", "-", "don't", "."]);
    }

    #[test]
    fn ten_am_after_infix_stays_whole() {
        assert_eq!(
            texts("Open from 10am-noon."),
            vec!["Open", "from", "10am", "-", "noon", "."]
        );
    }

    #[test]
    fn double_space_emits_space_token() {
        let toks = SpacyTokenizer::new().tokenize("a  b");
        assert_eq!(
            toks.iter().map(|t| (t.text.as_str(), t.whitespace.as_str())).collect::<Vec<_>>(),
            vec![("a", " "), (" ", ""), ("b", "")]
        );
    }

    #[test]
    fn tab_is_its_own_token() {
        assert_eq!(texts("Hello\tworld"), vec!["Hello", "\t", "world"]);
    }

    #[test]
    fn curly_youd_keeps_first_pass_norms() {
        let toks = SpacyTokenizer::new().tokenize("you’d played");
        assert_eq!(
            toks.iter()
                .map(|t| (t.text.as_str(), t.norm.as_deref()))
                .collect::<Vec<_>>(),
            vec![("you", Some("you")), ("’d", Some("'d")), ("played", None)]
        );
    }
}
