# Parity with misaki 0.9.4

This fork exists to make misaki-rs produce **byte-identical** output to
**misaki 0.9.4**, the G2P that Kokoro v1.0 was trained against, so a Rust
serving binary can replace a Python one without changing pronunciation or word
timestamps.

## Why differential testing, and not replaying upstream commits

Two facts settle the method:

1. **misaki 0.9.4 is the end of the line.** It is the latest PyPI release
   (2025-04-05) *and* `hexgrad/misaki` HEAD still reads
   `__version__ = '0.9.4'`. Only three commits exist after the release — Python
   3.13 support, a pip dependency fix, and an optional transformer fallback —
   and none bump the version. There is no forward history to catch up on.

2. **The divergences are not version drift.** Of the six found so far, five
   were never any misaki version's behaviour; they are shortcuts and
   inventions in the port:

   | divergence | drift, or port artefact? |
   |---|---|
   | lexicons from a different release | **drift** |
   | missing `ɾ`→`T`, `ʔ`→`t` pass | port omission (it is in 0.9.4's `en.py`) |
   | `whitespace` hardcoded to `" "` | port bug — no misaki ever did this |
   | `future_vowel` from spelling (`"aeiou"`) | port shortcut, self-labelled "simple heuristic" |
   | char-by-char spell-out for OOV words | port invention |
   | punctuation reduced to `;:,.!?` | port shortcut |

   A commit replay would have found exactly one of these.

So misaki 0.9.4 is used as an **oracle** instead: run it over a corpus, freeze
the output, assert against it from Rust. That also covers the rare paths a
changelog would never mention.

## The gate

```sh
cargo test --no-default-features --test parity -- --nocapture
```

`--no-default-features` is required: the fixtures were generated with
`fallback=None`, so espeak must be off to compare like with like.

Two properties are asserted per case:

* the **phoneme string** — what the model is fed;
* the **per-token `(text, phonemes)` pairs** — Kokoro's `pred_dur` has one entry
  per input token, so word→phoneme attribution is what the **word timestamps**
  are derived from. A suite checking only the string could pass with the
  timestamps silently wrong.

Fixtures are regenerated with `tools/gen_fixtures.py` against a venv holding
misaki 0.9.4; `tools/build_corpus.py` assembles the corpora. No Python is
needed to *run* the tests.

## Status: exact

```
adversarial    71/71   (100.0%)    tests/parity.rs
real prose    700/700  (100.0%)    tests/parity.rs
held-out      900/900  (100.0%)    tests/heldout.rs
tagger      20623/20623 tokens, 771/771 sentences
tokenizer   20623/20623 tokens, 771/771 sentences
lib                24 tests        doctests 0 failures
```

All three corpora match misaki 0.9.4 **byte for byte**, on the phoneme string
and on the per-token `(text, phonemes)` attribution that the word timestamps
are derived from. The second half is not redundant: three separate
bugs produced identical audio with the attribution shifted, and only the
token-pair check saw them.

## How it was done

By differential testing, not by reading. misaki 0.9.4 has no test suite -- not
in the pip package, not in `hexgrad/misaki`, and its one published
expected-output string is stale (`examples/usage.py` documents `ə` for the
article "a" where 0.9.4 emits `ɐ`). So misaki itself is the oracle:
`tools/gen_fixtures.py` runs it over a corpus and freezes the output, and the
tests assert against that with no Python at test time.

Every fix below was found by that gate.

### The two things that had to be ported, not approximated

Roughly half the work was discovering that misaki's behaviour is not
self-contained: it tokenizes AND tags with spaCy, and its rule for grouping
subtokens back into words reads the **tag of a spaCy token**. No character
heuristic substitutes for that, which the gate proved repeatedly:

| | spaCy's tag | misaki | a character rule |
|---|---|---|---|
| `_` in `_like` | `UH` | merges | cannot tell |
| `_` in `_it` | `NFP` | stands alone | these apart |
| `'` in `dogs'` | `POS` | merges | cannot tell |
| `’` in `Thomas’` | `''` | stands alone | these apart |

So both halves of spaCy are ported and separately verified:

* **`src/spacy_tagger.rs`** — MultiHashEmbed + 4x residual MaxoutWindowEncoder +
  softmax over 50 tags, 1,570,514 f32 in `data/tagger.bin`. Exact.
* **`src/spacy_tokenizer.rs`** — `_split_affixes` / `_attach_tokens` over the
  real prefix, suffix and infix patterns and 1,347 exceptions. Exact.

Four details in the tagger cost real accuracy and none are visible in the
architecture:

* Thinc's HashEmbed hash is **not** generic MurmurHash3 — it is hand-specialised
  for a single `u64`, ~25 lines; a stock crate does not reproduce it.
* spaCy's StringStore does not hash every string: **457 symbols** have fixed
  small IDs (the shape `X` is 101), worth ~0.7%.
* **`with_array(pad=4)`** wraps the encoder in four zero rows per side. Not the
  same as zero-padding inside `expand_window`, because the pad rows pass
  through maxout and layernorm. Omitting it held agreement at 99.0%.
* `word_shape` tests `isalpha()` **before** case, so a caseless letter is `x`.

And NORM is not lowercasing: it comes from `BASE_NORMS`, the tokenizer
exceptions and a 3,510-entry `lexeme_norm` table. The entry that matters most is
the curly apostrophe normalising to the straight one — with the wrong NORM the
tagger calls a possessive `''`, a PUNCT_TAG, and splits every `Alice’s` apart.

### Bugs in the fork worth naming

Several were not drift from a newer misaki but inventions or slips:

* `whitespace` hardcoded to `" "`. The output is `''.join(phonemes +
  whitespace)`, so **no** phoneme string could ever match.
* The subtoken regex's word-internal apostrophe class held three copies of the
  STRAIGHT apostrophe where misaki has straight plus both curly, so every curly
  possessive in real prose came apart. Its two lookahead alternatives were also
  missing, which is what splits camel case.
* `future_vowel` guessed from spelling, and the guess **overwrote** the correct
  value computed from phonemes, making the real context pass dead code.
* The stemmers lowercased first, which breaks the all-caps acronym rule in
  `is_known` — so every plural acronym resolved to silence.
* `lookup` had an extra `if is_nnp` guard around `get_NNP`, so an unknown word
  could only be spelled out when its tag was literally NNP.
* `resolve_phonemes` treated an explicit `None` in a POS-keyed entry as "fall
  back to DEFAULT" rather than "no pronunciation for this tag".
* An out-of-vocabulary word was spelled out character by character. misaki
  leaves it unresolved; spelling it made one token into six and corrupted every
  downstream timestamp.

## The held-out corpus, and why it exists

`tests/heldout.rs` runs 900 sentences from five books never used while fixing
anything (Sherlock Holmes, A Tale of Two Cities, Dracula, The Prince, Tom
Sawyer). The committed fixtures were also the development target, so they
cannot measure overfitting — and on its first run this corpus found **8
divergences the 771 did not contain**. All eight are fixed, and it now asserts
exactness like the others.

The last two were the subtlest bugs in the port, and neither was reachable from
the development corpora:

* **spaCy runs `_apply_special_cases` AFTER affix splitting**
  (`Tokenizer.__call__`: `_tokenize_affixes`, then `_apply_special_cases`). It
  re-joins a run of tokens whose text spells a special case. Peeling takes
  `’nuff!”` apart correctly at every step — prefix `’`, suffix `”`, suffix `!`
  — and only that later pass reunites the quote with the word. spaCy needs it
  for exactly the same reason, and without it `’nuff` arrived as two tokens.

* **A tokenizer rule's NORM must travel with the RULE.** `rule["you’d"]` gives
  its second piece the norm `'d`, while `rule["you’d’ve"]` gives the same orth
  `would`. A table keyed on the orth alone has to pick one, and picking wrong
  made the tagger call that piece `MD` where spaCy says `''` — which, since
  `''` is a PUNCT_TAG, decides whether the word stays joined. The tokenizer
  knows which rule fired, so it now carries each token's norm and the tables
  are only a fallback.

That second one is the general lesson of this port: misaki's output depends on
spaCy in more places than its source suggests, and the cheap approximation of
any of them is wrong on text you did not test against.
