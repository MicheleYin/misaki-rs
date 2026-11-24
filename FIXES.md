# Opt-in phonemizer fixes

Default misaki-rs builds match **stock misaki 0.9.4** byte-for-byte (see
[PARITY.md](PARITY.md)). Cargo feature `fixes` (off by default) enables a pack
of corrections for known 0.9.4 mistakes—including unmerged upstream PRs and a
few local rules.

Corrections were contributed by [@homeygravity](https://github.com/homeygravity)
in #30–#33; this tree keeps those commits and organizes them under `src/fixes/`
for long-term maintenance.

## Dual gates

| Gate | Command | Asserts |
|------|---------|---------|
| Stock mirror | `cargo test --no-default-features` | Frozen 0.9.4 fixtures + stock unit tests |
| Opt-in pack | `cargo test --no-default-features --features fixes` | `src/fixes` unit tests only (parity suites are `cfg`-off) |

Do not regenerate the default fixtures against a patched Python fork.

## Lifecycle

1. **Propose** — open a PR against `feature/phonemizer-fixes` (or `main` once this pack has landed).
2. **Opt-in** — add `src/fixes/<name>.rs`, re-export/hook from `src/fixes/mod.rs`, a row here and in `src/fixes/catalog.rs`, plus tests.
3. **Promote** — only when PyPI ships a new misaki release that includes the change: retarget the parity oracle, move the behavior into the stock path, set status to `promoted`.
4. **Remove** — if superseded, delete the module and mark `removed` in the catalog history.

Until a new released misaki becomes the oracle, corrections stay behind `fixes`.

## Registry

| id | behavior | upstream | status |
|----|----------|----------|--------|
| `ampersand_stop` | peel `.` after `&`+capital (`AT&T.`) | none (local; misaki-rs #30) | opt-in |
| `curly_quotes` | straighten `’` / `‘` before tagging | [hexgrad#101](https://github.com/hexgrad/misaki/pull/101) (misaki-rs #31) | opt-in |
| `dives_abbrevs` | `dives` plural + selected titles | [hexgrad#110](https://github.com/hexgrad/misaki/pull/110), [hexgrad#86](https://github.com/hexgrad/misaki/pull/86) (misaki-rs #32) | opt-in |
| `perfect_question` | Have/Has/Had…? VB→VBN | none (local; misaki-rs #33) | opt-in |

## Adding the next fix

1. Create `src/fixes/<name>.rs` with the correction logic.
2. Re-export from `src/fixes/mod.rs` and call from the core pipeline hook point.
3. Add a `FixEntry` to `src/fixes/catalog.rs` and a row to this table.
4. Add regression rows under `src/fixes/mod.rs` `tests`.
5. Confirm stock gate still passes with `fixes` off.
