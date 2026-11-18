"""Freeze misaki 0.9.4 recording-fallback output as a Rust golden.

Parity fixtures use fallback=None, so they cannot catch
"fallback stole an already-resolved multi-piece word" bugs. This generator
pins the fallback *policy* with a sentinel that returns "FALLBACK" and records
every word it is asked — the same contract as tests/fallback_invariant.rs.

Run with the venv that has misaki 0.9.4:
    .venv/bin/python tools/gen_fallback_fixtures.py \\
        tests/fixtures/golden_adversarial_us.json \\
        tests/fixtures/golden_fallback_adversarial_us.json a

The first argument may be a corpus `.txt` (one text per line) or a golden
`.json` whose `cases[].text` values are reused.
"""
import json
import sys

import misaki
from misaki import en


class RecordingFallback:
    """Mirrors the Rust RecordingFallback used in tests/fallback_invariant.rs."""

    def __init__(self):
        self.asked = []

    def __call__(self, token):
        self.asked.append(token.text)
        return "FALLBACK", 2


def main():
    corpus, out = sys.argv[1], sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "a"
    british = lang == "b"

    if corpus.endswith(".json"):
        texts = [c["text"] for c in json.load(open(corpus, encoding="utf-8"))["cases"]]
    else:
        texts = []
        for raw in open(corpus, encoding="utf-8"):
            line = raw.rstrip("\n")
            if line.strip():
                texts.append(line)

    cases = []
    for text in texts:
        fb = RecordingFallback()
        g = en.G2P(trf=False, british=british, fallback=fb, unk="")
        ps, toks = g(text)
        cases.append({
            "text": text,
            "asked": list(fb.asked),
            "ps": ps,
            "tokens": [[t.text, t.phonemes] for t in toks],
        })

    doc = {
        "misaki_version": getattr(misaki, "__version__", "0.9.4"),
        "lang": lang,
        "british": british,
        "fallback": "recording:FALLBACK",
        "oracle": "misaki",
        "unk": "",
        "n": len(cases),
        "cases": cases,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    asked_n = sum(1 for c in cases if c["asked"])
    print(
        "wrote %d cases to %s (%d ask, %d skip)"
        % (len(cases), out, asked_n, len(cases) - asked_n)
    )


if __name__ == "__main__":
    main()
