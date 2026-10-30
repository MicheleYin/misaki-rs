"""Freeze misaki 0.9.4's output as golden fixtures for the Rust test suite.

misaki has NO test suite of its own -- not in the pip package, not in
hexgrad/misaki. So parity cannot be established by porting tests. Instead
misaki 0.9.4 itself is the oracle: we run it over a corpus and freeze the
output, which is a stronger bar than hand-written cases because it covers the
real input distribution.

What is captured per line, and why:
  * `ps`     -- the phoneme string, with misaki's own ɾ->T, ʔ->t post-process
                already applied (that is what G2P.__call__ returns).
  * `tokens` -- [[text, phonemes], ...]. NOT redundant: Kokoro's `pred_dur` has
                one entry per input token, so word->phoneme attribution is what
                the WORD TIMESTAMPS are computed from. A fixture that checked
                only `ps` could pass while the timestamps were wrong.

Run with the venv that has misaki 0.9.4:
    python tools/gen_fixtures.py <corpus.txt> <out.json> [a|b]
"""
import json
import sys

import misaki
from misaki import en


def main():
    corpus, out = sys.argv[1], sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "a"
    british = lang == "b"

    # fallback=None on purpose: espeak is a separate dependency with its own
    # version skew, so fixtures pin the LEXICON layer. OOV words are recorded
    # as having no phonemes, which is itself behaviour worth freezing.
    g = en.G2P(trf=False, british=british, fallback=None, unk="")

    cases = []
    for raw in open(corpus, encoding="utf-8"):
        text = raw.rstrip("\n")
        if not text.strip():
            continue
        ps, toks = g(text)
        # `tags` records spaCy's POS tag per final token. It is not part of the
        # parity contract -- it exists so the Rust side can be run with
        # misaki's own tags injected, which separates failures caused by our
        # tagger from failures caused by missing logic. Without that split you
        # cannot tell whether a wrong phoneme is a port bug or a tag bug.
        cases.append({
            "text": text,
            "ps": ps,
            "tokens": [[t.text, t.phonemes] for t in toks],
            "tags": [t.tag for t in toks],
        })

    doc = {
        "misaki_version": getattr(misaki, "__version__", "0.9.4"),
        "lang": lang,
        "british": british,
        "fallback": None,
        "unk": "",
        "n": len(cases),
        "cases": cases,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("wrote %d cases to %s" % (len(cases), out))


if __name__ == "__main__":
    main()
