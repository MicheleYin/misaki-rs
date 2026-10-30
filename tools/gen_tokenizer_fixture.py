"""Golden tokenization from spaCy, for the Rust tokenizer port to match."""
import json
import sys

import spacy


def main():
    corpus, out = sys.argv[1], sys.argv[2]
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    cases = []
    for raw in open(corpus, encoding="utf-8"):
        text = raw.rstrip("\n")
        if not text.strip():
            continue
        doc = nlp.make_doc(text)
        cases.append({"text": text,
                      "tokens": [[t.text, t.whitespace_] for t in doc]})
    json.dump({"n": len(cases), "cases": cases}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote %d cases to %s" % (len(cases), out))


if __name__ == "__main__":
    main()
