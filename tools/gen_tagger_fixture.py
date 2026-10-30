"""Golden tags from spaCy, on spaCy's OWN tokens.

Isolates the tagger from the tokenizer: if the Rust tagger reproduces these,
any remaining tag disagreement in the pipeline is a tokenization difference,
not a model difference.
"""
import json
import sys

import spacy


def main():
    corpus, out = sys.argv[1], sys.argv[2]
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    t2v = nlp.get_pipe("tok2vec").model
    ef = [n for n in t2v.walk() if n.name == "extract_features"][0]
    cases = []
    for raw in open(corpus, encoding="utf-8"):
        text = raw.rstrip("\n")
        if not text.strip():
            continue
        doc = nlp(text)
        cases.append({
            "text": text,
            # (token text, norm, does whitespace follow) -- exactly the tagger's inputs
            "tokens": [[t.text, t.norm_, t.whitespace_ != ""] for t in doc],
            "tags": [t.tag_ for t in doc],
            # spaCy's own feature matrix, so a Rust-side mismatch can be
            # attributed to the feature computation rather than the network.
            "feats": [[int(v) for v in row] for row in ef.predict([nlp.make_doc(text)])[0]],
        })
    json.dump({"n": len(cases), "cases": cases}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote %d cases to %s" % (len(cases), out))


if __name__ == "__main__":
    main()
