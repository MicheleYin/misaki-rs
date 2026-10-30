"""Extract spaCy en_core_web_sm's tagger into a flat f32 blob for the Rust port.

misaki tags with spaCy, and its pronunciation is POS-conditional (790 tag-keyed
gold entries, plus get_special_case branches where `A` as DT is ɐ and as NNP is
ˈA). Exact parity therefore needs exactly spaCy's tags, and the embedded
perceptron tagger agrees only ~89%.

The model is small enough to carry: MultiHashEmbed + 4x residual
MaxoutWindowEncoder + softmax over 50 tags, 1,565,664 parameters.

Layout of tagger.bin -- all little-endian f32, concatenated in this order:
    E[0..6]      (5000,96) (1000,96) (2500,96) (2500,96) (50,96) (50,96)
    maxout0.W    (96,3,576)      maxout0.b   (96,3)
    layernorm0.G (96,)           layernorm0.b (96,)
    for i in 1..5:
        maxout_i.W   (96,3,288)  maxout_i.b  (96,3)
        layernorm_i.G (96,)      layernorm_i.b (96,)
    softmax.W    (50,96)         softmax.b   (50,)

tagger_labels.txt holds the 50 labels, one per line, in output order.
"""
import sys
import numpy as np
import spacy


def main():
    out_bin = sys.argv[1] if len(sys.argv) > 1 else "data/tagger.bin"
    out_lab = sys.argv[2] if len(sys.argv) > 2 else "data/tagger_labels.txt"

    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    t2v = nlp.get_pipe("tok2vec").model
    tg = nlp.get_pipe("tagger").model

    hes = [n for n in t2v.walk() if n.name == "hashembed"]
    mx = [n for n in t2v.walk() if n.name == "maxout"]
    ln = [n for n in t2v.walk() if n.name == "layernorm"]
    sm = [n for n in tg.walk() if n.name == "softmax"][0]

    # Order is load-bearing: the Rust side reads this blob positionally.
    assert [n.attrs["seed"] for n in hes] == [8, 9, 10, 11, 12, 13], "hashembed order changed"
    assert [n.get_dim("nV") for n in hes] == [5000, 1000, 2500, 2500, 50, 50], "nV changed"
    assert len(mx) == 5 and len(ln) == 5, "encoder depth changed"

    parts = []
    for n in hes:
        parts.append(np.asarray(n.get_param("E"), dtype="<f4").ravel())
    for i in range(5):
        parts.append(np.asarray(mx[i].get_param("W"), dtype="<f4").ravel())
        parts.append(np.asarray(mx[i].get_param("b"), dtype="<f4").ravel())
        parts.append(np.asarray(ln[i].get_param("G"), dtype="<f4").ravel())
        parts.append(np.asarray(ln[i].get_param("b"), dtype="<f4").ravel())
    parts.append(np.asarray(sm.get_param("W"), dtype="<f4").ravel())
    parts.append(np.asarray(sm.get_param("b"), dtype="<f4").ravel())

    blob = np.concatenate(parts)
    blob.tofile(out_bin)
    labels = list(nlp.get_pipe("tagger").labels)
    with open(out_lab, "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")

    print("wrote %s: %d f32 (%.2f MB)" % (out_bin, blob.size, blob.nbytes / 1e6))
    print("wrote %s: %d labels" % (out_lab, len(labels)))


if __name__ == "__main__":
    main()
