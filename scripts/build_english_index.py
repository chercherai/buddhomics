"""English-translation search index: vocabulary + doc-incidence bitmatrix.

Tokenizes the English translation of every mapped text; keeps content words
(20 <= df <= 2500, top 5000 by df). Same bit layout as concept_bits.bin:
rows = words in english_terms.json order, 8 docs per byte, little bit-order.

Writes artifacts/english_terms.json and artifacts/english_bits.bin.
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

WORD = re.compile(r"[a-z][a-z''-]*")


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


def main() -> None:
    pts = json.loads((A / "map.json").read_text())
    uids = [p["uid"] for p in pts]
    # commentary kept out of served indexes (licensing) — aligned all-zero columns
    comm = {p["uid"] for p in pts if p.get("kind") == "commentary"}
    from pipeline_input import read_segments
    segs = read_segments()
    texts = dict(
        segs.filter(pl.col("english").is_not_null())
        .group_by("uid")
        .agg(pl.col("english").str.join(" ").alias("text"))
        .iter_rows()
    )
    corpus = ["" if u in comm else texts.get(u, "") for u in uids]

    vec = CountVectorizer(analyzer=tokenize, binary=True, min_df=20, max_df=2500)
    X = vec.fit_transform(corpus)
    vocab = vec.get_feature_names_out()
    df = np.asarray(X.sum(axis=0)).ravel()
    keep = np.argsort(df)[::-1][:5000]
    keep = keep[np.argsort(vocab[keep])]          # alphabetical for readability
    words = vocab[keep].tolist()
    M = np.asarray(X[:, keep].todense(), dtype=np.uint8).T
    packed = np.packbits(M, axis=1, bitorder="little")
    (A / "english_bits.bin").write_bytes(packed.tobytes())
    (A / "english_terms.json").write_text(json.dumps(words, ensure_ascii=False))
    print(f"{len(words)} english words x {len(uids)} docs, "
          f"{packed.nbytes/1e6:.2f} MB bits, density {M.mean():.3f}")


if __name__ == "__main__":
    main()
