"""Term-document incidence bitmatrix for the dual concepts/documents explorer.

Rows = terms in concepts.json order; columns = documents in map.json order.
Row-major, 8 docs per byte, doc j -> byte j>>3, bit j&7.

Writes artifacts/concept_bits.bin
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

TOKEN = re.compile(r"[a-zāīūṁṃṅñṭḍṇḷ’']+")
TAGS = re.compile(r"<[^>]+>")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(TAGS.sub(" ", text.lower()))


def main() -> None:
    pts = json.loads((A / "map.json").read_text())
    uids = [p["uid"] for p in pts]
    terms = [c["t"] for c in json.loads((A / "concepts.json").read_text())]
    from pipeline_input import read_segments
    segs = read_segments()
    texts = dict(
        segs.filter(pl.col("pali").is_not_null())
        .group_by("uid")
        .agg(pl.col("pali").str.join(" ").alias("text"))
        .iter_rows()
    )
    # commentary is included so it links into the concept map — this is a
    # term-incidence index (which concept words a text uses), not the running
    # text, which stays only in the obfuscated reader JSON
    corpus = [texts.get(u, "") for u in uids]

    vec = CountVectorizer(analyzer=tokenize, binary=True, vocabulary=terms)
    X = vec.fit_transform(corpus)          # docs x terms, binary
    M = np.asarray(X.todense(), dtype=np.uint8).T   # terms x docs
    packed = np.packbits(M, axis=1, bitorder="little")
    (A / "concept_bits.bin").write_bytes(packed.tobytes())
    print(f"{len(terms)} terms x {len(uids)} docs -> "
          f"{packed.shape[1]} bytes/row, {packed.nbytes/1e6:.2f} MB total, "
          f"density {M.mean():.3f}")


if __name__ == "__main__":
    main()
