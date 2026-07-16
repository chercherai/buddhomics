"""PREVIEW: re-layout canon + commentary together (does commentary cluster away?).

Same pipeline as build_features.py but over segments.parquet (canon) PLUS
commentary_segments.parquet (Aṭṭhakathā/Ṭīkā/Anya). Writes a SEPARATE file
(artifacts/map_combined_preview.json) — the deployed map.json is untouched.
Local preview only (licensing on the commentary text is unresolved).
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
TOKEN = re.compile(r"[a-zāīūṁṃṅñṭḍṇḷ’']+")
TAGS = re.compile(r"<[^>]+>")


def tokenize(text):
    return TOKEN.findall(TAGS.sub(" ", text.lower()))


def docframe(path, genre_from_basket):
    df = (pl.read_parquet(path)
          .filter(pl.col("pali").is_not_null())
          .group_by("uid", maintain_order=True)
          .agg(pl.first("basket"), pl.first("nikaya"), pl.first("subpath"),
               pl.col("pali").str.join(" ").alias("text"), pl.len().alias("n"))
          .with_columns(chars=pl.col("text").str.len_chars())
          .filter(pl.col("chars") >= 200))
    return df


def main():
    canon = docframe(A / "segments.parquet", None)
    comm = docframe(A / "commentary_segments.parquet", None)
    canon = canon.with_columns(kind=pl.lit("canon"))
    comm = comm.with_columns(kind=pl.lit("commentary"))
    docs = pl.concat([canon, comm])
    print(f"{len(canon)} canon + {len(comm)} commentary = {len(docs)} docs")

    vec = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True, min_df=3,
                          max_df=0.6, max_features=40000)
    X = vec.fit_transform(docs["text"].to_list())
    print(f"tfidf {X.shape}")
    svd = TruncatedSVD(n_components=100, random_state=42)
    Z = svd.fit_transform(X)
    print(f"svd explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    txy = TSNE(n_components=2, perplexity=30, init="pca", random_state=42,
               max_iter=1000).fit_transform(normalize(Z))

    pts = []
    for i, r in enumerate(docs.iter_rows(named=True)):
        pts.append(dict(uid=r["uid"], basket=r["basket"], kind=r["kind"],
                        n=r["n"], tx=round(float(txy[i, 0]), 2),
                        ty=round(float(txy[i, 1]), 2)))
    (A / "map_combined_preview.json").write_text(json.dumps(pts, ensure_ascii=False))
    from collections import Counter
    print("wrote map_combined_preview.json;", dict(Counter(p["kind"] for p in pts)))


if __name__ == "__main__":
    main()
