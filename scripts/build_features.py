"""Discourse-level features: Pali TF-IDF -> SVD -> UMAP 2D + top terms per document.

Reads artifacts/segments.parquet. Writes:
  artifacts/map.json       — points for the site scatter
  artifacts/doc_svd.npy    — 100-d document vectors (for the tree + later layers)
  artifacts/doc_index.json — row order of doc_svd.npy
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import umap

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

TOKEN = re.compile(r"[a-zāīūṁṃṅñṭḍṇḷ’']+")
TAGS = re.compile(r"<[^>]+>")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(TAGS.sub(" ", text.lower()))


def main() -> None:
    segs = pl.read_parquet(A / "segments.parquet")
    docs = (
        segs.filter(pl.col("pali").is_not_null())
        .group_by("uid", maintain_order=True)
        .agg(
            pl.first("basket"), pl.first("nikaya"), pl.first("subpath"),
            pl.col("pali").str.concat(" ").alias("text"),
            pl.len().alias("n_segments"),
        )
        .with_columns(pl.col("text").str.len_chars().alias("chars"))
        .filter(pl.col("chars") >= 200)
    )
    print(f"{len(docs)} documents after min-length filter")

    texts = docs["text"].to_list()
    vec = TfidfVectorizer(
        analyzer=tokenize, sublinear_tf=True, min_df=3, max_df=0.6,
        max_features=40000,
    )
    X = vec.fit_transform(texts)
    print(f"tfidf matrix {X.shape}")

    svd = TruncatedSVD(n_components=100, random_state=42)
    Z = svd.fit_transform(X)
    print(f"svd explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    xy = reducer.fit_transform(Z)

    # top tf-idf terms per doc
    terms = np.array(vec.get_feature_names_out())
    top_terms = []
    Xcsr = X.tocsr()
    for i in range(X.shape[0]):
        row = Xcsr[i]
        if row.nnz == 0:
            top_terms.append([])
            continue
        idx = row.indices[np.argsort(row.data)[::-1][:8]]
        top_terms.append(terms[idx].tolist())

    points = []
    for i, r in enumerate(docs.iter_rows(named=True)):
        points.append(
            dict(
                uid=r["uid"], basket=r["basket"], nikaya=r["nikaya"],
                subpath=r["subpath"], n=r["n_segments"],
                x=round(float(xy[i, 0]), 3), y=round(float(xy[i, 1]), 3),
                terms=top_terms[i],
            )
        )
    (A / "map.json").write_text(json.dumps(points, ensure_ascii=False))
    np.save(A / "doc_svd.npy", Z)
    (A / "doc_index.json").write_text(
        json.dumps([r["uid"] for r in docs.iter_rows(named=True)])
    )
    print(f"wrote map.json ({len(points)} points), doc_svd.npy, doc_index.json")


if __name__ == "__main__":
    main()
