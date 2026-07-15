"""Concept-level map: LSA term vectors -> t-SNE, with doctrinal-list overlays.

Terms live in the same 100-dim SVD space as the documents (V·S). The famous
numbered lists (aggregates, faculties, awakening factors, dependent origination,
brahmaviharas, ...) are tagged by stem so they can be colored as overlays.

Writes artifacts/concepts.json.
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

MIN_DF = 30
MAX_TERMS = 5000

# (group, [stem prefixes]) — first match wins
GROUPS = [
    ("aggregates & elements", ["khandh", "āyatan", "dhātu", "rūp", "vedan", "saññ", "saṅkhār", "viññāṇ"]),
    ("dependent origination", ["paṭiccasamupp", "avijj", "nāmarūp", "phass", "taṇh", "upādān", "jāti", "jarāmaraṇ", "bhav"]),
    ("path & wings", ["satipaṭṭhān", "sammappadhān", "iddhipād", "indriy", "bojjhaṅg", "magg", "sammādiṭṭh", "sammāsaṅkapp", "sammāvāc", "sammākammant", "sammāājīv", "sammāvāyām", "sammāsat", "sammāsamādh"]),
    ("meditation", ["jhān", "samādhi", "samath", "vipassan", "sati", "ānāpān", "kasiṇ", "nimitt"]),
    ("four truths", ["sacc", "dukkh", "samuday", "nirodh"]),
    ("divine abidings", ["mett", "karuṇ", "mudit", "upekkh"]),
    ("ethics & training", ["sīl", "dān", "pāṇātipāt", "musāvād", "surāmeray", "adinnādān", "kāmesumicchācār", "sikkhāpad"]),
    ("liberation", ["nibbān", "vimutt", "arahatt", "amat", "virāg"]),
]


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(TAGS.sub(" ", text.lower()))


def group_of(term: str) -> str | None:
    for name, stems in GROUPS:
        if any(term.startswith(s) for s in stems):
            return name
    return None


def main() -> None:
    segs = pl.read_parquet(A / "segments.parquet")
    docs = (
        segs.filter(pl.col("pali").is_not_null())
        .group_by("uid", maintain_order=True)
        .agg(pl.col("pali").str.join(" ").alias("text"))
        .with_columns(pl.col("text").str.len_chars().alias("chars"))
        .filter(pl.col("chars") >= 200)
    )
    vec = TfidfVectorizer(
        analyzer=tokenize, sublinear_tf=True, min_df=3, max_df=0.6,
        max_features=40000,
    )
    X = vec.fit_transform(docs["text"].to_list())
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd.fit(X)

    terms = np.array(vec.get_feature_names_out())
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = np.where(df >= MIN_DF)[0]
    keep = keep[np.argsort(df[keep])[::-1][:MAX_TERMS]]
    print(f"{len(keep)} terms (df >= {MIN_DF})")

    T = svd.components_.T[keep] * svd.singular_values_  # V·S rows
    Tn = normalize(T)
    xy = TSNE(n_components=2, perplexity=40, init="pca", random_state=42,
              max_iter=1000).fit_transform(Tn)

    out = []
    n_tagged = 0
    for i, idx in enumerate(keep):
        t = str(terms[idx])
        g = group_of(t)
        n_tagged += g is not None
        out.append(dict(
            t=t, df=int(df[idx]), g=g,
            x=round(float(xy[i, 0]), 3), y=round(float(xy[i, 1]), 3),
        ))
    (A / "concepts.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote concepts.json ({len(out)} terms, {n_tagged} tagged into lists)")


if __name__ == "__main__":
    main()
