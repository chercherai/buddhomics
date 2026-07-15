"""Validate proposed taxonomy stems against the corpus before adding them.

Reads a JSON file of proposals (as emitted by the curation subagents) and prints,
per term, the document coverage plus any dead stems (0 hits) — so zero/near-zero
entries can be dropped before editing TAXONOMY. Matching mirrors build_taxonomy.

Usage: validate_stems.py <proposals.json> [min_docs]
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from build_taxonomy import A, tokenize

MIN_DOCS = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def load_corpus():
    uids = [p["uid"] for p in json.loads((A / "map.json").read_text())]
    segs = pl.read_parquet(A / "segments.parquet")
    texts = dict(segs.filter(pl.col("pali").is_not_null()).group_by("uid")
                 .agg(pl.col("pali").str.join(" ").alias("t")).iter_rows())
    corpus = [texts.get(u, "") for u in uids]
    vec = CountVectorizer(analyzer=tokenize, binary=True, min_df=1)
    X = vec.fit_transform(corpus).tocsc()
    vocab = vec.get_feature_names_out()
    flat = [" ".join(tokenize(t)) for t in corpus]
    return X, vocab, flat


def stem_hits(s, X, vocab, flat):
    if s.startswith("~"):
        phrase = " ".join(tokenize(s[1:]))
        return sum(1 for t in flat if phrase in t)
    if s.startswith("="):
        t = s[1:]
        i = np.searchsorted(vocab, t)
        if i < len(vocab) and vocab[i] == t:
            return int(np.asarray(X[:, [i]].sum(axis=1)).ravel().nonzero()[0].size)
        return 0
    lo = np.searchsorted(vocab, s)
    hi = np.searchsorted(vocab, s + "￿")
    if hi <= lo:
        return 0
    return int((np.asarray(X[:, range(lo, hi)].sum(axis=1)).ravel() > 0).sum())


def term_docs(stems, X, vocab, flat):
    cols, phrase = [], set()
    for s in stems:
        if s.startswith("~"):
            p = " ".join(tokenize(s[1:]))
            phrase.update(i for i, t in enumerate(flat) if p in t)
        elif s.startswith("="):
            t = s[1:]
            i = np.searchsorted(vocab, t)
            if i < len(vocab) and vocab[i] == t:
                cols.append(i)
        else:
            lo = np.searchsorted(vocab, s)
            hi = np.searchsorted(vocab, s + "￿")
            cols.extend(range(lo, hi))
    hits = set(phrase)
    if cols:
        m = np.asarray(X[:, cols].sum(axis=1)).ravel() > 0
        hits |= set(np.where(m)[0].tolist())
    return len(hits)


def main():
    proposals = json.loads(Path(sys.argv[1]).read_text())
    X, vocab, flat = load_corpus()
    for cat, subs in proposals.items():
        print(f"\n=== {cat} ===")
        for sub in subs:
            print(f"  [{sub['h']}]")
            for term in sub["terms"]:
                n = term_docs(term["stems"], X, vocab, flat)
                dead = [s for s in term["stems"] if stem_hits(s, X, vocab, flat) == 0]
                flag = "" if n >= MIN_DOCS else "  <-- LOW"
                deadstr = f"  dead:{dead}" if dead else ""
                print(f"    {n:>5}  {term['t']}{flag}{deadstr}")


if __name__ == "__main__":
    main()
