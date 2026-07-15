"""Lemma-based Pali full-text search index (via DPD form->lemma).

Every text's Pali tokens are resolved to DPD lemmas (surface form kept when
DPD has no entry); the doc set per lemma is inverted and sharded by the lemma's
first two ASCII-folded chars into site/plem/<key>.json = {lemma: [docidx,...]}.

Client does prefix search: a lemma matching a query prefix lives in the query's
own shard, so one lazy shard fetch answers a Pali full-text search — collapsing
inflected forms (suññataṁ / suññato / suñño ...) under their lemma.
"""

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SITE = REPO / "site"
A = REPO / "artifacts"

TOK = re.compile(r"[a-zāīūṁṃṅñṭḍṇḷ]+")
HOM = re.compile(r"\s+\d+(\.\d+)?$")
FOLD = str.maketrans({"ā": "a", "ī": "i", "ū": "u", "ṁ": "m", "ṃ": "m", "ṅ": "n",
                      "ñ": "n", "ṭ": "t", "ḍ": "d", "ṇ": "n", "ḷ": "l", "ṛ": "r",
                      "ṝ": "r", "ḥ": "h", "’": "", "'": ""})
MIN_DF = 2


def shardkey(w: str) -> str:
    f = re.sub(r"[^a-z]", "", w.lower().translate(FOLD))
    return (f[:2] or "_").ljust(2, "_")


def lemmas_of(defs: list[str]) -> set[str]:
    out = set()
    for line in defs:
        line = line.strip()
        if ":" in line:
            head = HOM.sub("", line.split(":", 1)[0].strip()).strip().lower()
            if head and " " not in head:
                out.add(head)
        elif " + " in line:
            for p in line.split(" + "):
                p = p.strip().lower()
                if p:
                    out.add(p)
    return out


def main() -> None:
    dpd = json.loads((DATA / "pli2en_dpd.json").read_text())
    form2lem = {}
    for e in dpd:
        L = lemmas_of(e["definition"])
        if L:
            form2lem[e["entry"].lower()] = L

    segs = pl.read_parquet(A / "segments.parquet")
    uids = [p["uid"] for p in json.loads((A / "map.json").read_text())]
    upos = {u: i for i, u in enumerate(uids)}
    doctext = dict(
        segs.filter(pl.col("pali").is_not_null() & pl.col("uid").is_in(uids))
        .group_by("uid").agg(pl.col("pali").str.join(" ")).iter_rows()
    )

    lem2docs: dict[str, set[int]] = defaultdict(set)
    for uid in uids:
        i = upos[uid]
        for w in set(TOK.findall(doctext.get(uid, "").lower())):
            for lem in form2lem.get(w, {w}):
                lem2docs[lem].add(i)

    shards: dict[str, dict] = defaultdict(dict)
    kept = 0
    for lem, docs in lem2docs.items():
        if len(docs) < MIN_DF:
            continue
        shards[shardkey(lem)][lem] = sorted(docs)
        kept += 1

    out = SITE / "plem"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for key, entries in shards.items():
        (out / f"{key}.json").write_text(json.dumps(entries, ensure_ascii=False))
    total = sum((out / f"{k}.json").stat().st_size for k in shards) / 1e6
    big = max(shards.values(), key=lambda d: len(json.dumps(d)))
    print(f"{kept:,} lemmas (df>={MIN_DF}) -> {len(shards)} shards, {total:.1f} MB")
    print(f"largest shard: {max((out/f'{k}.json').stat().st_size for k in shards)/1e3:.0f} KB")


if __name__ == "__main__":
    main()
