"""Ingest the Pali commentaries from dharmanexus-pali into a substrate table.

Adds the Aṭṭhakathā (commentaries), Ṭīkā (sub-commentaries), and Anya (other,
incl. the Visuddhimagga) — the layer bilara-data doesn't segment-align. Pali is
CST (via DharmaNexus); English is DharmaMitra's machine translation.

  data/dharmanexus-pali/PA_files.json          — file metadata (+ collection)
  data/dharmanexus-pali/segments/<file>.json   — [{segmentnr, original, …}]
  data/dharmanexus-pali/translated/<file>-translated.tsv — + translated column

Writes artifacts/commentary_segments.parquet (uid, basket, nikaya, subpath,
collection, segment_id, seq, pali, english, translator).

NOTE: licensing for the underlying text/translations is unresolved (VRI CST is
© all-rights-reserved; the dharmanexus repo has no license) — this substrate is
for LOCAL analysis; do not host the verbatim text/translations until cleared.
"""

import ast
import csv
import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
DN = REPO / "data" / "dharmanexus-pali"
A = REPO / "artifacts"

# commentary/other collections (exclude the canonical ones we already have from bilara)
COMMENTARY = {"Atthakatha-Suttas": "aṭṭhakathā", "Atthakatha-Vinaya": "aṭṭhakathā",
              "Atthakatha-Abhidhamma": "aṭṭhakathā", "Tika-Suttas": "ṭīkā",
              "Tika-Vinaya": "ṭīkā", "Tika-Abhidhamma": "ṭīkā", "Anya": "anya"}
TRANSLATOR = "dharmamitra"


def parse_ids(cell: str) -> list[str]:
    cell = (cell or "").strip()
    try:
        v = ast.literal_eval(cell)
        return list(v) if isinstance(v, (list, tuple)) else [str(v)]
    except (ValueError, SyntaxError):
        return [cell] if cell else []


def english_map(fn: str) -> dict[str, str]:
    tf = DN / "translated" / f"{fn}-translated.tsv"
    if not tf.exists():
        return {}
    out = {}
    with open(tf, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            en = (row.get("translated") or "").strip()
            if not en:
                continue
            for sid in parse_ids(row.get("segmentnr", "")):
                out[sid] = en
    return out


def main() -> None:
    files = json.loads((DN / "PA_files.json").read_text())
    comm = [e for e in files if e["collection"] in COMMENTARY]
    print(f"{len(comm)} commentary files")

    rows = []
    for e in comm:
        fn, cat, coll = e["filename"], e["category"], e["collection"]
        segf = DN / "segments" / f"{fn}.json"
        if not segf.exists():
            continue
        segs = json.loads(segf.read_text())
        en = english_map(fn)
        basket = COMMENTARY[coll]
        for seq, s in enumerate(segs):
            sid = s["segmentnr"]
            pali = (s.get("original") or "").strip()
            if not pali:
                continue
            rows.append(dict(
                uid=fn, basket=basket, nikaya=cat, subpath=f"{coll}/{cat}",
                collection=coll, segment_id=sid, seq=seq, pali=pali,
                english=en.get(sid), translator=TRANSLATOR if en.get(sid) else None,
            ))

    df = pl.DataFrame(rows, schema={
        "uid": pl.Utf8, "basket": pl.Utf8, "nikaya": pl.Utf8, "subpath": pl.Utf8,
        "collection": pl.Utf8, "segment_id": pl.Utf8, "seq": pl.Int64,
        "pali": pl.Utf8, "english": pl.Utf8, "translator": pl.Utf8,
    })
    A.mkdir(exist_ok=True)
    df.write_parquet(A / "commentary_segments.parquet")

    docs = df["uid"].n_unique()
    trans = df.filter(pl.col("english").is_not_null()).height
    print(f"docs: {docs}  segments: {len(df)}  with English: {trans} ({trans/len(df)*100:.0f}%)")
    print(df.group_by("basket").agg(pl.col("uid").n_unique().alias("docs"),
                                    pl.len().alias("segs")).sort("basket"))


if __name__ == "__main__":
    main()
