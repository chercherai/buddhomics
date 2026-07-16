"""Ingest the Pali commentaries from dharmanexus-pali, split into sections.

Each commentary volume is split at its numbered sutta/section headings
("1. Brahmajālasuttavaṇṇanā", chapter niddesas, …) so it maps section-by-section
(≈5k sub-documents) rather than as one coarse volume. Tiny sections merge into
the previous; oversized ones are chunked.

Pali is CST (via DharmaNexus); English is DharmaMitra's machine translation.
Writes artifacts/commentary_segments.parquet + artifacts/commentary_titles.json
(uid -> section heading, for map titles).

NOTE: licensing for the text/translations is unresolved (VRI CST is © all-rights-
reserved; the repo is unlicensed) — served text is obfuscated, not open.
"""

import ast
import csv
import json
import re
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
DN = REPO / "data" / "dharmanexus-pali"
A = REPO / "artifacts"

COMMENTARY = {"Atthakatha-Suttas": "aṭṭhakathā", "Atthakatha-Vinaya": "aṭṭhakathā",
              "Atthakatha-Abhidhamma": "aṭṭhakathā", "Tika-Suttas": "ṭīkā",
              "Tika-Vinaya": "ṭīkā", "Tika-Abhidhamma": "ṭīkā", "Anya": "anya"}
TRANSLATOR = "dharmamitra"
# a real section heading: a numbered "N. …vaṇṇanā" (sutta commentary) or a
# niddesa/chapter heading (Visuddhimagga & treatises) — NOT numbered body items
HEAD = re.compile(r"^(\d+\.\s.*(vaṇṇanā|kathāvaṇṇanā)"
                  r"|.{0,40}(niddeso|niddesā|kaṇḍaṃ|paricchedo|bhāṇavāro))\s*$")
MIN, MAX = 40, 400                   # merge sections below MIN, chunk above MAX


def is_head(t: str) -> bool:
    t = (t or "").strip()
    return len(t) <= 60 and bool(HEAD.match(t))


def parse_ids(cell: str) -> list[str]:
    try:
        v = ast.literal_eval((cell or "").strip())
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
            if en:
                for sid in parse_ids(row.get("segmentnr", "")):
                    out[sid] = en
    return out


def sections(seg: list) -> list[tuple[int, int, str]]:
    """(start, end, heading) sections split at numbered headings, merged/chunked."""
    heads = {i: s["original"].strip() for i, s in enumerate(seg)
             if is_head(s.get("original", ""))}
    starts = sorted({0, *heads})
    bnds = starts + [len(seg)]
    raw = [(bnds[i], bnds[i + 1], heads.get(bnds[i], "")) for i in range(len(bnds) - 1)]
    merged = []
    for a, b, h in raw:
        if merged and (b - a) < MIN:
            merged[-1] = (merged[-1][0], b, merged[-1][2])
        else:
            merged.append((a, b, h))
    out = []
    for a, b, h in merged:
        if b - a > MAX:
            for c in range(a, b, MAX):
                out.append((c, min(c + MAX, b), h if c == a else ""))
        else:
            out.append((a, b, h))
    return out


def main() -> None:
    files = json.loads((DN / "PA_files.json").read_text())
    comm = [e for e in files if e["collection"] in COMMENTARY]
    rows, titles = [], {}
    for e in comm:
        fn, cat, coll = e["filename"], e["category"], e["collection"]
        segf = DN / "segments" / f"{fn}.json"
        if not segf.exists():
            continue
        segs = json.loads(segf.read_text())
        en = english_map(fn)
        basket = COMMENTARY[coll]
        for k, (a, b, head) in enumerate(sections(segs)):
            uid = f"{fn}-s{k:03d}"
            titles[uid] = re.sub(r"^\d+\.\s*", "", head) or e.get("displayName", fn)
            for seq, s in enumerate(segs[a:b]):
                pali = (s.get("original") or "").strip()
                if not pali:
                    continue
                sid = s["segmentnr"]
                rows.append(dict(
                    uid=uid, basket=basket, nikaya=cat,
                    subpath=f"{coll}/{cat}/{fn}", collection=coll,
                    segment_id=sid, seq=seq, pali=pali,
                    english=en.get(sid),
                    translator=TRANSLATOR if en.get(sid) else None,
                ))

    df = pl.DataFrame(rows, schema={
        "uid": pl.Utf8, "basket": pl.Utf8, "nikaya": pl.Utf8, "subpath": pl.Utf8,
        "collection": pl.Utf8, "segment_id": pl.Utf8, "seq": pl.Int64,
        "pali": pl.Utf8, "english": pl.Utf8, "translator": pl.Utf8,
    })
    A.mkdir(exist_ok=True)
    df.write_parquet(A / "commentary_segments.parquet")
    (A / "commentary_titles.json").write_text(json.dumps(titles, ensure_ascii=False))
    print(f"docs: {df['uid'].n_unique()}  segments: {len(df)}  "
          f"(from {len(comm)} volumes)")
    print(df.group_by("basket").agg(pl.col("uid").n_unique().alias("docs"),
                                    pl.len().alias("segs")).sort("basket"))


if __name__ == "__main__":
    main()
