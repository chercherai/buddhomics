"""Flatten bilara-data into aligned segment + translation tables.

Walks root/pli/ms/** for Pali segments and ALL of translation/en/** for every
published English translation (not just one per text), writing:

  artifacts/segments.parquet         backbone + best-available English per segment
  artifacts/translations_human.parquet  long: (segment_id, translator, english, kind)
  artifacts/segments.sqlite          (tables: segments, documents)
  artifacts/documents.parquet        per-doc meta + available human translators

'Best' per segment prefers human translators in HUMAN_PREF order; machine layers
(Fable, GPT) fold in later via merge_translations.py, which rebuilds
translations.parquet (all sources) and recomputes segments/documents.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
BILARA = REPO / "data" / "bilara-data"
ARTIFACTS = REPO / "artifacts"

# translator preference for the per-segment 'best' pick and reader default order
HUMAN_PREF = ["sujato", "brahmali", "kelly", "soma", "patton",
              "suddhaso", "anandajoti", "kovilo"]


def human_rank(tr: str) -> int:
    return HUMAN_PREF.index(tr) if tr in HUMAN_PREF else len(HUMAN_PREF)


def uid_from_filename(path: Path) -> str:
    # "mn1_root-pli-ms.json" -> "mn1"; "mn1_translation-en-sujato.json" -> "mn1"
    return path.name.split("_")[0]


def load_json(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_translations() -> dict[str, list[tuple[str, Path]]]:
    """uid -> [(translator, path), ...] for all English translations, best-first."""
    out: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    en_root = BILARA / "translation" / "en"
    for path in en_root.rglob("*.json"):
        if "_translation-en-" not in path.name:
            continue
        translator = path.stem.split("_translation-en-")[-1]
        out[uid_from_filename(path)].append((translator, path))
    for uid in out:
        out[uid].sort(key=lambda tp: (human_rank(tp[0]), tp[0]))
    return out


def main() -> None:
    if not BILARA.is_dir():
        sys.exit(f"bilara-data not found at {BILARA}; clone it first")

    translations = index_translations()
    root_base = BILARA / "root" / "pli" / "ms"

    seg_rows: list[dict] = []          # backbone + best english
    tr_rows: list[dict] = []           # long: every human translation
    doc_translators: dict[str, list[str]] = {}
    n_docs = 0

    for path in sorted(root_base.rglob("*_root-pli-ms.json")):
        rel = path.relative_to(root_base)
        parts = rel.parts  # e.g. ("sutta", "sn", "sn1", "sn1.1_root-pli-ms.json")
        basket = parts[0]
        if basket == "xplayground":
            continue
        nikaya = parts[1] if len(parts) > 2 else basket
        subpath = "/".join(parts[1:-1])
        uid = uid_from_filename(path)

        pali = load_json(path)
        cands = translations.get(uid, [])   # already best-first
        loaded = [(tr, load_json(p)) for tr, p in cands]
        doc_translators[uid] = [tr for tr, _ in cands]

        # long rows: every (segment, translator) with English text
        for tr, en in loaded:
            for seg_id, txt in en.items():
                if txt is not None and txt != "":
                    tr_rows.append(dict(segment_id=seg_id, translator=tr,
                                        english=txt, kind="human"))

        def best_en(seg_id: str):
            for tr, en in loaded:            # loaded is best-first
                if en.get(seg_id):
                    return en[seg_id], tr
            return None, None

        seen = set()
        seq = 0
        for seg_id, pali_text in pali.items():
            en, tr = best_en(seg_id)
            seg_rows.append(dict(
                uid=uid, basket=basket, nikaya=nikaya, subpath=subpath,
                segment_id=seg_id, seq=seq, pali=pali_text,
                english=en, translator=tr,
            ))
            seen.add(seg_id)
            seq += 1
        # translation-only segments (rare) after root segments
        extra = {}
        for _, en in loaded:
            for seg_id, txt in en.items():
                if seg_id not in seen and txt:
                    extra.setdefault(seg_id, None)
        for seg_id in extra:
            en, tr = best_en(seg_id)
            seg_rows.append(dict(
                uid=uid, basket=basket, nikaya=nikaya, subpath=subpath,
                segment_id=seg_id, seq=seq, pali=None,
                english=en, translator=tr,
            ))
            seq += 1
        n_docs += 1

    seg = pl.DataFrame(seg_rows, schema={
        "uid": pl.Utf8, "basket": pl.Utf8, "nikaya": pl.Utf8, "subpath": pl.Utf8,
        "segment_id": pl.Utf8, "seq": pl.Int64, "pali": pl.Utf8,
        "english": pl.Utf8, "translator": pl.Utf8,
    })
    ARTIFACTS.mkdir(exist_ok=True)
    seg.write_parquet(ARTIFACTS / "segments.parquet")

    tr = pl.DataFrame(tr_rows, schema={
        "segment_id": pl.Utf8, "translator": pl.Utf8,
        "english": pl.Utf8, "kind": pl.Utf8,
    }).unique(["segment_id", "translator"], keep="first")
    tr.write_parquet(ARTIFACTS / "translations_human.parquet")

    write_documents(seg, doc_translators)
    write_sqlite(seg)

    multi = sum(1 for v in doc_translators.values() if len(v) > 1)
    print(f"documents: {n_docs} ({multi} with >1 human translator)")
    print(f"segments:  {len(seg)}")
    print(f"human translations (long): {len(tr)} rows")
    print(seg.group_by("basket").len().sort("basket"))


def write_documents(seg: pl.DataFrame, doc_translators: dict[str, list[str]]) -> None:
    docs = (
        seg.group_by("uid", maintain_order=True)
        .agg(
            pl.first("basket"), pl.first("nikaya"), pl.first("subpath"),
            pl.first("translator"),
            pl.len().alias("n_segments"),
            pl.col("pali").str.len_chars().sum().alias("pali_chars"),
            pl.col("english").str.len_chars().sum().alias("english_chars"),
            (pl.col("english").is_not_null().sum() / pl.len()).alias("translated_frac"),
        )
    )
    trl = pl.DataFrame(
        {"uid": list(doc_translators), "translators": list(doc_translators.values())},
        schema={"uid": pl.Utf8, "translators": pl.List(pl.Utf8)},
    )
    docs = docs.join(trl, on="uid", how="left")
    docs.write_parquet(ARTIFACTS / "documents.parquet")


def write_sqlite(seg: pl.DataFrame) -> None:
    db_path = ARTIFACTS / "segments.sqlite"
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE segments (
            uid TEXT, basket TEXT, nikaya TEXT, subpath TEXT,
            segment_id TEXT, seq INTEGER,
            pali TEXT, english TEXT, translator TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO segments VALUES (?,?,?,?,?,?,?,?,?)",
        seg.select("uid", "basket", "nikaya", "subpath", "segment_id", "seq",
                   "pali", "english", "translator").iter_rows(),
    )
    con.executescript(
        "CREATE INDEX idx_segments_uid ON segments(uid);"
        "CREATE INDEX idx_segments_nikaya ON segments(nikaya);"
    )
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
