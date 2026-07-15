"""Flatten bilara-data into one aligned segment table.

Walks root/pli/ms/** for Pali segments, joins English translations by
segment id (preferring sujato, falling back to any translator), and writes:

  artifacts/segments.parquet
  artifacts/segments.sqlite   (table: segments, documents)

Row: uid, basket, nikaya, subpath, segment_id, seq, pali, english, translator
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

PREFERRED_TRANSLATORS = ["sujato", "brahmali"]


def uid_from_filename(path: Path) -> str:
    # "mn1_root-pli-ms.json" -> "mn1"; "mn1_translation-en-sujato.json" -> "mn1"
    return path.name.split("_")[0]


def load_json(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_translations() -> dict[str, list[tuple[str, Path]]]:
    """uid -> [(translator, path), ...] for all English translations."""
    out: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    en_root = BILARA / "translation" / "en"
    for path in en_root.rglob("*.json"):
        if "_translation-en-" not in path.name:
            continue
        translator = path.stem.split("_translation-en-")[-1]
        out[uid_from_filename(path)].append((translator, path))
    return out


def pick_translation(candidates: list[tuple[str, Path]]) -> tuple[str, Path] | None:
    if not candidates:
        return None
    for pref in PREFERRED_TRANSLATORS:
        for translator, path in candidates:
            if translator == pref:
                return translator, path
    return sorted(candidates)[0]


def main() -> None:
    if not BILARA.is_dir():
        sys.exit(f"bilara-data not found at {BILARA}; clone it first")

    translations = index_translations()
    root_base = BILARA / "root" / "pli" / "ms"

    rows: list[dict] = []
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
        chosen = pick_translation(translations.get(uid, []))
        english: dict[str, str] = {}
        translator = None
        if chosen:
            translator, tpath = chosen
            english = load_json(tpath)

        seen = set()
        seq = 0
        for seg_id, pali_text in pali.items():
            rows.append(
                dict(
                    uid=uid, basket=basket, nikaya=nikaya, subpath=subpath,
                    segment_id=seg_id, seq=seq,
                    pali=pali_text, english=english.get(seg_id),
                    translator=translator,
                )
            )
            seen.add(seg_id)
            seq += 1
        # translation-only segments (rare) appended after root segments
        for seg_id, en_text in english.items():
            if seg_id not in seen:
                rows.append(
                    dict(
                        uid=uid, basket=basket, nikaya=nikaya, subpath=subpath,
                        segment_id=seg_id, seq=seq,
                        pali=None, english=en_text, translator=translator,
                    )
                )
                seq += 1
        n_docs += 1

    df = pl.DataFrame(
        rows,
        schema={
            "uid": pl.Utf8, "basket": pl.Utf8, "nikaya": pl.Utf8,
            "subpath": pl.Utf8, "segment_id": pl.Utf8, "seq": pl.Int64,
            "pali": pl.Utf8, "english": pl.Utf8, "translator": pl.Utf8,
        },
    )
    ARTIFACTS.mkdir(exist_ok=True)
    df.write_parquet(ARTIFACTS / "segments.parquet")

    docs = (
        df.group_by("uid", maintain_order=True)
        .agg(
            pl.first("basket"), pl.first("nikaya"), pl.first("subpath"),
            pl.first("translator"),
            pl.len().alias("n_segments"),
            pl.col("pali").str.len_chars().sum().alias("pali_chars"),
            pl.col("english").str.len_chars().sum().alias("english_chars"),
            (pl.col("english").is_not_null().sum() / pl.len()).alias("translated_frac"),
        )
    )
    docs.write_parquet(ARTIFACTS / "documents.parquet")

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
        CREATE TABLE documents (
            uid TEXT, basket TEXT, nikaya TEXT, subpath TEXT, translator TEXT,
            n_segments INTEGER, pali_chars INTEGER, english_chars INTEGER,
            translated_frac REAL
        );
        """
    )
    con.executemany(
        "INSERT INTO segments VALUES (?,?,?,?,?,?,?,?,?)",
        df.select(
            "uid", "basket", "nikaya", "subpath", "segment_id", "seq",
            "pali", "english", "translator",
        ).iter_rows(),
    )
    con.executemany(
        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
        docs.select(
            "uid", "basket", "nikaya", "subpath", "translator",
            "n_segments", "pali_chars", "english_chars", "translated_frac",
        ).iter_rows(),
    )
    con.executescript(
        """
        CREATE INDEX idx_segments_uid ON segments(uid);
        CREATE INDEX idx_segments_nikaya ON segments(nikaya);
        CREATE INDEX idx_documents_uid ON documents(uid);
        """
    )
    con.commit()
    con.close()

    print(f"documents: {n_docs}")
    print(f"segments:  {len(df)}")
    print(df.group_by("basket").len().sort("basket"))
    print(f"wrote {ARTIFACTS / 'segments.parquet'}, documents.parquet, segments.sqlite")


if __name__ == "__main__":
    main()
