"""Shared helpers for the Fable 5 translation pipeline (probe + batch)."""

import json
import os
import re
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
MODEL = "claude-fable-5"
SEG_CAP = 100          # max segments per request (bounds output, avoids truncation)

# batched pricing per token
PRICE_IN = 2.50 / 1e6
PRICE_OUT = 12.50 / 1e6

SYSTEM = """You are translating Pali Buddhist canonical texts into English.

You receive numbered segments from one continuous passage. Translate each segment
into clear, accurate English in the register of a modern scholarly translation
(Bhikkhu Sujato / Bhikkhu Bodhi). Preserve doctrinal terminology precisely and use
established renderings where they exist.

Rules:
- Translate segment by segment: each segment's English must correspond to that
  segment's Pali, never merged or redistributed across segments.
- Preserve any <b>…</b> markup around lemmas in place.
- Keep the elision marker "…pe…" verbatim where it appears.
- Keep verse compact and natural; render a fragment (verse quarter, clause) so the
  sequence reads continuously.
- Proper names stay in Pali (transliterated), not translated."""


def load_env() -> None:
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def schema(seg_ids: list[str] | None = None) -> dict:
    # NB: generic (no per-request enum) so the grammar compiles once and caches;
    # a unique enum per request trips the 20/min grammar-compilation limit.
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "en": {"type": "string"},
                        },
                        "required": ["id", "en"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        },
    }


def untranslated_docs(subpaths: list[str] | None = None) -> pl.DataFrame:
    docs = pl.read_parquet(A / "documents.parquet").filter(pl.col("translated_frac") < 0.05)
    if subpaths:
        docs = docs.filter(pl.col("subpath").is_in(subpaths))
    return docs


def doc_chunks(uid: str, segs: pl.DataFrame) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return [(custom_id, [(segment_id, pali), ...]), ...] for one uid."""
    rows = (
        segs.filter((pl.col("uid") == uid) & pl.col("pali").is_not_null())
        .filter(pl.col("pali").str.strip_chars() != "")
        .sort("seq")
        .select("segment_id", "pali")
        .rows()
    )
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", uid)   # batch custom_id charset
    out = []
    for i in range(0, len(rows), SEG_CAP):
        chunk = rows[i : i + SEG_CAP]
        out.append((f"{safe}_{i // SEG_CAP}", chunk))
    return out


def build_prompt(label: str, chunk: list[tuple[str, str]]) -> str:
    body = "\n".join(f"{sid}\t{pali}" for sid, pali in chunk)
    return (f"Passage: {label}.\nTranslate all {len(chunk)} segments below "
            f"(id<TAB>Pali per line).\n\n{body}")
