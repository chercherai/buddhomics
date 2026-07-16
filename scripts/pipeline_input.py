"""Pipeline reads the combined canon+commentary substrate when present."""
from pathlib import Path
import polars as pl

_A = Path(__file__).resolve().parent.parent / "artifacts"


def segments_path() -> Path:
    c = _A / "combined_segments.parquet"
    return c if c.exists() else _A / "segments.parquet"


def read_segments() -> pl.DataFrame:
    return pl.read_parquet(segments_path())
