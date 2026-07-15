"""Fold artifacts/translations_fable.parquet into the substrate.

Fills the english column for segments that had none, marking translator
'claude-fable-5', and rewrites segments.parquet + documents.parquet so
translated_frac and the reader reflect the machine translations.
Run scripts/build_texts.py afterwards to regenerate the reader JSON.
"""

from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"


def main() -> None:
    seg = pl.read_parquet(A / "segments.parquet")
    tr = pl.read_parquet(A / "translations_fable.parquet")
    before = seg.filter(pl.col("english").is_not_null()).height

    seg = seg.join(tr.rename({"english": "en_fable"}), on="segment_id", how="left")
    fill = pl.col("english").is_null() & pl.col("en_fable").is_not_null()
    seg = seg.with_columns(
        english=pl.when(fill).then(pl.col("en_fable")).otherwise(pl.col("english")),
        translator=pl.when(fill).then(pl.lit("claude-fable-5"))
                     .otherwise(pl.col("translator")),
    ).drop("en_fable")

    after = seg.filter(pl.col("english").is_not_null()).height
    seg.write_parquet(A / "segments.parquet")

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
    docs.write_parquet(A / "documents.parquet")
    print(f"filled {after - before} segments with Fable English "
          f"(now {after}/{seg.height} translated)")


if __name__ == "__main__":
    main()
