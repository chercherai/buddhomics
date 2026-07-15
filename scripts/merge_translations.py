"""Compose all translation sources into the substrate.

Combines human translations (translations_human.parquet, from build_substrate)
with machine layers (translations_fable.parquet, translations_gpt.parquet) into
a single long table artifacts/translations.parquet:

    segment_id, translator, english, kind   (kind: human | machine)

then recomputes the per-segment BEST English (human preferred, machine filling
gaps) back into segments.parquet, and rewrites documents.parquet with each
doc's available translators. Run build_texts.py afterwards for the reader JSON.
"""

from pathlib import Path

import polars as pl

from build_substrate import HUMAN_PREF

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

# machine parquet (segment_id, english) -> translator id, in preference order
MACHINE_SOURCES = [
    ("translations_fable.parquet", "claude-fable-5"),
    ("translations_gpt.parquet", "gpt-5.6-sol"),
]
MACHINE_PREF = [m for _, m in MACHINE_SOURCES]
ORDER = HUMAN_PREF + MACHINE_PREF


def main() -> None:
    seg = pl.read_parquet(A / "segments.parquet")
    human = pl.read_parquet(A / "translations_human.parquet")

    parts = [human]
    for fn, mid in MACHINE_SOURCES:
        p = A / fn
        if not p.exists():
            continue
        m = (pl.read_parquet(p)
             .select("segment_id", "english")
             .filter(pl.col("english").is_not_null()
                     & (pl.col("english").str.strip_chars() != ""))
             .with_columns(translator=pl.lit(mid), kind=pl.lit("machine")))
        parts.append(m.select("segment_id", "translator", "english", "kind"))

    trans = pl.concat(parts).unique(["segment_id", "translator"], keep="first")
    trans.write_parquet(A / "translations.parquet")

    rankdf = pl.DataFrame({"translator": ORDER, "r": list(range(len(ORDER)))},
                          schema={"translator": pl.Utf8, "r": pl.Int64})
    ranked = trans.join(rankdf, on="translator", how="left").with_columns(
        pl.col("r").fill_null(len(ORDER)))
    best = (ranked.sort("r")
            .group_by("segment_id", maintain_order=True)
            .agg(pl.first("english").alias("english"),
                 pl.first("translator").alias("translator")))

    before = seg.filter(pl.col("english").is_not_null()).height
    seg = (seg.drop("english", "translator")
           .join(best, on="segment_id", how="left")
           .sort(["uid", "seq"]))
    after = seg.filter(pl.col("english").is_not_null()).height
    seg.write_parquet(A / "segments.parquet")

    # documents: per-doc meta + best translator + available translators (ranked)
    seg2doc = seg.select("uid", "segment_id")
    avail = (trans.join(seg2doc, on="segment_id", how="inner")
             .join(rankdf, on="translator", how="left")
             .with_columns(pl.col("r").fill_null(len(ORDER)))
             .unique(["uid", "translator"])
             .sort("r")
             .group_by("uid", maintain_order=True)
             .agg(pl.col("translator").alias("translators")))
    docs = (
        seg.group_by("uid", maintain_order=True)
        .agg(
            pl.first("basket"), pl.first("nikaya"), pl.first("subpath"),
            pl.len().alias("n_segments"),
            pl.col("pali").str.len_chars().sum().alias("pali_chars"),
            pl.col("english").str.len_chars().sum().alias("english_chars"),
            (pl.col("english").is_not_null().sum() / pl.len()).alias("translated_frac"),
        )
        .join(avail, on="uid", how="left")
        # doc's primary translator = best (lowest-rank) available
        .with_columns(translator=pl.col("translators").list.first())
    )
    docs.write_parquet(A / "documents.parquet")

    multi = docs.filter(pl.col("translators").list.len() > 1).height
    print(f"translations.parquet: {len(trans)} rows across {trans['translator'].n_unique()} translators")
    print(f"filled {after - before} segments (now {after}/{seg.height} translated)")
    print(f"documents with >1 translator available: {multi}")


if __name__ == "__main__":
    main()
