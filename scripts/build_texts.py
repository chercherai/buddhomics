"""Export per-document segment JSON for the site's reading panel.

One file per uid: site/texts/{uid}.json
  {"uid": ..., "translator": ..., "segs": [[segment_id, pali, english], ...]}
Segments in seq order; english null where untranslated.
"""

import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "site" / "texts"


def main() -> None:
    segs = pl.read_parquet(REPO / "artifacts" / "segments.parquet")
    OUT.mkdir(exist_ok=True)
    n = 0
    for (uid,), df in segs.group_by("uid", maintain_order=True):
        rows = [
            [r["segment_id"], r["pali"], r["english"]]
            for r in df.sort("seq").iter_rows(named=True)
        ]
        translator = df["translator"][0]
        (OUT / f"{uid}.json").write_text(
            json.dumps(dict(uid=uid, translator=translator, segs=rows),
                       ensure_ascii=False)
        )
        n += 1
    total_mb = sum(f.stat().st_size for f in OUT.glob("*.json")) / 1e6
    print(f"wrote {n} files, {total_mb:.1f} MB total")


if __name__ == "__main__":
    main()
