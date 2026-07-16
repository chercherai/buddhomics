"""Export the machine translations as bilara-data (sc-data) translation files.

For every machine-translated segment, write per-document JSON keyed by segment id
into the bilara layout, one author directory per model:

  translation/en/<author>/<basket>/<subpath>/<uid>_translation-en-<author>.json

Directories mirror root/pli/ms exactly (looked up from the local bilara clone).
Output goes to a staging dir for committing to a bilara-data fork.
"""

import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
ROOT = REPO / "data" / "bilara-data" / "root" / "pli" / "ms"
OUT = A / "bilara_mt"

# model id -> bilara author slug (dots -> hyphens for path/convention safety)
AUTHOR = {"claude-fable-5": "claude-fable-5", "gpt-5.6-sol": "gpt-5-6-sol"}


def main() -> None:
    uid_dir = {p.name.split("_")[0]: p.parent.relative_to(ROOT)
               for p in ROOT.rglob("*_root-pli-ms.json")}

    trans = pl.read_parquet(A / "translations.parquet").filter(pl.col("kind") == "machine")
    seg = pl.read_parquet(A / "segments.parquet").select("segment_id", "uid", "seq")
    m = trans.join(seg, on="segment_id", how="inner")

    n_files = n_segs = 0
    per_author = {}
    for (uid, model), g in m.group_by(["uid", "translator"]):
        author = AUTHOR.get(model, model)
        d = uid_dir.get(uid)
        if d is None:
            continue
        rows = g.sort("seq").select("segment_id", "english").rows()
        obj = {sid: en for sid, en in rows}
        outdir = OUT / "translation" / "en" / author / d
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{uid}_translation-en-{author}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
        n_files += 1
        n_segs += len(obj)
        per_author[author] = per_author.get(author, 0) + 1

    print(f"wrote {n_files} files, {n_segs} segments -> {OUT}")
    for a, n in sorted(per_author.items()):
        print(f"  {a}: {n} docs")


if __name__ == "__main__":
    main()
