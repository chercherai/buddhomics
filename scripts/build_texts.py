"""Export per-document segment JSON for the site's reading panel.

One file per uid: site/texts/{uid}.json
  {"uid", "translators": [{"id","kind","name"}...], "default": id,
   "segs": [[segment_id, pali, {translatorId: english, ...}], ...]}

Segments in seq order; the per-segment object holds every available translation
(all sc-data humans + machine layers), so the reader can switch translators.
"""

import json
from collections import defaultdict
from pathlib import Path

import polars as pl

from build_substrate import HUMAN_PREF
from merge_translations import MACHINE_PREF

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
OUT = REPO / "site" / "texts"

ORDER = HUMAN_PREF + MACHINE_PREF
NAMES = {
    "sujato": "Bhikkhu Sujato", "brahmali": "Bhikkhu Brahmali", "kelly": "John Kelly",
    "soma": "Bhikkhu Soma", "patton": "Charles Patton", "suddhaso": "Bhikkhu Suddhāso",
    "anandajoti": "Bhikkhu Ānandajoti", "kovilo": "Bhikkhu Kovilo",
    "claude-fable-5": "Claude Fable 5", "gpt-5.6-sol": "GPT-5.6",
}


def rank(t: str) -> int:
    return ORDER.index(t) if t in ORDER else len(ORDER)


def main() -> None:
    segs = pl.read_parquet(A / "segments.parquet")
    trans = pl.read_parquet(A / "translations.parquet")
    tkind = dict(trans.select("translator", "kind").unique().iter_rows())

    bytr: dict[str, dict[str, str]] = defaultdict(dict)
    for sid, tr, en in trans.select("segment_id", "translator", "english").iter_rows():
        bytr[sid][tr] = en

    OUT.mkdir(exist_ok=True)
    n = 0
    for (uid,), df in segs.group_by("uid", maintain_order=True):
        rows, present, cover = [], set(), defaultdict(int)
        for r in df.sort("seq").iter_rows(named=True):
            sid = r["segment_id"]
            td = bytr.get(sid, {})
            rows.append([sid, r["pali"], td])
            for t in td:
                present.add(t)
                cover[t] += 1
        translators = [{"id": t, "kind": tkind.get(t, "human"), "name": NAMES.get(t, t)}
                       for t in sorted(present, key=rank)]
        # default: the most-preferred translator that substantially covers the doc
        # (so sujato wins over a marginally-fuller soma, but a token human presence
        # doesn't beat a complete machine rendering)
        default = None
        if present:
            maxcov = max(cover.values())
            eligible = [t for t in present if cover[t] >= 0.5 * maxcov]
            default = min(eligible, key=rank)
        (OUT / f"{uid}.json").write_text(json.dumps(
            dict(uid=uid, translators=translators, default=default, segs=rows),
            ensure_ascii=False))
        n += 1
    total_mb = sum(f.stat().st_size for f in OUT.glob("*.json")) / 1e6
    print(f"wrote {n} files, {total_mb:.1f} MB total")


if __name__ == "__main__":
    main()
