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
from obfusc import encode
from pipeline_input import read_segments

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
OUT = REPO / "site" / "texts"

ORDER = HUMAN_PREF + MACHINE_PREF + ["dharmamitra"]
NAMES = {
    "sujato": "Bhikkhu Sujato", "brahmali": "Bhikkhu Brahmali", "kelly": "John Kelly",
    "soma": "Bhikkhu Soma", "patton": "Charles Patton", "suddhaso": "Bhikkhu Suddhāso",
    "anandajoti": "Bhikkhu Ānandajoti", "kovilo": "Bhikkhu Kovilo",
    "claude-fable-5": "Claude Fable 5", "gpt-5.6-sol": "GPT-5.6",
    "dharmamitra": "DharmaMitra",
}


def rank(t: str) -> int:
    return ORDER.index(t) if t in ORDER else len(ORDER)


def main() -> None:
    segs = read_segments()
    trans = pl.read_parquet(A / "translations.parquet")
    tkind = dict(trans.select("translator", "kind").unique().iter_rows())
    tkind["dharmamitra"] = "machine"

    bytr: dict[str, dict[str, str]] = defaultdict(dict)
    for sid, tr, en in trans.select("segment_id", "translator", "english").iter_rows():
        bytr[sid][tr] = en
    # commentary English (DharmaMitra) lives on the combined segments, not in
    # translations.parquet — fold it in
    if "translator" in segs.columns:
        for sid, tr, en in segs.filter(pl.col("translator") == "dharmamitra").select(
                "segment_id", "translator", "english").iter_rows():
            if en:
                bytr[sid][tr] = en

    has_kind = "kind" in segs.columns
    OUT.mkdir(exist_ok=True)
    n = 0
    for (uid,), df in segs.group_by("uid", maintain_order=True):
        commentary = has_kind and df["kind"][0] == "commentary"
        rows, present, cover = [], set(), defaultdict(int)
        for r in df.sort("seq").iter_rows(named=True):
            sid = r["segment_id"]
            td = bytr.get(sid, {})
            pali = r["pali"]
            if commentary:   # obfuscate served text (licensing pending); reader decodes
                pali = encode(pali, sid) if pali else pali
                td = {t: encode(v, sid) for t, v in td.items()}
            rows.append([sid, pali, td])
            for t in td:
                present.add(t)
                cover[t] += 1
        translators = [{"id": t, "kind": tkind.get(t, "human"), "name": NAMES.get(t, t)}
                       for t in sorted(present, key=rank)]
        default = None
        if present:
            maxcov = max(cover.values())
            eligible = [t for t in present if cover[t] >= 0.5 * maxcov]
            default = min(eligible, key=rank)
        doc = dict(uid=uid, translators=translators, default=default, segs=rows)
        if commentary:
            doc["obf"] = True
        (OUT / f"{uid}.json").write_text(json.dumps(doc, ensure_ascii=False))
        n += 1
    total_mb = sum(f.stat().st_size for f in OUT.glob("*.json")) / 1e6
    print(f"wrote {n} files, {total_mb:.1f} MB total")


if __name__ == "__main__":
    main()
