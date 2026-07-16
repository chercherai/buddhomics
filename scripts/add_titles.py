"""Augment map.json with each text's titles (from the :0.* header segments).

Adds two fields per point, in place (layout coordinates untouched):
  te — English title lines joined ("Middle Discourses 121 · The Shorter Discourse…")
  tl — Pali title lines joined    ("Majjhima Nikāya 121 · Cūḷasuññatasutta")

So titles become searchable (client folds te+tl into the doc search key) and
showable in the text tooltip. Run after any translation/substrate rebuild.
"""

import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"


def main() -> None:
    from pipeline_input import read_segments
    seg = read_segments()
    hdr = (seg.filter(pl.col("segment_id").str.contains(r":0\."))
           .sort(["uid", "seq"])
           .group_by("uid", maintain_order=True)
           .agg(pl.col("pali"), pl.col("english")))

    def join(vals):
        seen, out = set(), []
        for v in vals:
            v = (v or "").strip()
            if v and v not in seen:
                seen.add(v); out.append(v)
        return " · ".join(out)

    titles = {uid: (join(en), join(pa))
              for uid, pa, en in hdr.select("uid", "pali", "english").iter_rows()}

    # commentary segments have no :0. header — use the bibliographic displayName
    # (a title, not body text) from dharmanexus so they're identifiable/searchable
    comm_titles = {}
    dnf = REPO / "data" / "dharmanexus-pali" / "PA_files.json"
    if dnf.exists():
        comm_titles = {e["filename"]: e.get("displayName", "")
                       for e in json.loads(dnf.read_text())}

    pts = json.loads((A / "map.json").read_text())
    n = 0
    for p in pts:
        te, tl = titles.get(p["uid"], ("", ""))
        if not te and p.get("kind") == "commentary":
            te = comm_titles.get(p["uid"], "")
        if te:
            p["te"] = te
        if tl:
            p["tl"] = tl
        n += bool(te or tl)
    (A / "map.json").write_text(json.dumps(pts, ensure_ascii=False))
    print(f"added titles to {n}/{len(pts)} points")
    print("sample:", {k: titles.get(k) for k in ["mn121"] if k in titles})


if __name__ == "__main__":
    main()
