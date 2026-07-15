"""Multi-scale cluster labels for the TEXTS map (document t-SNE).

Single-linkage on the document t-SNE points, cut at three distance heights.
Each cluster carries its collection mix, aggregated top terms, and sample uids
for subagent semantic labelling (see clusters_T*.txt payloads).

Writes artifacts/text_clusters.json:
  {"levels": [{"thr": .., "clusters": [{"t","terms","sub","uids","x","y","n"}]}]}
ordered coarse -> fine.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

LEVELS = [(4.2, 40), (3.3, 20), (2.6, 10)]   # (distance threshold, min size) coarse->fine


def main() -> None:
    pts = json.loads((A / "map.json").read_text())
    xy = np.array([[p["tx"], p["ty"]] for p in pts])
    L = linkage(xy, method="single")
    levels = []
    for thr, minsize in LEVELS:
        assign = fcluster(L, t=thr, criterion="distance")
        clusters = []
        for c in np.unique(assign):
            idx = np.where(assign == c)[0]
            if len(idx) < minsize:
                continue
            members = [pts[i] for i in idx]
            term_ct = Counter(t for m in members for t in m["terms"])
            sub_ct = Counter(m["subpath"] for m in members)
            clusters.append({
                "t": term_ct.most_common(1)[0][0] if term_ct else "?",
                "terms": [t for t, _ in term_ct.most_common(15)],
                "sub": ", ".join(f"{s} ({n})" for s, n in sub_ct.most_common(4)),
                "uids": [m["uid"] for m in members[:5]],
                "x": round(float(xy[idx, 0].mean()), 2),
                "y": round(float(xy[idx, 1].mean()), 2),
                "n": int(len(idx)),
            })
        clusters.sort(key=lambda c: -c["n"])
        levels.append({"thr": thr, "clusters": clusters})
        print(f"thr={thr}: {len(clusters)} clusters (largest n={clusters[0]['n']})")

    (A / "text_clusters.json").write_text(json.dumps({"levels": levels}, ensure_ascii=False))

    # payload files for subagent labelling
    sp = Path("/private/tmp/claude-501/-Users-ericcollins-Desktop-claude-code-buddhomics/"
              "d8aef17c-4a7b-4f39-81c4-6b782fb8b480/scratchpad")
    sp.mkdir(parents=True, exist_ok=True)
    for li, lvl in enumerate(levels):
        lines = []
        for i, c in enumerate(lvl["clusters"]):
            lines.append(f"Cluster {i} [{c['n']} texts; collections: {c['sub']}; "
                         f"e.g. {', '.join(c['uids'])}]: {', '.join(c['terms'])}")
        (sp / f"clusters_T{li}.txt").write_text("\n".join(lines))
    print("wrote text_clusters.json + clusters_T0/1/2.txt")


if __name__ == "__main__":
    main()
