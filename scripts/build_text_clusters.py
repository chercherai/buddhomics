"""Multi-scale cluster labels for the TEXTS map (document t-SNE).

Ward linkage on the document t-SNE points, cut to a target cluster count at
three zoom scales (coarse -> fine). Ward gives spatially compact clusters, so
label anchors sit at meaningful centroids and a big text (e.g. the finely-split
Paṭṭhāna) stays one region instead of fragmenting into near-duplicate blobs the
way single-linkage chaining did.

Each cluster carries its collection mix, aggregated top terms, and sample uids
for subagent semantic labelling (see clusters_T*.txt payloads).

Writes artifacts/text_clusters.json:
  {"levels": [{"k": .., "clusters": [{"t","terms","sub","uids","x","y","n"}]}]}
ordered coarse -> fine. (Client keys levels by zoom index, not by k/thr.)
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

LEVELS = [24, 44, 72]   # target cluster count, coarse -> fine


def main() -> None:
    pts = json.loads((A / "map.json").read_text())
    xy = np.array([[p["tx"], p["ty"]] for p in pts])
    L = linkage(xy, method="ward")
    levels = []
    for k in LEVELS:
        assign = fcluster(L, t=k, criterion="maxclust")
        clusters = []
        for c in np.unique(assign):
            idx = np.where(assign == c)[0]
            members = [pts[i] for i in idx]
            term_ct = Counter(t for m in members for t in m["terms"])
            sub_ct = Counter(m["subpath"] for m in members)
            # sample section/text titles — the strongest signal for what a
            # cluster is (esp. commentary, whose titles name the commented text)
            title_ct = Counter(m["te"].split(" · ")[-1] for m in members if m.get("te"))
            clusters.append({
                "t": term_ct.most_common(1)[0][0] if term_ct else "?",
                "terms": [t for t, _ in term_ct.most_common(15)],
                "titles": [t for t, _ in title_ct.most_common(8)],
                "sub": ", ".join(f"{s} ({n})" for s, n in sub_ct.most_common(4)),
                "uids": [m["uid"] for m in members[:5]],
                "x": round(float(xy[idx, 0].mean()), 2),
                "y": round(float(xy[idx, 1].mean()), 2),
                "n": int(len(idx)),
            })
        clusters.sort(key=lambda c: -c["n"])
        levels.append({"k": k, "clusters": clusters})
        print(f"k={k}: {len(clusters)} clusters "
              f"(largest n={clusters[0]['n']}, smallest n={clusters[-1]['n']})")

    (A / "text_clusters.json").write_text(json.dumps({"levels": levels}, ensure_ascii=False))

    # payload files for subagent labelling
    sp = Path("/private/tmp/claude-501/-Users-ericcollins-Desktop-claude-code-buddhomics/"
              "d8aef17c-4a7b-4f39-81c4-6b782fb8b480/scratchpad")
    sp.mkdir(parents=True, exist_ok=True)
    for li, lvl in enumerate(levels):
        lines = []
        for i, c in enumerate(lvl["clusters"]):
            titles = f" titles: {'; '.join(c['titles'])}." if c["titles"] else ""
            lines.append(f"Cluster {i} [{c['n']} texts; collections: {c['sub']};"
                         f"{titles} e.g. {', '.join(c['uids'])}]: {', '.join(c['terms'])}")
        (sp / f"clusters_T{li}.txt").write_text("\n".join(lines))
    print("wrote text_clusters.json + clusters_T0/1/2.txt")


if __name__ == "__main__":
    main()
