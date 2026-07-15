"""Multi-scale cluster labels for the concept map.

Single-linkage hierarchical clustering of the concept t-SNE points, cut at
several distance heights (single-linkage on 2D t-SNE = spatial blobs at each
height). Each cluster of adequate size gets a centroid and a summary label =
its highest-document-frequency *content* term (function words filtered out).
The client shows coarse labels zoomed out, finer ones as you zoom in.

Writes artifacts/concept_clusters.json:
  {"levels": [{"thr": .., "clusters": [{"t": "..", "x": .., "y": .., "n": ..}]}]}
  ordered coarse -> fine.
"""

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

# (distance threshold, min cluster size to label) — coarse to fine
LEVELS = [(2.2, 28), (1.6, 10), (1.2, 5)]

# Pali particles, pronouns, copulas — poor cluster summaries
STOP = set("""ca kho hi pana ce vā tu no so sā taṁ tena tassa tāya na api eva evaṁ
iti ti yaṁ ye yo yā yena yassa imaṁ ayaṁ idaṁ ime imesaṁ ete etaṁ etassa esa esā
atha atho tato tattha tathā yathā seyyathā seyyathāpi hoti honti hessati ahosi
atthi natthi siyā assa assu me te vo ahaṁ tvaṁ mayaṁ tumhe tumhākaṁ amhākaṁ
tesaṁ tāsaṁ nesaṁ sabbe sabbaṁ sabbā sace yadā tadā yasmā tasmā yena kena
pe la vuccati ādi ādayo ādīni pi vā’ti cāti ceti hetaṁ tveva khvassa hidaṁ
bhavati bhavanti taṁyeva svāssa yañca yampi yena’ssa svāyaṁ""".split())


def main() -> None:
    terms = json.loads((A / "concepts.json").read_text())
    xy = np.array([[t["x"], t["y"]] for t in terms])
    df = np.array([t["df"] for t in terms])
    labels = [t["t"] for t in terms]

    L = linkage(xy, method="single")
    levels = []
    for thr, minsize in LEVELS:
        assign = fcluster(L, t=thr, criterion="distance")
        clusters = []
        for c in np.unique(assign):
            idx = np.where(assign == c)[0]
            if len(idx) < minsize:
                continue
            order = idx[np.argsort(df[idx])[::-1]]
            content = [labels[i] for i in order if labels[i] not in STOP]
            clusters.append({
                "t": content[0] if content else labels[order[0]],
                "terms": content[:18],          # for subagent semantic labelling
                "x": round(float(xy[idx, 0].mean()), 2),
                "y": round(float(xy[idx, 1].mean()), 2),
                "n": int(len(idx)),
            })
        clusters.sort(key=lambda c: -c["n"])
        levels.append({"thr": thr, "clusters": clusters})
        print(f"thr={thr}: {len(clusters)} labels — " +
              ", ".join(c["t"] for c in clusters[:8]))

    (A / "concept_clusters.json").write_text(
        json.dumps({"levels": levels}, ensure_ascii=False))
    print("wrote concept_clusters.json")


if __name__ == "__main__":
    main()
