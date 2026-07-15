"""Collection-level tree inference vs the canonical hierarchy.

Groups documents by collection (subpath), computes centroid SVD vectors,
runs average-linkage clustering on cosine distance, and renders an SVG
dendrogram with leaves colored by canonical top-level group — so agreement
and disagreement with the canonical tree is visible directly.

Writes artifacts/tree.svg and artifacts/tree.json.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

GROUP_COLORS = {
    "dn": "#3D5A80", "mn": "#5C7FA6", "sn": "#7FA0C4", "an": "#A3BEDA",
    "kn": "#8E2A18", "vinaya": "#3D6B4F", "abhidhamma": "#9A7B2F",
}


def top_group(basket: str, subpath: str) -> str:
    if basket == "vinaya":
        return "vinaya"
    if basket == "abhidhamma":
        return "abhidhamma"
    return subpath.split("/")[0]


def main() -> None:
    Z100 = np.load(A / "doc_svd.npy")
    uids = json.loads((A / "doc_index.json").read_text())
    docs = pl.read_parquet(A / "documents.parquet")
    meta = {r["uid"]: r for r in docs.iter_rows(named=True)}

    groups: dict[str, list[int]] = {}
    for i, uid in enumerate(uids):
        m = meta[uid]
        key = m["subpath"] if m["subpath"] else m["nikaya"]
        groups.setdefault(f"{m['basket']}::{key}", []).append(i)

    labels, cents, colors = [], [], []
    for key, idxs in sorted(groups.items()):
        if len(idxs) < 2:
            continue
        basket, sub = key.split("::")
        labels.append(sub)
        cents.append(Z100[idxs].mean(axis=0))
        colors.append(GROUP_COLORS.get(top_group(basket, sub), "#888"))
    C = np.vstack(cents)
    print(f"{len(labels)} collections")

    D = pdist(C, metric="cosine")
    L = linkage(D, method="average")
    dd = dendrogram(L, no_plot=True, labels=labels)

    # hand-rolled SVG: horizontal dendrogram
    order = dd["ivl"]
    n = len(order)
    row_h, label_w, w, pad = 16, 150, 900, 8
    h = n * row_h + 2 * pad
    xmax = float(max(max(c) for c in dd["dcoord"]))
    ypos = {lab: pad + (i + 0.5) * row_h for i, lab in enumerate(order)}
    lab_color = dict(zip(labels, colors))

    def sx(d):  # distance -> x (root at left)
        return label_w + (w - label_w - pad) * (1 - d / xmax)

    seg = []
    # scipy gives icoord (y in units of 5+10*i across leaves) and dcoord (heights)
    def iy(v):
        return pad + (v - 5.0) / 10.0 * row_h + 0.5 * row_h

    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        pts = [(sx(d), iy(i)) for i, d in zip(xs, ys)]
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        seg.append(f'<path d="{path}" fill="none" stroke="#98938a" stroke-width="1"/>')

    texts = [
        f'<text x="{label_w - 6}" y="{ypos[lab] + 4:.1f}" text-anchor="end" '
        f'fill="{lab_color[lab]}" font-weight="600">{lab}</text>'
        for lab in order
    ]
    legend = []
    for i, (g, c) in enumerate(GROUP_COLORS.items()):
        lx = label_w + 20 + i * 105
        legend.append(
            f'<rect x="{lx}" y="{h - 18}" width="10" height="10" fill="{c}"/>'
            f'<text x="{lx + 14}" y="{h - 9}" fill="#555">{g}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + 26}" '
        f'font-family="system-ui,sans-serif" font-size="11">'
        + "".join(seg) + "".join(texts) + "".join(legend) + "</svg>"
    )
    (A / "tree.svg").write_text(svg)
    (A / "tree.json").write_text(json.dumps(
        dict(labels=labels, order=order,
             linkage=[[float(v) for v in row] for row in L])))
    print("wrote tree.svg, tree.json")


if __name__ == "__main__":
    main()
