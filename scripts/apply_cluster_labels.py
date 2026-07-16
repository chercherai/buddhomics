"""Apply subagent semantic labels to a clusters file.

Usage: apply_cluster_labels.py <scratchdir> [clusters.json] [label_prefix]
Reads scratchdir/labels_{prefix}{0,1,2}.json ({cluster_index: label}) and sets
each cluster's "t" to the label, dropping the bulky "terms" arrays for the client.
Defaults: concept_clusters.json, prefix "L".  (Text clusters: text_clusters.json L->T)
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
TARGET = sys.argv[2] if len(sys.argv) > 2 else "concept_clusters.json"
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "L"


def main() -> None:
    cc = json.loads((A / TARGET).read_text())
    for li, lvl in enumerate(cc["levels"]):
        lf = SP / f"labels_{PREFIX}{li}.json"
        labels = json.loads(lf.read_text()) if lf.exists() else {}
        for i, cl in enumerate(lvl["clusters"]):
            lab = labels.get(str(i))
            if lab:
                cl["t"] = lab.strip()
            cl.pop("terms", None)
    (A / TARGET).write_text(json.dumps(cc, ensure_ascii=False))
    print(f"applied labels to {TARGET}; sample coarse:",
          ", ".join(c["t"] for c in cc["levels"][0]["clusters"][:6]))


if __name__ == "__main__":
    main()
