"""Apply subagent semantic labels to concept_clusters.json.

Reads scratchpad/labels_L{0,1,2}.json ({cluster_index: label}) and sets each
cluster's "t" to the label, dropping the bulky "terms" arrays for the client.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")


def main() -> None:
    cc = json.loads((A / "concept_clusters.json").read_text())
    for li, lvl in enumerate(cc["levels"]):
        labels = json.loads((SP / f"labels_L{li}.json").read_text())
        for i, cl in enumerate(lvl["clusters"]):
            lab = labels.get(str(i))
            if lab:
                cl["t"] = lab.strip()
            cl.pop("terms", None)
    (A / "concept_clusters.json").write_text(json.dumps(cc, ensure_ascii=False))
    print("applied labels; sample coarse:",
          ", ".join(c["t"] for c in cc["levels"][0]["clusters"][:6]))


if __name__ == "__main__":
    main()
