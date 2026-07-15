"""NCPED (via sc-data) -> site/dictionary.json: headword -> list of senses.

Source: data/pli2en_ncped.json, downloaded from
https://raw.githubusercontent.com/suttacentral/sc-data/main/dictionaries/simple/en/pli2en_ncped.json
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "pli2en_ncped.json"


def main() -> None:
    entries = json.loads(SRC.read_text())
    out: dict[str, list] = {}
    for e in entries:
        hw = e["entry"].strip().lower()
        sense = {}
        if e.get("grammar"):
            sense["g"] = e["grammar"]
        if e.get("definition"):
            d = e["definition"]
            sense["d"] = "; ".join(d) if isinstance(d, list) else d
        if e.get("xr"):
            sense["x"] = e["xr"]
        if sense:
            out.setdefault(hw, []).append(sense)
    dst = REPO / "site" / "dictionary.json"
    dst.write_text(json.dumps(out, ensure_ascii=False))
    print(f"{len(out)} headwords, {dst.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
