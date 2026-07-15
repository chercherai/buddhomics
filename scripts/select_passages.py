"""Select ~12 complex untranslated passages for the translation bake-off.

Each passage is a contiguous run of segments from one document, skipping
heading segments (:0.*). Writes artifacts/bakeoff_passages.json.
"""

import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "artifacts" / "segments.sqlite"

# (uid, label, start_seq, n_segments) — start_seq chosen to land in real content
PICKS = [
    ("bv2", "Buddhavamsa 2 (Sumedha narrative, verse)", 40, 28),
    ("thi-ap21", "Theri-apadana 21 (verse autobiography)", 20, 28),
    ("ja531", "Jataka 531 Kusa (archaic narrative verse)", 60, 28),
    ("pv36", "Petavatthu 36 (ghost-story dialogue verse)", 30, 24),
    ("mil3.1.13", "Milindapanha 3.1.13 (dialogue prose)", 2, 14),
    ("ne10", "Netti 10 (hermeneutic technical prose)", 20, 12),
    ("pe3", "Petakopadesa 3 (technical, textually rough)", 40, 14),
    ("ps1.2", "Patisambhidamagga 1.2 (analytic prose)", 100, 28),
    ("cnd12", "Culaniddesa 12 (word-gloss commentary)", 30, 20),
    ("kv1.2", "Kathavatthu 1.2 (dialectical exchange)", 20, 28),
    ("ya1.2.1", "Yamaka 1.2.1 (paired logical questions)", 20, 14),
    ("patthana1.11", "Patthana 1.11 (conditional relations)", 20, 12),
]


def main() -> None:
    con = sqlite3.connect(DB)
    passages = []
    for uid, label, start, n in PICKS:
        rows = con.execute(
            """SELECT segment_id, pali FROM segments
               WHERE uid=? AND seq>=? AND pali IS NOT NULL AND trim(pali) != ''
               AND segment_id NOT LIKE '%:0.%'
               ORDER BY seq LIMIT ?""",
            (uid, start, n),
        ).fetchall()
        passages.append(
            dict(
                uid=uid, label=label,
                segments=[dict(id=sid, pali=p) for sid, p in rows],
                total_chars=sum(len(p) for _, p in rows),
            )
        )
    out = REPO / "artifacts" / "bakeoff_passages.json"
    out.write_text(json.dumps(passages, ensure_ascii=False, indent=2))
    for p in passages:
        print(f"{p['uid']:16} {len(p['segments']):3} segs {p['total_chars']:5} chars  {p['label']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
