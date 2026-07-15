"""Prepare the layered dictionary for the site.

- DPD (inflected-form keyed, 142k entries) -> sharded site/dict/<key>.json
  (key = first two ASCII-folded chars), lazy-loaded on lookup.
- DPPN proper names -> site/names.json (cleaned text), loaded once.
- Concept-term English glosses: resolve each concepts.json term through DPD
  (fallback NCPED) to a short gloss, written back as term["en"] for the
  Pali/English toggle on the concept map.
- NCPED stays as site/dictionary.json (already built) for reverse English
  search + concise prose fallback.
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SITE = REPO / "site"
A = REPO / "artifacts"

FOLD = str.maketrans({
    "ā": "a", "ī": "i", "ū": "u", "ṁ": "m", "ṃ": "m", "ṅ": "n", "ñ": "n",
    "ṭ": "t", "ḍ": "d", "ṇ": "n", "ḷ": "l", "ṛ": "r", "ṝ": "r", "ḥ": "h",
    "’": "", "'": "",
})
TAG = re.compile(r"<[^>]+>")
BOLD = re.compile(r"<b>(.*?)</b>")
WS = re.compile(r"\s+")


def shardkey(form: str) -> str:
    f = form.lower().translate(FOLD)
    f = re.sub(r"[^a-z]", "", f)
    return (f[:2] or "_").ljust(2, "_")


# curated glosses for core terms where DPD's first (often homonym/adjective)
# sense misleads; keyed by lemma stem, matched as a prefix of the concept form.
STEM_OVERRIDES = {
    "nibbān": "extinguishment", "nibbāy": "extinguishment",
    "dukkh": "suffering", "sukh": "happiness", "magg": "path",
    "taṇh": "craving", "avijj": "ignorance", "mett": "loving-kindness",
    "karuṇ": "compassion", "mudit": "rejoicing", "upekkh": "equanimity",
    "paññ": "wisdom", "samādh": "immersion", "saññ": "perception",
    "saṅkhār": "conditions", "viññāṇ": "consciousness", "vedan": "feeling",
    "khandh": "aggregate", "kamm": "action", "saṁsār": "transmigration",
    "brahmacariy": "spiritual life", "arahant": "perfected one",
    "arahat": "perfected one", "sīl": "ethics", "jhān": "absorption",
    "saddh": "faith", "vīriy": "energy", "viriy": "energy",
    "bojjhaṅg": "awakening factor", "satipaṭṭhān": "mindfulness meditation",
    "indriy": "faculty", "āsav": "defilement", "upādān": "grasping",
    "bhav": "existence", "jāt": "rebirth", "nāmarūp": "name and form",
    "ariyasacc": "noble truth", "anicc": "impermanence", "anatt": "not-self",
}


def override(form: str) -> str | None:
    for stem, g in STEM_OVERRIDES.items():
        if form.startswith(stem):
            return g
    return None


def gloss_dpd(defs: list[str]) -> str | None:
    for d in defs:
        m = BOLD.search(d)
        if m:
            g = WS.sub(" ", m.group(1).split(";")[0]).strip()
            if g:
                return g[:42]
    return None


def gloss_ncped(senses: list) -> str | None:
    for s in senses:
        if s.get("d"):
            return WS.sub(" ", s["d"].split(";")[0].split(",")[0]).strip()[:42]
    return None


def main() -> None:
    dpd = json.loads((DATA / "pli2en_dpd.json").read_text())
    dpd_map = {e["entry"].lower(): e["definition"] for e in dpd}

    # shard DPD
    out = SITE / "dict"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    shards: dict[str, dict] = defaultdict(dict)
    for form, defs in dpd_map.items():
        shards[shardkey(form)][form] = defs
    for key, entries in shards.items():
        (out / f"{key}.json").write_text(json.dumps(entries, ensure_ascii=False))
    total = sum((out / f"{k}.json").stat().st_size for k in shards) / 1e6
    print(f"DPD: {len(dpd_map):,} forms -> {len(shards)} shards, {total:.1f} MB")

    # DPPN names -> plain-text
    dppn = json.loads((DATA / "pli2en_dppn.json").read_text())
    names = {}
    for e in dppn:
        txt = WS.sub(" ", TAG.sub(" ", e["text"])).strip()
        names[e["word"].lower()] = txt[:600]
    (SITE / "names.json").write_text(json.dumps(names, ensure_ascii=False))
    print(f"DPPN: {len(names)} names -> names.json "
          f"({(SITE/'names.json').stat().st_size/1e6:.2f} MB)")

    # concept glosses
    ncped = {}
    for e in json.loads((DATA / "pli2en_ncped.json").read_text()):
        hw = e["entry"].strip().lower()
        d = e.get("definition")
        d = "; ".join(d) if isinstance(d, list) else d
        ncped.setdefault(hw, []).append({"d": d} if d else {})

    terms = json.loads((A / "concepts.json").read_text())
    n_gloss = 0
    for t in terms:
        w = t["t"].lower()
        g = override(w)
        if not g and w in dpd_map:
            g = gloss_dpd(dpd_map[w])
        if not g and w in ncped:
            g = gloss_ncped(ncped[w])
        if not g:
            # try DPD lemma via truncation of long forms
            for k in (w[:-1], w[:-2], w[:-3]):
                if len(k) >= 4 and k in dpd_map:
                    g = gloss_dpd(dpd_map[k]); break
        t["en"] = g or ""
        n_gloss += bool(g)
    (A / "concepts.json").write_text(json.dumps(terms, ensure_ascii=False))
    shutil.copy(A / "concepts.json", SITE / "concepts.json")
    print(f"concept glosses: {n_gloss}/{len(terms)} terms resolved")


if __name__ == "__main__":
    main()
