"""Curated taxonomy -> per-term document membership for the map's browse panel.

Each term is defined by stem prefixes (and/or exact token forms, prefixed "="),
matched against the full token inventory of every mapped document. Output is
artifacts/taxonomy.json:

  {"order": [uid, ...],            # map.json point order
   "tree": [{"h": "People", "subs": [
       {"h": "Chief disciples", "terms": [
           {"t": "Sāriputta", "docs": [i, i, ...]}, ...]}]}]}

Doc indices index into map.json's point array. Header/sub counts are computed
client-side as unions. Edit the TAXONOMY table freely — stems are matched as
token prefixes, "=token" must match exactly.
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

TOKEN = re.compile(r"[a-zāīūṁṃṅñṭḍṇḷ’']+")
TAGS = re.compile(r"<[^>]+>")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(TAGS.sub(" ", text.lower()))


TAXONOMY = [
    ("People", [
        ("Chief disciples", [
            ("Sāriputta", ["sāriputt", "upatiss"]),
            ("Moggallāna", ["moggallān", "kolit"]),
            ("Mahākassapa", ["mahākassap"]),
            ("Ānanda", ["ānand"]),
            ("Anuruddha", ["anuruddh"]),
            ("Upāli", ["upāli"]),
            ("Rāhula", ["rāhul"]),
            ("Mahākaccāna", ["mahākaccān", "kaccān"]),
        ]),
        ("Nuns", [
            ("Mahāpajāpatī", ["mahāpajāpat"]),
            ("Uppalavaṇṇā", ["uppalavaṇṇ"]),
            ("Dhammadinnā", ["dhammadinn"]),
            ("Kisāgotamī", ["kisāgotam"]),
            ("Bhaddā Kuṇḍalakesā", ["bhaddākuṇḍalakes"]),
        ]),
        ("Kings & patrons", [
            ("Pasenadi", ["pasenad"]),
            ("Bimbisāra", ["bimbisār"]),
            ("Ajātasattu", ["ajātasatt"]),
            ("Anāthapiṇḍika", ["anāthapiṇḍik"]),
            ("Visākhā", ["visākh"]),
            ("Jīvaka", ["jīvak"]),
            ("Milinda", ["milind"]),
        ]),
        ("Interlocutors & rivals", [
            ("Devadatta", ["devadatt"]),
            ("Saccaka", ["saccak"]),
            ("Vacchagotta", ["vacchagott"]),
            ("Nigaṇṭha Nātaputta", ["nigaṇṭh"]),
            ("Ambaṭṭha", ["ambaṭṭh"]),
            ("Assalāyana", ["assalāyan"]),
        ]),
    ]),
    ("Beings", [
        ("Devas & brahmās", [
            ("Sakka", ["=sakko", "=sakkassa", "=sakkena", "=sakkaṁ", "devānamind"]),
            ("Brahmā", ["=brahmā", "=brahmuno", "=brahmunā", "sahampat", "brahmalok"]),
            ("Māra", ["=māro", "=mārassa", "=māraṁ", "=mārena", "māradheyy", "mārasen"]),
            ("Devatās", ["devat", "tāvatiṁs", "devalok"]),
        ]),
        ("Other beings", [
            ("Nāgas", ["nāg"]),
            ("Yakkhas", ["yakkh"]),
            ("Gandhabbas", ["gandhabb"]),
            ("Asuras", ["asur"]),
            ("Petas", ["=petā", "=peto", "=petānaṁ", "pettivisay", "petavatth"]),
        ]),
    ]),
    ("Places", [
        ("Cities", [
            ("Sāvatthī", ["sāvatth"]),
            ("Rājagaha", ["rājagah"]),
            ("Vesālī", ["vesāl"]),
            ("Kosambī", ["kosamb"]),
            ("Kapilavatthu", ["kapilavatth"]),
            ("Bārāṇasī", ["bārāṇas"]),
        ]),
        ("Parks & mountains", [
            ("Jetavana", ["jetavan"]),
            ("Veḷuvana", ["veḷuvan"]),
            ("Gijjhakūṭa", ["gijjhakūṭ"]),
            ("Isipatana", ["isipatan"]),
            ("Migāramātupāsāda", ["migāramāt"]),
            ("Nigrodhārāma", ["nigrodhārām"]),
        ]),
        ("Regions", [
            ("Kosala", ["kosal"]),
            ("Magadha", ["magadh"]),
            ("Sakya country", ["saky", "sākiy"]),
            ("Vajjī", ["=vajjī", "=vajjīnaṁ", "vajjiput"]),
        ]),
    ]),
    ("Numbered Lists", [
        ("Aggregates (5)", [
            ("khandhā (the set)", ["khandh"]),
            ("vedanā", ["vedan"]),
            ("saññā", ["saññ"]),
            ("saṅkhārā", ["saṅkhār"]),
            ("viññāṇa", ["viññāṇ"]),
        ]),
        ("Faculties & powers (5)", [
            ("indriya", ["indriy"]),
            ("saddhā", ["saddh"]),
            ("vīriya", ["vīriy", "viriy"]),
            ("sati", ["sati"]),
            ("samādhi", ["samādh"]),
            ("paññā", ["paññ"]),
        ]),
        ("Wings to awakening (37)", [
            ("satipaṭṭhānā (4)", ["satipaṭṭhān"]),
            ("sammappadhānā (4)", ["sammappadhān"]),
            ("iddhipādā (4)", ["iddhipād"]),
            ("bojjhaṅgā (7)", ["bojjhaṅg"]),
            ("eightfold path", ["aṭṭhaṅgik", "sammādiṭṭh", "sammāsaṅkapp",
                                "sammāvāc", "sammākammant", "sammāājīv",
                                "sammāvāyām", "sammāsati", "sammāsamādh"]),
        ]),
        ("Truths & origination", [
            ("four truths", ["ariyasacc", "dukkhasamuday", "dukkhanirodh"]),
            ("dependent origination (12)", ["paṭiccasamupp", "avijj", "nāmarūp",
                                            "saḷāyatan", "jarāmaraṇ"]),
            ("taṇhā", ["taṇh"]),
            ("upādāna", ["upādān"]),
        ]),
        ("Meditations", [
            ("jhānā (4)", ["jhān"]),
            ("formless attainments (4)", ["ākāsānañc", "viññāṇañc", "ākiñcaññ",
                                          "nevasaññ"]),
            ("brahmavihārā (4)", ["mett", "karuṇ", "mudit", "upekkh"]),
            ("ānāpānassati", ["ānāpān"]),
            ("kasiṇā (10)", ["kasiṇ"]),
        ]),
        ("Hindrances & fetters", [
            ("nīvaraṇā (5)", ["nīvaraṇ", "kāmacchand", "byāpād", "thinamiddh",
                              "thīnamiddh", "uddhaccakukkucc", "vicikicch"]),
            ("saṁyojanā (10)", ["saṁyojan", "saññojan"]),
            ("āsavā", ["āsav"]),
        ]),
    ]),
]


def main() -> None:
    uids = [p["uid"] for p in json.loads((A / "map.json").read_text())]
    segs = pl.read_parquet(A / "segments.parquet")
    texts = dict(
        segs.filter(pl.col("pali").is_not_null())
        .group_by("uid")
        .agg(pl.col("pali").str.join(" ").alias("text"))
        .iter_rows()
    )
    corpus = [texts[u] for u in uids]

    vec = CountVectorizer(analyzer=tokenize, binary=True, min_df=1)
    X = vec.fit_transform(corpus).tocsc()
    vocab = vec.get_feature_names_out()
    print(f"vocab: {len(vocab)} tokens over {len(uids)} docs")

    def docs_for(stems: list[str]) -> list[int]:
        cols = []
        for s in stems:
            if s.startswith("="):
                t = s[1:]
                idx = np.searchsorted(vocab, t)
                if idx < len(vocab) and vocab[idx] == t:
                    cols.append(idx)
            else:
                lo = np.searchsorted(vocab, s)
                hi = np.searchsorted(vocab, s + "￿")
                cols.extend(range(lo, hi))
        if not cols:
            return []
        hit = np.asarray(X[:, cols].sum(axis=1)).ravel() > 0
        return np.where(hit)[0].tolist()

    tree = []
    for header, subs in TAXONOMY:
        node = {"h": header, "subs": []}
        for sub, terms in subs:
            snode = {"h": sub, "terms": []}
            for label, stems in terms:
                d = docs_for(stems)
                snode["terms"].append({"t": label, "docs": d})
                if not d:
                    print(f"  WARNING: no docs for {label} ({stems})")
            node["subs"].append(snode)
        tree.append(node)

    out = {"order": uids, "tree": tree}
    (A / "taxonomy.json").write_text(json.dumps(out, ensure_ascii=False))
    size = (A / "taxonomy.json").stat().st_size / 1e6
    nterms = sum(len(s["terms"]) for n in tree for s in n["subs"])
    print(f"wrote taxonomy.json: {nterms} terms, {size:.2f} MB")


if __name__ == "__main__":
    main()
