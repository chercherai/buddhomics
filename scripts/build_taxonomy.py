"""Curated taxonomy -> per-term document membership for the map's browse panel.

Each term is defined by stem prefixes, exact token forms (prefixed "="), or
multi-word phrases (prefixed "~", substring match on the tag-stripped lowercase
text), matched against every mapped document. Output is
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
    ("Rhetoric", [
        ("Openings & formulas", [
            ("evaṁ me sutaṁ — “thus have I heard”", ["~evaṁ me sutaṁ"]),
            ("sāvatthinidāna — the Sāvatthī setting", ["sāvatthinidān"]),
            ("yathābhūtaṁ — “as it really is”", ["yathābhūt"]),
            ("…pe… — elided repetition", ["=pe"]),
        ]),
        ("Modes of address", [
            ("bhikkhave — “monks!”", ["=bhikkhave"]),
            ("bhante — “venerable sir”", ["=bhante"]),
            ("āvuso — “friend”", ["=āvuso"]),
            ("mahārāja — “great king”", ["mahārāj"]),
            ("gahapati — “householder”", ["gahapat"]),
        ]),
        ("Similes", [
            ("seyyathāpi — “just as…”", ["seyyathāp"]),
            ("upamā — comparisons", ["upam"]),
            ("opamma — the simile named", ["opamm"]),
        ]),
        ("Verse & debate", [
            ("gāthā — verses", ["gāth"]),
            ("na … vattabbe — “it should not be said”", ["vattabb"]),
            ("āmantā — debate assent", ["=āmantā"]),
            ("pucchā — questions", ["pucch"]),
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

    # normalized full text for phrase matching
    flat = [" ".join(tokenize(t)) for t in corpus]

    def docs_for(stems: list[str]) -> list[int]:
        cols, phrase_hits = [], set()
        for s in stems:
            if s.startswith("~"):
                phrase = " ".join(tokenize(s[1:]))
                phrase_hits.update(i for i, t in enumerate(flat) if phrase in t)
            elif s.startswith("="):
                t = s[1:]
                idx = np.searchsorted(vocab, t)
                if idx < len(vocab) and vocab[idx] == t:
                    cols.append(idx)
            else:
                lo = np.searchsorted(vocab, s)
                hi = np.searchsorted(vocab, s + "￿")
                cols.extend(range(lo, hi))
        hits = phrase_hits
        if cols:
            mask = np.asarray(X[:, cols].sum(axis=1)).ravel() > 0
            hits = hits | set(np.where(mask)[0].tolist())
        return sorted(hits)

    # ---- Translations: which languages each text is available in ----
    LANG_NAMES = {
        "en": "English", "de": "German", "ru": "Russian", "sr": "Serbian",
        "fr": "French", "lt": "Lithuanian", "it": "Italian", "pt": "Portuguese",
        "pl": "Polish", "tr": "Turkish", "jpn": "Japanese", "es": "Spanish",
        "zh": "Chinese", "vi": "Vietnamese", "id": "Indonesian", "et": "Estonian",
        "cs": "Czech", "nl": "Dutch", "no": "Norwegian", "fi": "Finnish",
        "sl": "Slovenian", "ko": "Korean", "hi": "Hindi", "gu": "Gujarati",
        "mr": "Marathi", "ta": "Tamil", "kan": "Kannada", "ka": "Georgian",
        "si": "Sinhala", "th": "Thai", "my": "Burmese", "lo": "Lao",
        "gsw": "Swiss German",
    }
    uid_pos = {u: i for i, u in enumerate(uids)}
    tdir = REPO / "data" / "bilara-data" / "translation"
    lang_docs: dict[str, set[int]] = {}
    for langdir in sorted(tdir.iterdir()):
        if not langdir.is_dir():
            continue
        hits = set()
        for f in langdir.rglob("*.json"):
            i = uid_pos.get(f.name.split("_")[0])
            if i is not None:
                hits.add(i)
        if hits:
            lang_docs[langdir.name] = hits

    tree = []
    for header, subs in TAXONOMY:
        node = {"h": header, "subs": []}
        for sub, terms in subs:
            snode = {"h": sub, "terms": []}
            for label, stems in terms:
                d = docs_for(stems)
                snode["terms"].append({"t": label, "docs": d, "stems": stems})
                if not d:
                    print(f"  WARNING: no docs for {label} ({stems})")
            node["subs"].append(snode)
        tree.append(node)

    # merge subagent-curated additions (stems pre-validated against the corpus in
    # taxonomy_additions.json): extend existing categories/subgroups by name, and
    # append new categories (Cosmology, Doctrine & Mind, Abhidhamma, Vinaya, …)
    add_path = REPO / "scripts" / "taxonomy_additions.json"
    if add_path.exists():
        node_by_h = {n["h"]: n for n in tree}
        for cat, subs in json.loads(add_path.read_text()).items():
            node = node_by_h.get(cat)
            if node is None:
                node = {"h": cat, "subs": []}
                tree.append(node)
                node_by_h[cat] = node
            sub_by_h = {s["h"]: s for s in node["subs"]}
            for sub in subs:
                snode = sub_by_h.get(sub["h"])
                if snode is None:
                    snode = {"h": sub["h"], "terms": []}
                    node["subs"].append(snode)
                    sub_by_h[sub["h"]] = snode
                for term in sub["terms"]:
                    d = docs_for(term["stems"])
                    if d:
                        snode["terms"].append(
                            {"t": term["t"], "docs": d, "stems": term["stems"]})

    # ---- Translations: language availability + who produced the English ----
    MACHINE = {"claude-fable-5", "gpt-5.6-sol"}
    TR_NAMES = {"sujato": "Bhikkhu Sujato", "brahmali": "Bhikkhu Brahmali",
                "kelly": "John Kelly", "kovilo": "Bhikkhu Kovilo",
                "soma": "Bhikkhu Soma", "suddhaso": "Bhikkhu Suddhāso",
                "patton": "Charles Patton", "anandajoti": "Bhikkhu Ānandajoti",
                "claude-fable-5": "Claude Fable 5", "gpt-5.6-sol": "GPT-5.6"}
    docmeta = pl.read_parquet(A / "documents.parquet")

    # English availability = any English at all (human OR machine), from the substrate
    frac = dict(docmeta.select("uid", "translated_frac").iter_rows())
    english = {i for i, u in enumerate(uids) if (frac.get(u) or 0) > 0}
    by_lang = [("English", sorted(english))] + sorted(
        ((LANG_NAMES.get(l, l.title()), sorted(d)) for l, d in lang_docs.items()
         if l != "en" and len(d) >= 25),
        key=lambda kv: -len(kv[1]),
    )

    # categorize by every AVAILABLE translator (not just the doc's primary), so
    # alternate sc-data humans and the GPT fallback both surface; a doc appears
    # under each translator that has a version of it
    uid_trs = dict(docmeta.select("uid", "translators").iter_rows())
    tr_docs: dict[str, list[int]] = {}
    for i, u in enumerate(uids):
        for t in (uid_trs.get(u) or []):
            tr_docs.setdefault(t, []).append(i)
    human = sorted(((t, d) for t, d in tr_docs.items() if t not in MACHINE),
                   key=lambda kv: -len(kv[1]))
    machine = sorted(((t, d) for t, d in tr_docs.items() if t in MACHINE),
                     key=lambda kv: -len(kv[1]))

    trans_subs = [{"h": "By language", "terms": [
        {"t": name, "docs": d} for name, d in by_lang]}]
    if human:
        trans_subs.append({"h": "Human", "terms": [
            {"t": TR_NAMES.get(t, t.title()), "docs": sorted(d)} for t, d in human]})
    if machine:
        trans_subs.append({"h": "Machine", "terms": [
            {"t": TR_NAMES.get(t, t.title()), "docs": sorted(d)} for t, d in machine]})
    tree.append({"h": "Translations", "subs": trans_subs})

    out = {"order": uids, "tree": tree}
    (A / "taxonomy.json").write_text(json.dumps(out, ensure_ascii=False))
    size = (A / "taxonomy.json").stat().st_size / 1e6
    nterms = sum(len(s["terms"]) for n in tree for s in n["subs"])
    print(f"wrote taxonomy.json: {nterms} terms, {size:.2f} MB")


if __name__ == "__main__":
    main()
