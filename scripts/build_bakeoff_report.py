"""Generate the bake-off comparison HTML from the passage/result JSONs."""

import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"

passages = json.loads((A / "bakeoff_passages.json").read_text())
results = json.loads((A / "bakeoff_results.json").read_text())
fable = json.loads((A / "bakeoff_fable.json").read_text())

gpt_path = A / "bakeoff_results_gpt.json"
if gpt_path.exists():
    gpt = json.loads(gpt_path.read_text())
    for uid, models in gpt.items():
        results.setdefault(uid, {}).update(models)

ROWS = [
    ("fable", "Fable 5 (me)"),
    ("claude-opus-4-8", "Opus 4.8"),
    ("openai/gpt-5.6-sol", "GPT-5.6 sol"),
    ("claude-sonnet-5", "Sonnet 5"),
    ("openai/gpt-5.6-terra", "GPT-5.6 terra"),
    ("claude-haiku-4-5", "Haiku 4.5"),
    ("openai/gpt-5.6-luna", "GPT-5.6 luna"),
]

# (uid, seg_id, model) -> annotation. model key as in ROWS.
NOTES = {
    ("bv2", "bv2:14.1", "claude-haiku-4-5"): "Parses kilesamaladhovaṁ (“washing the stain of defilements”, an epithet of the lake) as “defilements and filth exist” — the simile collapses.",
    ("thi-ap21", "thi-ap21:6.2", "claude-haiku-4-5"): "“With these three” — misreads tebhipatthitaṁ (= taṁ te abhipatthitaṁ, “what you aspired to”) as containing “three”.",
    ("thi-ap21", "thi-ap21:8.1", "claude-sonnet-5"): "Plural “heirs/children” — dāyādā and orasā here are feminine singular, describing Bhaddā herself.",
    ("ja531", "ja531:15.3", "claude-haiku-4-5"): "“Serpent-nosed lady” — nāganāsūrū is “she whose thighs are like an elephant’s trunk” (nāga ‘elephant’ + nāsā-ūru); repeated in every refrain, plus me/you person swap.",
    ("ja531", "ja531:19.2", "claude-haiku-4-5"): "“No contentment in grass” — Kusa is the king’s name, not kusa-grass; the whole stanza is lost.",
    ("ja531", "ja531:19.4", "claude-opus-4-8"): "“Toils for wages” inverts anatthike — “having no need of wages”. Sonnet got it right.",
    ("ja531", "ja531:20.2", "claude-haiku-4-5"): "Negation inverted — the line is a threat that she deserves to have her tongue cut out.",
    ("ja531", "ja531:21.1", "claude-haiku-4-5"): "Garbled — the hunchback is telling Pabhāvatī not to measure Kusa by his looks; not about “boasting of your beauty”.",
    ("pv36", "pv36:8.2", "claude-sonnet-5"): "“Seven peaks” — sattussada is satta ‘beings’ + ussada, “crowded with beings”, not satta ‘seven’.",
    ("pv36", "pv36:9.2", "claude-haiku-4-5"): "Misses the point of the whole stanza: the stake is better by far than that hell.",
    ("pv36", "pv36:11.1", "claude-haiku-4-5"): "“Beyond knowledge” — aññāto means “understood”, misread as a-ññāto “unknown”.",
    ("pv36", "pv36:12.2", "claude-sonnet-5"): "Garbles “there is no disclosing to one without trust” (nācikkhanā appasannassa hoti).",
    ("pv36", "pv36:12.3", "claude-sonnet-5"): "“Made my word trustworthy” — it is the asker’s word being deemed worthy of belief.",
    ("mil3.1.13", "mil3.1.13:1.2", "claude-haiku-4-5"): "Haiku returned only 4 of 14 segments on this passage — the rest are missing.",
    ("cnd12", "cnd12:8.1", "claude-haiku-4-5"): "Haiku returned only 1 of 20 segments (and this one empty) — near-total failure on this passage.",
    ("cnd12", "cnd12:8.3", "claude-opus-4-8"): "“Before the teaching of Gotama” — huraṁ means “outside / apart from”, not temporally before.",
    ("cnd12", "cnd12:9.1", "claude-sonnet-5"): "Sonnet alone preserved the <b>…</b> lemma markup — useful for our pipeline.",
    ("kv1.2", "kv1.2:5.1", "claude-haiku-4-5"): "“Chieftain … [servants]” — seṭṭhī is a banker/treasurer and the 400,000 is money; “servants” is hallucinated.",
    ("kv1.2", "kv1.2:7.0", "claude-haiku-4-5"): "saṁsandana = “comparison” (across the noble persons), not “association”.",
    ("ya1.2.1", "ya1.2.1:15.1", "claude-haiku-4-5"): "aññamañña (“mutually”) rendered “non-mutual” — a logical inversion repeated through the whole passage.",
    ("ya1.2.1", "ya1.2.1:24.1", "claude-sonnet-5"): "Identical to its 15.1 rendering although the Pali differs (mūlamūla vs mūlaka) — the distinction the Yamaka section structure is built on is flattened.",
    ("patthana1.11", "patthana1.11:15.1", "claude-haiku-4-5"): "upādārūpa (“derived form”) confused with upacaya (“accumulation”); in 18.1 kaṭattārūpa (kamma-born form) is also mis-rendered.",
    ("patthana1.11", "patthana1.11:14.1", "claude-haiku-4-5"): "ācayagāmī/apacayagāmī as “ascending/declining” loses the doctrinal sense (leading to accumulation / dismantling of rebirth).",
    ("ps1.2", "ps1.2:10.13", "claude-haiku-4-5"): "kasiṇa flattened to “meditation object” — the technical term disappears from the corpus.",
    # ---- GPT-5.6 round ----
    ("bv2", "bv2:14.1", "openai/gpt-5.6-terra"): "“One soiled by the filth of defilements” — kilesamaladhovaṁ is “washing the stain of defilements”, an epithet of the lake (dhova ‘washing’, not ‘soiled’).",
    ("thi-ap21", "thi-ap21:7.3", "openai/gpt-5.6-terra"): "Content redistributed across segments 7.3–7.4 (“a teacher in the world named Gotama / by clan, will arise”) — breaks segment alignment, which matters for our pipeline.",
    ("thi-ap21", "thi-ap21:8.1", "openai/gpt-5.6-sol"): "Shifts the prophecy to second person (“You will be…”) — hessati is third person; contextually natural but changes the grammar.",
    ("ja531", "ja531:15.3", "openai/gpt-5.6-sol"): "All three GPT tiers parsed nāganāsūrū correctly (“thighs like an elephant’s trunk”) — where Haiku 4.5 produced “serpent-nosed”.",
    ("ja531", "ja531:19.2", "openai/gpt-5.6-luna"): "Segments 19.1–19.4 redistributed and garbled (“she supports a cook … yet has no interest in wages”) — alignment and sense both suffer.",
    ("ja531", "ja531:20.2", "openai/gpt-5.6-luna"): "“Does not deserve to have her tongue cut out … ?” — inverts the threat, like Haiku.",
    ("ja531", "ja531:21.4", "openai/gpt-5.6-luna"): "“Make yourself pleasing to your lovely beloved” — karassu rucire piyaṁ is “hold him dear, O lovely one”.",
    ("pv36", "pv36:8.2", "openai/gpt-5.6-sol"): "All three GPT tiers read sattussada as “seven-fold” — same choice as Sonnet. Correction to my earlier note: the ‘seven ussada (subsidiary hells)’ parse is attested in the tradition alongside “crowded with beings” (sattehi ussanno), so this is a debatable reading, not a plain error.",
    ("pv36", "pv36:9.4", "claude-opus-4-8"): "Revision against my own reference: the relative reading “into which he would fall” (Opus, Sonnet, all GPT tiers) is smoother than my carry-over of the prohibitive mā from 7.4. Score one against the referee.",
    ("pv36", "pv36:12.3", "openai/gpt-5.6-sol"): "Speaker flipped — saddheyyavaco is the asker’s word being deemed trustworthy, not “you would trust my words”. Terra and Luna make the same flip.",
    ("mil3.1.13", "mil3.1.13:1.3", "openai/gpt-5.6-sol"): "All three GPT tiers render apilāpana with the traditional negative etymology (“not losing track / not forgetting”); modern scholarship (and the treasurer simile itself, sarāpeti “reminds”) favors “calling to mind”. Terra has to contort apilāpeti into “prevents…from being forgotten” to sustain it. Defensible, not an error.",
    ("ps1.2", "ps1.2:10.13", "openai/gpt-5.6-sol"): "Expands every elided segment to the full three-clause formula — doctrinally correct but interpolates text the segment doesn’t contain; misalignment risk for our corpus.",
    ("cnd12", "cnd12:8.3", "openai/gpt-5.6-sol"): "“Before Gotama’s teaching” — same huraṁ (“outside/apart from”) slip as Opus; Terra too. Luna alone among the GPT tiers got “outside”.",
    ("ya1.2.1", "ya1.2.1:15.1", "openai/gpt-5.6-sol"): "Sol preserves the mūlamūla/mūlaka distinction between sections (“roots” vs “based”) — better than Sonnet and Terra, which flatten it.",
    ("ya1.2.1", "ya1.2.1:24.1", "openai/gpt-5.6-terra"): "Near-identical to its 15.1 rendering — the mūlamūla/mūlaka distinction is flattened, as with Sonnet.",
    ("patthana1.11", "patthana1.11:14.1", "openai/gpt-5.6-luna"): "Condition-type mismatch: “depending on an accumulation-going phenomenon, a decline-going phenomenon arises” — the Pali has apacayagāmī on both sides. Exactly the kind of error that poisons downstream analysis.",
    ("patthana1.11", "patthana1.11:18.1", "openai/gpt-5.6-luna"): "purejāta (“prenascence”) rendered “proximity-born” — wrong gloss.",
}

VERDICTS = [
    ("bv2", "Buddhavaṁsa verse", "Clean", "Clean", "Simile at 14.1 collapses", "Clean", "14.1 misparse (“soiled”)", "Clean"),
    ("thi-ap21", "Therī-apadāna verse", "Clean", "Number slip at 8.1", "“Three” hallucinated at 6.2", "Person shift at 8.1", "7.3–7.4 realigned", "Clean"),
    ("ja531", "Jātaka archaic verse", "One inversion (19.4)", "Clean — best here", "Unusable (names, persons, negation)", "Clean", "Clean", "19–21 garbled, inversion"),
    ("pv36", "Petavatthu dialogue", "Clean — best here", "Two slips (12.2–3)", "Unusable", "12.3 speaker flip", "12.3 speaker flip", "12.2–3 wrong"),
    ("mil3.1.13", "Milindapañha prose", "Clean", "Clean", "Incomplete: 4/14 segments", "Traditional apilāpana", "apilāpeti contorted", "Traditional apilāpana"),
    ("ne10", "Netti technical", "Clean", "Clean", "Serviceable", "Clean", "Clean", "Clean"),
    ("pe3", "Peṭakopadesa", "Clean", "Clean", "dosa → “harm” (should be hate)", "Clean", "Clean", "Clean"),
    ("ps1.2", "Paṭisambhidāmagga lists", "Clean", "Clean (expands elisions)", "kasiṇa term lost", "Over-expands elisions", "Clean", "Clean"),
    ("cnd12", "Cūḷaniddesa gloss", "huraṁ slip at 8.3", "Clean + keeps markup", "Incomplete: 1/20 segments", "huraṁ slip", "huraṁ slip", "Clean on huraṁ"),
    ("kv1.2", "Kathāvatthu dialectic", "Clean", "Clean", "seṭṭhī simile hallucinated", "Clean (modal shift)", "Clean", "Clean"),
    ("ya1.2.1", "Yamaka logic", "Clean — best here", "mūlamūla/mūlaka flattened", "“Mutual” → “non-mutual” inversion", "Distinction preserved", "Distinction flattened", "Muddled but no inversion"),
    ("patthana1.11", "Paṭṭhāna conditions", "Clean", "Clean", "Term confusions (upādā/upacaya)", "Clean", "“Increase/decrease” weak", "Condition-type mismatch at 14.1"),
]


def esc(s):
    return html.escape(s, quote=False)


parts = []
parts.append("""<title>Pali Translation Bake-off</title>
<style>
:root{
  --bg:#FAFAF7; --ink:#212633; --muted:#6B7080; --pali:#8E2A18; --rule:#DDDCD4;
  --card:#F1F0EA; --flag:#8E2A18; --flagbg:#F6E9E5; --good:#3D6B4F; --chip:#E7E6DE;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#15171D; --ink:#DEDCD2; --muted:#8B8F9C; --pali:#C9A45C; --rule:#2C2F38;
  --card:#1C1F27; --flag:#D98873; --flagbg:#2A1E1B; --good:#8FBE9F; --chip:#262933;
}}
:root[data-theme="dark"]{
  --bg:#15171D; --ink:#DEDCD2; --muted:#8B8F9C; --pali:#C9A45C; --rule:#2C2F38;
  --card:#1C1F27; --flag:#D98873; --flagbg:#2A1E1B; --good:#8FBE9F; --chip:#262933;
}
:root[data-theme="light"]{
  --bg:#FAFAF7; --ink:#212633; --muted:#6B7080; --pali:#8E2A18; --rule:#DDDCD4;
  --card:#F1F0EA; --flag:#8E2A18; --flagbg:#F6E9E5; --good:#3D6B4F; --chip:#E7E6DE;
}
body{background:var(--bg); color:var(--ink);
  font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  line-height:1.55; margin:0; padding:2.5rem 1.25rem 5rem;}
main{max-width:56rem; margin:0 auto;}
h1{font-size:2rem; font-weight:600; letter-spacing:-0.01em; text-wrap:balance; margin:0 0 .3rem;}
h2{font-size:1.3rem; font-weight:600; margin:3rem 0 .8rem; text-wrap:balance;}
.sub{color:var(--muted); margin:0 0 2rem; max-width:44rem;}
.eyebrow{font-family:system-ui,sans-serif; font-size:.7rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted);}
table{border-collapse:collapse; width:100%; font-size:.86rem;
  font-family:system-ui,sans-serif;}
.tablewrap{overflow-x:auto;}
th{text-align:left; font-weight:600; font-size:.7rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); padding:.45rem .7rem;
  border-bottom:1px solid var(--rule);}
td{padding:.45rem .7rem; border-bottom:1px solid var(--rule); vertical-align:top;}
td.num{font-variant-numeric:tabular-nums; white-space:nowrap;}
.passage{margin-top:2.6rem; border-top:2px solid var(--rule); padding-top:1.2rem;}
.seg{margin:1.15rem 0 0; padding:0 0 1rem; border-bottom:1px solid var(--rule);}
.segid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.7rem;
  color:var(--muted);}
.pali{color:var(--pali); font-style:italic; font-size:1.02rem; margin:.15rem 0 .5rem;}
.tr{display:grid; grid-template-columns:6.2rem 1fr; gap:.15rem .8rem; margin:.22rem 0;}
.who{font-family:system-ui,sans-serif; font-size:.68rem; letter-spacing:.07em;
  text-transform:uppercase; color:var(--muted); padding-top:.18rem; white-space:nowrap;}
.note{grid-column:2; font-family:system-ui,sans-serif; font-size:.78rem;
  color:var(--flag); background:var(--flagbg); border-radius:4px;
  padding:.3rem .55rem; margin:.15rem 0 .2rem;}
.note.good{color:var(--good); background:var(--chip);}
.missing{color:var(--muted); font-style:italic;}
.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));
  gap:.8rem; margin:1.2rem 0;}
.card{background:var(--card); border-radius:6px; padding:.9rem 1rem;}
.card h3{margin:0 0 .3rem; font-size:1rem;}
.card p{margin:0; font-size:.85rem; font-family:system-ui,sans-serif; color:var(--muted);}
.card .big{font-size:1.5rem; font-variant-numeric:tabular-nums; color:var(--ink);
  font-family:system-ui,sans-serif; font-weight:600;}
p{max-width:44rem;}
</style>
<main>
<p class="eyebrow">buddhomics · translation bake-off · 2026-07-14</p>
<h1>Twelve hard passages, four translators</h1>
<p class="sub">The most complex untranslated Pali in bilara-data — archaic Jātaka verse,
Kathāvatthu dialectic, Yamaka logic, Paṭṭhāna formulas — run through Opus 4.8, Sonnet 5,
Haiku 4.5, and the GPT-5.6 family (sol / terra / luna via OpenRouter) with identical
prompts and segment-aligned structured output, compared against my own translations done
directly from the Pali before reading any model output. Pali lines in red; flagged
readings annotated inline, including two places where the GPT round forced a revision
of my earlier judgments.</p>
""")

# cost cards
parts.append("""
<div class="cards">
<div class="card"><h3>Opus 4.8</h3><p class="big">$0.40</p>
<p>Complete. Two slips across 330 segments. Best on Yamaka logic and Petavatthu. Still the overall winner.</p></div>
<div class="card"><h3>GPT-5.6 sol</h3><p class="big">$0.72</p>
<p>Complete. Roughly Sonnet-tier accuracy, preserved the Yamaka distinctions, cleanest GPT tier — and the most expensive model tested.</p></div>
<div class="card"><h3>Sonnet 5</h3><p class="big">$0.44</p>
<p>Complete. Reads beautifully; a handful of real errors; flattened one structural distinction. Kept lemma markup (alone in the field).</p></div>
<div class="card"><h3>GPT-5.6 terra</h3><p class="big">$0.22</p>
<p>Complete. Solid mid-tier; one misparse, one segment-realignment, Yamaka distinction flattened.</p></div>
<div class="card"><h3>GPT-5.6 luna</h3><p class="big">$0.10</p>
<p>Complete — no dropped segments, and it parsed compounds Haiku mangled. But a Paṭṭhāna condition-type mismatch and Jātaka garbles rule it out.</p></div>
<div class="card"><h3>Haiku 4.5</h3><p class="big">$0.05</p>
<p>Dropped 29 segments on the two long-segment passages; logical inversions and name/compound misreadings throughout.</p></div>
</div>
""")

# verdict table
parts.append('<h2>Per-passage verdicts</h2><div class="tablewrap"><table><tr><th>Passage</th><th>Opus 4.8</th><th>Sonnet 5</th><th>Haiku 4.5</th><th>GPT-5.6 sol</th><th>GPT-5.6 terra</th><th>GPT-5.6 luna</th></tr>')
for uid, label, o, s, h, gs, gt, gl in VERDICTS:
    parts.append(f"<tr><td><b>{esc(label)}</b><br><span class='segid'>{uid}</span></td>"
                 f"<td>{esc(o)}</td><td>{esc(s)}</td><td>{esc(h)}</td>"
                 f"<td>{esc(gs)}</td><td>{esc(gt)}</td><td>{esc(gl)}</td></tr>")
parts.append("</table></div>")

parts.append("<h2>Side-by-side, all segments</h2>")
for p in passages:
    uid = p["uid"]
    parts.append(f'<section class="passage"><p class="eyebrow">{esc(uid)}</p>'
                 f'<h2 style="margin-top:.2rem">{esc(p["label"])}</h2>')
    for s in p["segments"]:
        sid = s["id"]
        parts.append(f'<div class="seg"><span class="segid">{esc(sid)}</span>'
                     f'<p class="pali">{esc(s["pali"].strip())}</p>')
        for key, name in ROWS:
            if key == "fable":
                text = fable[uid].get(sid)
            else:
                text = results[uid][key].get("segments", {}).get(sid)
            body = esc(text.strip()) if text and text.strip() else '<span class="missing">— missing —</span>'
            parts.append(f'<div class="tr"><span class="who">{esc(name)}</span><span>{body}</span>')
            note = NOTES.get((uid, sid, key))
            if note:
                positive = note.startswith(("Sonnet alone", "All three GPT tiers parsed", "Sol preserves"))
                cls = "note good" if positive else "note"
                parts.append(f'<span class="{cls}">{esc(note)}</span>')
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</section>")

parts.append("</main>")
out = A / "bakeoff_report.html"
out.write_text("\n".join(parts))
print(f"wrote {out} ({out.stat().st_size//1024} KB)")
