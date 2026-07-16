# buddhomics

A hierarchical explorer of the Pali canon at <https://buddhomics.cherch.org>. The deployed
release is a single four-panel instrument over the whole canon:

- **Taxonomy browser** — a curated tree of ~340 topics (People, Places, Beings, Cosmology,
  Rhetoric, Numbered Lists, Doctrine, Abhidhamma, Vinaya, Narrative, Translations) that
  filters both maps at once; each entry is a set of Pali stem-patterns validated against the corpus.
- **Concept map** — t-SNE of LSA term vectors: every distinctive Pali term placed by the
  company it keeps.
- **Text map** — t-SNE of document vectors from the same SVD space: 6,959 texts placed by
  shared vocabulary, coloured by their traditional grouping (which the layout never sees).
- **Reader + dictionary** — Pali/English by segment with a per-translator multi-select
  (human circles, machine squares), layered NCPED/DPD/DPPN word lookup, and accent-insensitive
  lemma search.

Both maps share one SVD of a Pali TF-IDF matrix and are linked by a term–document incidence
bitmatrix; the full canon now has English (see [Translation](#translation)).

**Roadmap** — the broader "zoom" vision (sound/meter, word collocation, collection-level
bundling, cross-tradition Āgama parallels), the tree-inference-vs-canonical view (the
pipeline's `build_tree.py` is a start), and an embedding-backbone upgrade are tracked as
[issues](https://github.com/chercherai/buddhomics/issues).

## Architecture

Analysis runs locally (Python + uv), producing static JSON/parquet artifacts that are
rsynced to `~/buddhomics.cherch.org` on the shared host (`ssh cherch`). The site is a single
dependency-free HTML/JS page rendering both maps on `<canvas>`; the server is a dumb host.

## Substrate

Everything reads from one aligned table built from
[bilara-data](https://github.com/suttacentral/bilara-data):

```sh
git clone --depth 1 https://github.com/suttacentral/bilara-data data/bilara-data
uv run scripts/build_substrate.py
```

Outputs in `artifacts/` (gitignored):

- `segments.parquet` / `segments.sqlite` — one row per aligned segment (`uid, basket,
  nikaya, subpath, segment_id, seq, pali, english, translator`); `english`/`translator`
  hold the *best-available* rendering (human preferred, machine filling gaps)
- `translations_human.parquet` — long table of **every** human translation
  (`segment_id, translator, english, kind`), not just one per text
- `documents.parquet` — per-text rollup with segment counts, `translated_frac`, and the
  list of available `translators`

`merge_translations.py` then composes the human table with the machine layers into
`translations.parquet` (all sources) and recomputes the best-available backbone — see
Translation below.

Current counts: 7,288 documents, 447,069 segments (sutta 284,708 / vinaya 73,947 /
abhidhamma 88,414). With machine translation folded in, **96.5% of segments** now have
English (up from ~48%); the remainder is the intentional gaps in partial human translations.
104 texts carry more than one human translation; all are preserved and selectable.

## Pipeline

```sh
uv run scripts/merge_translations.py  # compose human + machine translations -> translations.parquet, recompute backbone
uv run scripts/build_features.py   # Pali TF-IDF -> SVD(100) -> UMAP + t-SNE; map.json + doc_svd.npy
uv run scripts/build_tree.py       # collection centroids -> cosine average-linkage -> tree.svg
uv run scripts/build_concepts.py   # LSA term vectors -> t-SNE concept map with numbered-list overlays
uv run scripts/build_taxonomy.py   # curated taxonomy (+ scripts/taxonomy_additions.json) -> per-term doc membership
uv run scripts/build_texts.py      # per-document multi-translation segment JSON for the reader
uv run scripts/build_concept_bits.py  # term-doc incidence bitmatrix linking the two maps
uv run scripts/build_concept_clusters.py  # single-linkage cluster centroids per zoom level (concept map)
uv run scripts/build_text_clusters.py     # same, for the texts map (with collection mix for labelling)
# then label clusters via subagents -> apply labels into {concept,text}_clusters.json
uv run scripts/build_dictionary.py    # NCPED (sc-data) -> site/dictionary.json (reverse-search + prose)
uv run scripts/build_dict_shards.py   # DPD sharded + DPPN names + concept English glosses
uv run scripts/build_pali_index.py    # DPD-lemma Pali full-text search index (sharded)
uv run scripts/build_english_index.py # English-translation search index (vocab + bitmatrix)
cp artifacts/map.json artifacts/concepts.json artifacts/taxonomy.json artifacts/tree.svg site/
rsync -avz site/ cherch:~/buddhomics.cherch.org/
```

The site (`site/`) is dependency-free static HTML/JS, live at
<https://buddhomics.cherch.org>: one four-panel page — taxonomy browser | concept
map | text map | dictionary + reader. Both maps are t-SNE over the shared SVD
space, linked by a full-text term-document bitmatrix. Click a term to see its
texts; click a text to read it; click any Pali word in the reader for an NCPED
definition and its place on the concept map. (UMAP coords remain in map.json;
tree inference still runs in the pipeline via build_tree.py.)

## License

The code in this repository (analysis pipeline + static site) is [MIT](LICENSE)-licensed.
The **texts** are not ours to relicense: the Pali root and human translations come from
[SuttaCentral / bilara-data](https://github.com/suttacentral/bilara-data) under **CC0**;
our machine translations are likewise offered as CC0 (see the fork). Dictionaries (NCPED,
DPD, DPPN) remain under their upstream licenses — see the [about page](https://buddhomics.cherch.org/about.html).

## Translation

The untranslated remainder (Abhidhamma, dense Khuddaka, Vinaya) was machine-translated so
the whole canon is legible and searchable in English. A blind **bake-off**
(`select_passages.py` + `run_bakeoff.py` for Claude, `run_bakeoff_openrouter.py` for the
GPT-5.6 family, scored against Fable-5 reference translations, `build_bakeoff_report.py`)
picked **Claude Fable 5** — a single primary translator keeps a uniform register so a model
seam doesn't fall on basket boundaries.

```sh
uv run scripts/translate_batch.py submit-all       # one Batch job per t-SNE cluster over all untranslated docs
uv run scripts/translate_batch.py poll-all          # ... watch the manifest
uv run scripts/translate_batch.py merge-all         # results -> translations_fable.parquet
uv run scripts/translate_batch.py submit-refused    # retry Fable-refused segments (never human partials)
uv run scripts/translate_fallback.py                # persistent refusals -> GPT-5.6 sol via OpenRouter (smaller chunks)
uv run scripts/merge_translations.py                # fold everything into the substrate
```

Prompt pins: preserve `<b>` lemma markup, keep `…pe…` elisions verbatim, cap segments per
request (32k `max_tokens`). Outcome: **215,610 segments across 2,608 docs** —
2,599 by `claude-fable-5`, 9 by `gpt-5.6-sol` (segments Fable declined; a different vendor
doesn't share the refusal boundary, and 40-segment chunks dodge dense-text truncation).
Machine segments are marked with their model and labelled as unedited renderings in the reader.

`scripts/export_bilara.py` writes them back to bilara/sc-data format (one author dir per
model), published at
[chercherai/bilara-data](https://github.com/chercherai/bilara-data) (branch
`machine-translations`) as a research aid — not published translations.
