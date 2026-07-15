# buddhomics

A hierarchical explorer of the Pali canon at <https://buddhomics.cherch.org>. Each level is a
"zoom" with its own feature signal and visualization:

| Level | Unit | Feature signal | Viz |
|---|---|---|---|
| Sound | Pali syllables | phonotactics, meter, assonance | frequency spectra, meter maps |
| Word | words/lemmas | TF-IDF, collocations, rare terms | word clouds, collocation graphs |
| Segment | aligned lines | Pali↔English vectors | drill-in reader |
| Discourse | suttas | document embeddings | tSNE/UMAP scatter |
| Collection | vagga → saṃyutta → nikāya | centroid drift, dispersion | hierarchical bundling |
| Basket / canon | whole structure | branching tree | dendrogram |
| Cross-tradition | Pali ↔ Chinese Āgama parallels | shared vs divergent segments | alignment ribbons |

Two ML backbones: (1) tSNE/UMAP over discourse-level vectors (TF-IDF first, multilingual
sentence embeddings later) to see clusters that may or may not respect nikāya boundaries;
(2) distance-based tree inference over discourses/collections, compared against the
canonical hierarchy — the contrast is the story.

## Architecture

Analysis runs locally (Python + uv), producing static JSON/parquet artifacts that are
rsynced to `~/buddhomics.cherch.org` on the shared host (`ssh cherch`). The site is static
with client-side rendering (d3 / deck.gl); the server is a dumb host.

## Substrate

Everything reads from one aligned table built from
[bilara-data](https://github.com/suttacentral/bilara-data):

```sh
git clone --depth 1 https://github.com/suttacentral/bilara-data data/bilara-data
uv run scripts/build_substrate.py
```

Outputs in `artifacts/` (gitignored):

- `segments.parquet` / `segments.sqlite` — one row per aligned segment:
  `uid, basket, nikaya, subpath, segment_id, seq, pali, english, translator`
- `documents.parquet` (also a table in the sqlite) — per-text rollup with segment counts
  and `translated_frac`

Current counts: 7,288 documents, 447,069 segments (sutta 284,708 / vinaya 73,947 /
abhidhamma 88,414). English is sujato-first, brahmali fallback, then any translator.
KN is only ~33% translated; DN/MN/SN/AN are ~93–98%.

## Pipeline

```sh
uv run scripts/build_features.py   # Pali TF-IDF -> SVD(100) -> UMAP + t-SNE; map.json + doc_svd.npy
uv run scripts/build_tree.py       # collection centroids -> cosine average-linkage -> tree.svg
uv run scripts/build_concepts.py   # LSA term vectors -> t-SNE concept map with numbered-list overlays
uv run scripts/build_taxonomy.py   # curated People/Beings/Places/Lists -> per-term doc membership
uv run scripts/build_texts.py      # per-document segment JSON for the reading panel
uv run scripts/build_concept_bits.py  # term-doc incidence bitmatrix linking the two maps
uv run scripts/build_dictionary.py    # NCPED (sc-data) -> site/dictionary.json
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

## Translation bake-off (2026-07-14)

`scripts/select_passages.py` + `run_bakeoff.py` (Claude models) + `run_bakeoff_openrouter.py`
(GPT-5.6 family) translate 12 hard untranslated passages, compared against Fable 5
reference translations in `artifacts/bakeoff_fable.json`. Report generator:
`build_bakeoff_report.py`. Verdict: Fable 5 ≥ Opus 4.8 > GPT-5.6 sol ≈ Sonnet 5 >
GPT-5.6 terra > GPT-5.6 luna ≫ Haiku 4.5. Plan: translate the untranslated canon
with `claude-fable-5` via the Batch API (~$190), Opus 4.8 fallback on refusal;
prompt must pin markup preservation, `…pe…` elisions, and per-request segment caps.
