# Architecture

How this repo is organized, what each piece does, and how the published site
is built and deployed.

## What the repo ships

The public artifact is a static GitHub Pages site at
`https://cccia-sio.github.io/socal-heat-hub/`. Linked pages:

- `/` — landing page (`index.html`).
- `/main/heat_risk_map.html` — the interactive heat-risk viewer (single self-contained HTML).
- `/userguide/` — plain-language user guide (`userguide/index.html`).

One more page deploys but is **not linked from the landing nav**:

- `/main/hospitalizations.html` — standalone "Heat-attributable hospitalizations
  by ZIP" viewer. (Distinct from the tract-level "Heat hospitalizations" Data
  Layer inside the main viewer.) It ships only because it's tracked under
  `main/`; if it isn't meant to be public, untrack it.

The viewers run entirely in the browser. There is no backend, no API, and no
runtime data fetching beyond the Leaflet basemap tiles and the deployed
PNG / PMTiles overlays.

## What's tracked vs. what's not

`.gitignore` treats `scripts/` and `data/` as **allow-lists**, not blanket
excludes: everything under them is ignored *except* a named set of build scripts
and rendered data products, which are committed and deployed. Raw source data
(CSV / GeoJSON / GeoTIFF inputs) and local-only tooling stay on disk and never
ship.

**Tracked build scripts** (un-ignored via `!scripts/...`):
`build_diy_pca_viewer.py`, `build_lst_overlay.py`, `build_data_overlays.py`,
`build_userguide_pdf.sh`. Everything else in `scripts/` is local-only.

**Tracked data products** (un-ignored via `!data/...`):
`data/exposure/processed/landsat/` (LST overlay PNGs + bounds JSON),
`data/context/sandag_landuse.pmtiles`, and `data/context/overlays/` (Data-Layer
raster PNGs + bounds JSON). All other `data/**` is local-only.

Also tracked at the root: `CLAUDE.md`, `pytest.ini`, `tests/`, plus `docs/`,
`assets/`, and the site pages. **Gitignored local-only:** raw `data/` inputs,
non-allow-listed `scripts/`, `literature/` (the research corpus — see below),
`__pycache__/`, `.venv/`, LaTeX aux files, `.DS_Store`, editor dirs, and
`decks/`.

The GitHub Pages workflow re-applies the same allow-list at deploy time through
its rsync include/exclude rules (belt-and-suspenders — see
`.github/workflows/pages.yml` and the **Deployment** section below).

## Top-level layout

```
socal-heat-hub/
├── index.html                ← landing page
├── README.md                 ← short project blurb + launch link
├── ARCHITECTURE.md           ← this file
├── CLAUDE.md                 ← repo conventions for Claude Code
├── pytest.ini                ← test config
├── .nojekyll                 ← tells GitHub Pages to skip Jekyll processing
├── .gitignore
│
├── main/
│   ├── heat_risk_map.html         ← the viewer (built; single-file HTML)
│   └── hospitalizations.html      ← standalone hospitalizations-by-ZIP viewer (built)
│
├── userguide/
│   ├── index.html            ← user guide page
│   └── figure_*.png          ← figures embedded in the guide
│
├── assets/                   ← logos (viewer top-right card) + comment PDFs
│
├── tests/                    ← (tracked) pytest suite guarding build output & deploy config
│   ├── conftest.py
│   ├── test_build_script.py        ← build script constants / invariants
│   ├── test_generated_html.py      ← generated viewer has expected structure
│   ├── test_landsat_assets.py      ← LST overlay PNGs + bounds present/valid
│   ├── test_pages_workflow.py      ← pages.yml include/exclude rules
│   ├── test_static_links.py        ← internal links resolve
│   └── test_tracked_files.py       ← gitignore allow-list stays correct
│
├── docs/                     ← ancillary docs (deployed, but not linked from nav)
│   ├── userguide/            ← external/draft PDFs + older LaTeX guide
│   ├── pca_overview/         ← LaTeX beamer deck + figures on the PCA method
│   ├── writing_guidelines/   ← AI-writing guardrail notes
│   └── uhi_index_methodology.md
│
├── .github/
│   └── workflows/
│       └── pages.yml         ← deploys to GitHub Pages on push to main
│
├── scripts/                  ← build pipeline (allow-listed scripts tracked; rest local-only)
│   ├── build_diy_pca_viewer.py    ← [tracked] main builder → main/heat_risk_map.html
│   ├── build_lst_overlay.py       ← [tracked] Landsat LST → PNG overlay + bounds JSON
│   ├── build_data_overlays.py     ← [tracked] Data-Layer rasters → data/context/overlays/
│   ├── build_userguide_pdf.sh     ← [tracked] userguide/index.html → docs/userguide PDF
│   └── …                          ← (local-only) slide/figure builders, data fetchers, mockups
│
├── data/                     ← source data + rendered products (only allow-listed products tracked)
│   ├── exposure/
│   │   ├── raw/, processed/…        ← raw inputs (local-only)
│   │   └── processed/landsat/       ← [tracked] LST overlay PNGs + bounds JSON
│   ├── sensitivity/                 ← raw inputs (local-only)
│   ├── adaptive_capacity/           ← raw inputs (local-only)
│   └── context/
│       ├── overlays/                ← [tracked] Data-Layer PNGs + bounds JSON
│       └── sandag_landuse.pmtiles   ← [tracked] SANDAG land-use vector tiles
│
└── literature/               ← (gitignored) markdown RAG corpus — see "Literature corpus"
    ├── INDEX.md  TAXONOMY.md  README.md  build_index.py  refs.bib
    └── sources/  pdfs/  reports/  synthesis/
```

## The viewer build pipeline

The viewer (`main/heat_risk_map.html`) is regenerated by
`scripts/build_diy_pca_viewer.py`. Run from repo root:

```
python3 scripts/build_diy_pca_viewer.py
```

What it does (top-down through the script):

1. **Constants & config** (lines ~28–230)
   - `OUTPUT_PATH = ROOT / "main" / "heat_risk_map.html"`
   - `TRACTS_PATH`, `JURIS_PATH` — census GeoJSON inputs
   - `LOCA2_DIR`, `FUTURE_PERIODS`, `SSPS` — CMIP6 projection settings
   - `INVERTED_VARS` — variables whose risk polarity inverts their pillar default
   - `DELTA_VAR_MAP` — which DIY-PCA variables receive a `tasmax`/`tasmin` delta
   - `VARIABLES` — the canonical list of `(key, label, pillar, subsection)` tuples
   - `VAR_SOURCES` — per-variable source URLs displayed in the legend
2. **`load_data()`** (lines ~334–495) — reads each CSV under `data/`, joins on
   `GEOID` (via the `_safe_merge` helper), imputes NaN with the column median,
   simplifies geometry, returns one GeoDataFrame keyed by tract.
3. **`compute_tract_deltas(tracts)`** (lines ~497–565) — interpolates LOCA2
   ensemble `.nc` deltas to tract centroids; returns
   `{period: {ssp: {var: [delta_per_tract]}}}`.
4. **`main()`** (lines ~567–715) — serializes everything to JSON and splices it
   into the `HTML_TEMPLATE` string at the `__*_PLACEHOLDER__` tokens, then writes
   the result to `main/heat_risk_map.html`. The 15 placeholders group as:
   - tract data & polarity: `__GEOJSON_PLACEHOLDER__`, `__VARMETA_PLACEHOLDER__`,
     `__VARSOURCES_PLACEHOLDER__`, `__POLARITY_PLACEHOLDER__`
   - climate deltas: `__DELTAS_PLACEHOLDER__`, `__DELTAMAP_PLACEHOLDER__`
   - boundary / point overlays: `__JURIS_PLACEHOLDER__`,
     `__UNINCORP_PLACEHOLDER__`, `__TRIBAL_PLACEHOLDER__`,
     `__COOLZONES_PLACEHOLDER__` (county cool-zone sites)
   - raster-overlay bounds (shape matches the LST sidecar): `__LST_PLACEHOLDER__`,
     `__POP_PLACEHOLDER__`, `__IMPERVIOUS_PLACEHOLDER__`, `__CANOPY_PLACEHOLDER__`
   - preset layer bundles: `__HEATMODES_PLACEHOLDER__` (from the `HEAT_MODES` list)
5. **`HTML_TEMPLATE`** (lines ~718–end; the file is ~6,340 lines total) — a single
   triple-quoted string containing the full HTML, CSS, and JavaScript for the
   viewer (Leaflet UI, client-side PCA via Jacobi eigendecomposition, choropleth
   rendering, the compare-two-tracts modal, the legend, etc.).

### Editing the viewer

Because the rendered HTML is committed but is also a build artifact, **edits
must go to both files** to stay coherent:

- Edit the source: `scripts/build_diy_pca_viewer.py` (inside `HTML_TEMPLATE`
  for UI changes, or in the data-loading / variable-definition blocks for
  pipeline changes).
- Rebuild: `python3 scripts/build_diy_pca_viewer.py`
- Commit: the regenerated `main/heat_risk_map.html` (the script itself stays
  gitignored).

If you only edit `main/heat_risk_map.html` directly, the next rebuild will revert
your changes.

## The Landsat LST overlay pipeline

Separate, optional pipeline used by the viewer's compare-mode mini-maps:

```
scripts/build_lst_overlay.py
   ↓  reads ../heat-risk-sd/data/downscaling/annual_lst.tif (sibling repo)
   ↓
data/exposure/processed/landsat/
   ├── annual_lst_4326.png          ← turbo colormap (default)
   ├── annual_lst_4326_cb.png       ← viridis colormap (colorblind-safe)
   └── annual_lst_bounds.json       ← bounds + legend stops
```

`build_diy_pca_viewer.py` reads `annual_lst_bounds.json` at build time and
inlines the bounds + legend into the viewer. The PNGs are fetched at runtime
by the browser (relative paths to `data/exposure/processed/landsat/`).

The relative path means **the LST PNGs are deployed alongside the HTML**, even
though the rest of `data/` is excluded: `pages.yml` explicitly includes
`data/exposure/processed/landsat/**` ahead of the catch-all `data/**` exclude,
so the overlays ship to production. (rsync is "first match wins" — the includes
must precede the exclude; don't reorder them. The same applies to the Data-Layer
overlays and the PMTiles file below.)

## The Data-Layer overlays pipeline

`scripts/build_data_overlays.py` renders the viewer's informational "Data Layers"
context rasters to `data/context/overlays/` as EPSG:4326 RGBA PNGs (default +
colorblind variant) plus a `*_bounds.json` sidecar whose shape matches
`build_lst_overlay.py`'s output, so the viewer consumes them through the same
code path (the `__POP_/__IMPERVIOUS_/__CANOPY_PLACEHOLDER__` bounds + relative
PNG fetch).

```
scripts/build_data_overlays.py
   ↓
data/context/overlays/
   ├── population_4326.png / _cb.png / population_bounds.json   ← EPA EnviroAtlas dasymetric 2020 (log-scaled)
   ├── impervious_4326.png / _cb.png / impervious_bounds.json   ← heat-risk-sd NLCD impervious_100m.tif
   └── canopy_4326.png / _cb.png / canopy_bounds.json           ← heat-risk-sd NLCD treecanopy_100m.tif
```

Population self-downloads from the EPA EnviroAtlas ImageServer (cached locally
under `data/context/population/`, gitignored). Impervious and canopy read the
sibling `heat-risk-sd` repo, located via `$HEAT_RISK_SD_DIR` or fallback
candidate paths. Only the PNG + JSON products are committed and deployed.

## The user guide PDF pipeline

`scripts/build_userguide_pdf.sh` renders `userguide/index.html` to
`docs/userguide/user_guide_external.pdf` using headless Chrome. The HTML
version is the source of truth; the PDF is a distributable snapshot.

There is also an older LaTeX version of the user guide in `docs/userguide/`
(`userguide.tex` → `userguide.pdf`). It is no longer the canonical source,
but is kept for reference.

## Data layout (gitignored)

The `data/` tree mirrors the conceptual structure of the heat-risk model:

```
data/
├── exposure/
│   ├── raw/
│   │   ├── census/
│   │   │   ├── tracts_sd.geojson         ← canonical tract polygons
│   │   │   ├── jurisdictions_sd.geojson  ← city boundaries (optional overlay)
│   │   │   └── acs_demographics_sd.csv   ← used for pct_renter
│   │   └── loca2/
│   │       └── deltas_ensemble/
│   │           ├── ssp245/{tasmax,tasmin}_delta_{2035-2064,2065-2094}.nc
│   │           ├── ssp370/...
│   │           └── ssp585/...
│   ├── processed/
│   │   ├── extreme_heat/
│   │   │   ├── tract_heat_risk_all.csv      ← mean_tmax, p85_tmax, p99_tmax, max_tmax
│   │   │   └── tract_nighttime_heat.csv     ← mean_tmin_overnight, no_relief_fraction
│   │   └── landsat/
│   │       ├── annual_lst_4326.png          ← turbo (built by build_lst_overlay.py)
│   │       ├── annual_lst_4326_cb.png       ← viridis
│   │       └── annual_lst_bounds.json
│   ├── tract_lst_dtr.csv                    ← mean_lst (Landsat tract-mean)
│   └── tract_hrrr_exposure.csv              ← p85_wbgt, p99_wbgt, mean_rh_*, p95_dewpoint
│
├── sensitivity/
│   ├── cdc_places_sd_tracts.csv             ← *_prev (chronic disease prevalence)
│   ├── cdc_svi_2022_sd.csv                  ← EP_*, RPL_THEME*, RPL_THEMES
│   └── hpi/
│       ├── hpi_san_diego_2020tracts.csv     ← HPI domain-level percentiles
│       └── hpi_san_diego.csv                ← HPI indicator-level percentiles
│
├── adaptive_capacity/
│   ├── aircon/
│   │   ├── tract_ac_prevalence_sd_summary.csv  ← ac_prob, cbsa_perc_rank_ac_prob (Romitti)
│   │   └── ahn_etal/Uncertainty_tract/summary_ACprob_tract_2020.csv ← prob_NoAC, prob_Central
│   └── building_age/
│       └── acs_b25034_year_built_sd_tracts.csv ← pct_pre1980, pct_pre1960
│
├── context/
│   ├── overlays/                            ← [tracked, deployed] Data-Layer PNGs + bounds JSON
│   │   ├── population_4326{,_cb}.png + population_bounds.json
│   │   ├── impervious_4326{,_cb}.png + impervious_bounds.json
│   │   └── canopy_4326{,_cb}.png + canopy_bounds.json
│   ├── sandag_landuse.pmtiles               ← [tracked, deployed] SANDAG land-use vector tiles
│   ├── Land_Use_2023_20260313.csv           ← reference data, not currently merged
│   └── tract_exploration.csv                ← pct_green
│
└── cool_zones/
    └── coolzone_sites.csv                   ← county cool-zone sites (COOLZONES overlay)
```

The mapping from CSV columns → variables shown in the viewer is in
`scripts/build_diy_pca_viewer.py` under the `VARIABLES` list and the
`load_data()` function.

## Variable taxonomy (viewer-facing)

`VARIABLES` in the build script defines four columns per row:

| Column     | Meaning                                                           |
|------------|-------------------------------------------------------------------|
| `key`      | Column name in the merged tract dataframe (and JSON property)     |
| `label`    | Human-readable name shown in the UI                               |
| `pillar`   | One of `exposure`, `sensitivity`, `adaptive`                      |
| `sub`      | Subsection grouping inside the pillar (e.g., "Heat", "Health")    |

Per-variable polarity is computed from `(pillar, INVERTED_VARS)`:

- Exposure / sensitivity default: higher value → higher risk → red.
- Adaptive default: higher value → lower risk → green (more capacity protects).
- Anything in `INVERTED_VARS` flips its pillar default (e.g., `pct_pre1980`
  is in the adaptive pillar but is a vulnerability, not a protection).

The polarity dict is serialized into the HTML as `POLARITY` and used by both
the choropleth color scale and the compare-tract modal's red/green coloring.

## Climate scenarios

Variables in `DELTA_VAR_MAP` get the corresponding `tasmax`/`tasmin` LOCA2
ensemble delta added when the user selects a future scenario. Currently
mapped:

- `mean_tmax`, `p85_tmax`, `p99_tmax`, `max_tmax` → `tasmax`
- `mean_tmin_overnight`, `p90_tmin_overnight` → `tasmin`
- `mean_lst`, `p85_wbgt`, `p99_wbgt` → `tasmax` (first-order proxy)

Variables not in this map are held at present-day under all scenarios. The
scenario-adjustment math runs entirely in JavaScript at viewer runtime
(`getAdj(varKey, i)` in the HTML template); the build script just supplies
the per-tract delta tables.

## Tests

A `pytest` suite (`pytest.ini` + `tests/`, both tracked) guards the build output
and deploy config rather than the build internals. Run from repo root:

```
python3 -m pytest
```

- `test_build_script.py` — constants/invariants in `build_diy_pca_viewer.py`.
- `test_generated_html.py` — the committed viewer has the expected structure
  (no leftover `__*_PLACEHOLDER__` tokens, required elements present).
- `test_export_licensing.py` — restricted sources (HPI) never reach the data
  download, and every export carries its citation/license.
- `test_landsat_assets.py` — the LST overlay PNGs + bounds JSON exist and parse.
- `test_pages_workflow.py` — `pages.yml` keeps its include-before-exclude rsync
  ordering (the deploy gotcha above).
- `test_static_links.py` — internal links across the pages resolve.
- `test_tracked_files.py` — the `.gitignore` allow-list still tracks exactly the
  intended scripts/data products.
- `test_tract_pdf.py` — the single-tract PDF export: XML-safe rose-plot labels,
  one page, row-cap overflow reported, imagery attributed, and the SANDAG
  land-use inset dropped (caption and credit with it) when its parcels don't
  render.
- `test_risk_orientation.py` — the rose plot, the detail modal and the tract PDF
  all print RISK percentiles, so protective variables (tree canopy, income, AC)
  are flipped to `100 - pct` everywhere. Each surface owns a copy of this logic
  and they have drifted apart twice.
- `test_composite_staleness.py` — only `compute()` may write the composite.
  Changing a heat mode or a variable marks it stale and highlights Build instead
  of silently rebuilding.

These are the guardrails that catch the kinds of drift this document is prone to.

### Writing new ones

The viewer is one big inlined script, so most guards work by slicing a function
out of the built HTML and asserting on its source. `conftest.py` provides
`js_fn("name")` for that, and `js_code("name")` for the same thing with comments
stripped.

**Use `js_code` for any assertion about what the code does or doesn't contain.**
Fixed code tends to *describe* the bug it fixed, so `assert "_rgbTriplet" not in
src` cheerfully trips over the comment `// _rgbTriplet only reads rgb()`, and
`assert "getImageData" in src` passes even after the real check is deleted. That
mistake was made three times while writing these tests.

And these guards only earn their keep if they fail on the real bug. When adding
one, reintroduce the bug in `main/heat_risk_map.html`, confirm the test goes red,
then restore the file.

## Deployment

`.github/workflows/pages.yml` triggers on push to `main` (and
`workflow_dispatch`):

1. `actions/checkout@v6`
2. `actions/configure-pages@v6`
3. `rsync` the repo into `_site/`, excluding `.git`, `.github`, `scripts`,
   `_site`, `.gitignore`, `README.md`, and `data/**` — but with `--include`
   rules placed *before* the `data/**` exclude that re-admit three subtrees:
   `data/exposure/processed/landsat/**`, `data/context/sandag_landuse.pmtiles`,
   and `data/context/overlays/**`. rsync is **first-match-wins**, so these
   includes must stay ahead of the `data/**` exclude — do not reorder.
4. `actions/upload-pages-artifact@v5` and `actions/deploy-pages@v5`.

Result: GitHub Pages serves `index.html`, `main/`, `userguide/`, `assets/`,
`docs/`, the three allow-listed `data/` subtrees, and `.nojekyll` from the root
of the deployed site. (`scripts/` and all other `data/` are not deployed;
`literature/` is gitignored and never reaches the workflow.)

## Literature corpus (gitignored)

`literature/` is a **local-only research corpus** — a markdown "RAG proxy" for
heat-island / urban-heat-risk / urban-downscaling literature. It is *not* part of
the site, never deploys, and is gitignored in full. Retrieval is plain
`grep` + `read` over structured markdown, driven by frontmatter and an
auto-generated index — no vector database.

```
literature/
├── README.md            ← conventions + how to add a source
├── INDEX.md             ← AUTO-GENERATED retrieval entry point (do not hand-edit)
├── TAXONOMY.md          ← controlled vocab: themes, tags, methods, enums
├── build_index.py       ← rebuilds INDEX.md from source frontmatter (stdlib only)
├── refs.bib             ← Zotero Better BibTeX auto-export (citekey source of truth)
├── sources/             ← one .md per paper (frontmatter + notes); _TEMPLATE.md to start
├── pdfs/                ← open-access PDFs only, fetched programmatically
├── reports/             ← raw deep-research outputs before decomposition
└── synthesis/           ← optional per-theme synthesis notes citing sources/
```

How it works:

- Each source is one `sources/<citekey>.md` with YAML frontmatter (`themes`,
  `tags`, `methods`, `relevance`, `status`, …). Filenames/citekeys mirror the
  Zotero Better BibTeX keys in `refs.bib`.
- `python3 literature/build_index.py` scans every source's frontmatter, groups
  by theme, and regenerates `INDEX.md` (each entry's hook is the source's
  `## TL;DR` first line). Run it after adding or editing sources.
- deep-research output lands in `reports/`, is decomposed into `sources/` stubs
  (`status: to-import`) + a `synthesis/` note, and OA PDFs are pulled into
  `pdfs/`.

See `literature/README.md` for the full workflow.

## Working on this repo

| Task                                  | Where to edit                                  | Then                              |
|---------------------------------------|------------------------------------------------|-----------------------------------|
| Landing page copy / styling           | `index.html`                                   | commit                            |
| User guide content                    | `userguide/index.html`                         | commit (rerun PDF script optional)|
| Viewer UI / map / legend / modals     | `HTML_TEMPLATE` in `scripts/build_diy_pca_viewer.py` | rebuild + commit `main/heat_risk_map.html` |
| Add or remove a viewer variable       | `VARIABLES` + `VAR_SOURCES` (+ source-file load in `load_data()`) | rebuild + commit |
| Change polarity of a variable         | `INVERTED_VARS` in build script                | rebuild + commit                  |
| Update Landsat LST overlay            | run `scripts/build_lst_overlay.py`             | rebuild viewer + commit           |
| Update a Data-Layer overlay           | run `scripts/build_data_overlays.py`           | rebuild viewer + commit           |
| Add a logo to the top-right card      | drop PNG in `assets/`, edit logo block in `HTML_TEMPLATE` | rebuild + commit       |
| Change deployment behavior            | `.github/workflows/pages.yml`                  | commit (rerun `pytest`)           |
| Add / curate a literature source      | `literature/sources/<citekey>.md`              | `python3 literature/build_index.py` (local-only, no commit) |
| Run the test suite                    | n/a                                            | `python3 -m pytest`               |

Local preview: open any HTML file directly in a browser, or run
`python3 -m http.server 8000` from the repo root and visit
`http://localhost:8000/`. Relative paths (`../assets/*.png`,
`../data/exposure/processed/landsat/*.png`) work the same locally and on
GitHub Pages.
