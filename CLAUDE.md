# Project conventions

Static GitHub Pages site for a San Diego County heat-risk viewer. See `ARCHITECTURE.md` for the full repo map.

## `main/heat_risk_map.html` is generated — never edit it directly

It's built by `scripts/build_diy_pca_viewer.py` from the `HTML_TEMPLATE` string at the bottom of that script, with tract data spliced in via `__*_PLACEHOLDER__` tokens. UI/CSS/JS edits go in the template. Rebuild and commit the regenerated HTML alongside the script change:

```
python3 scripts/build_diy_pca_viewer.py
```

## `scripts/` and `data/` are gitignored except allow-list entries

Tracked scripts: `build_diy_pca_viewer.py`, `build_lst_overlay.py`, `build_data_overlays.py`, `build_userguide_pdf.sh`, `build_ucla_heat_tract_xwalk.py`, `build_hero_figure.py`, `build_coolzones.py`, `build_hospitalizations_viewer.py`. Tracked data: `data/exposure/processed/landsat/`, `data/context/sandag_landuse.pmtiles`, and `data/context/overlays/` (Data-Layer raster PNGs + `*_bounds.json`). Source CSV/GeoJSON/raster inputs live under `data/` locally but aren't committed — files dropped elsewhere under `data/` won't deploy.

Tracking a script takes **three** agreeing edits: the `!scripts/...` line in `.gitignore`, `ALLOWED_SCRIPT_FILES` in `tests/test_tracked_files.py`, and an actual `git add`. Miss the middle one and the suite stays green until the commit that makes the file visible to `git ls-files` — which is how `main` went red after `build_coolzones.py` landed.

## The repo is private. The Pages site is not. Tracked ≠ published.

Those are separate settings and it's easy to assume the private repo protects the files. It doesn't — Pages serves the *deployed artifact*, so anything the workflow stages is world-readable at `cccia-sio.github.io/socal-heat-hub/`.

The staging step in `.github/workflows/pages.yml` is an **allow-list**: it names every file that ships (20 of them — the deploy log prints the full list under `--- staged for deploy ---`, so re-derive it there rather than trusting this count). If a path isn't in that list, it isn't public. **Adding a file to this repo must never publish it.** This step used to be a deny-list, which silently published `ARCHITECTURE.md`, `CLAUDE.md`, `tests/`, `docs/`, and a partner's private review PDF.

Tracked ≠ published applies *within* the staged directories too. `data/context/overlays/` and `data/exposure/processed/landsat/` are tracked whole, but the `*_bounds.json` and `downscaled_*` files in them are read at **build** time and inlined into the viewer, so the deploy deliberately does not stage them.

Adding a new runtime asset (an overlay the viewer fetches) takes two edits: the staging step in `pages.yml`, and `MUST_DEPLOY` in `tests/test_pages_workflow.py`. The staging step names files one at a time — never `cp -R` a directory, and never a blanket copy of the repo root. Both are load-bearing: `test_staged_data_matches_what_the_viewer_fetches` compares the staged `data/` paths against the paths the built viewer actually requests, and fails on a mismatch in either direction, so a recursive copy trips it even though the extra files are never named.

## Front-end security invariants

Guarded by `tests/test_frontend_security.py`; all fixes go in `HTML_TEMPLATE`, never in the generated HTML:

- Geocoder output is untrusted. Nominatim is backed by OpenStreetMap, which anyone can edit — escape `display_name` with `_addrEsc()` before it reaches `innerHTML`.
- CDN scripts carry an SRI hash. Compute with `curl -sL <url> | openssl dgst -sha384 -binary | openssl base64 -A`, and check the origin sends CORS or `crossorigin="anonymous"` will break the load.
- No analytics. The README and user guide promise no tracking; keep that true.

## Other rebuilds

- LST overlay: `python3 scripts/build_lst_overlay.py` (when raw Landsat inputs change)
- Data Layer overlays: `python3 scripts/build_data_overlays.py` (population pulled from EPA EnviroAtlas; impervious/canopy from the sibling `heat-risk-sd` NLCD rasters via `$HEAT_RISK_SD_DIR` or a fallback path)
- UCLA excess-ER + CalHeatScore thresholds (tract crosswalk): `python3 scripts/build_ucla_heat_tract_xwalk.py` (fetches the SD slice live from the UCLA Heat Maps ArcGIS service, crosswalks ZIP→tract by centroid → `data/context/ucla_heat_tract.csv`, baked into the viewer's Exploratory Data Layers at build time)
- User guide hero figure: `python3 scripts/build_hero_figure.py` → `userguide/figure_hero_composite.png`. Reproduces the viewer's **default view** (Daytime heat mode + the six `DEFAULT_ON` variables) by porting `exposureScore()`, `pillarPCA()` and `compute()` out of `HTML_TEMPLATE`. If you change that math, change it here too — the script's header documents where the two intentionally differ.
- User guide PDF: `bash scripts/build_userguide_pdf.sh` (when `userguide/index.html` changes)
