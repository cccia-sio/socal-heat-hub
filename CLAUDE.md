# Project conventions

Static GitHub Pages site for an LA County heat-risk viewer. See `ARCHITECTURE.md` for the full repo map.

## Source of truth: the build script, not the rendered HTML

`main/diy_pca.html` is **generated** by `scripts/build_diy_pca_viewer.py`. Edit the build script, never the HTML.

- The full HTML/CSS/JS lives inside the `HTML_TEMPLATE` string at the bottom of `build_diy_pca_viewer.py`.
- Tract data and computed values are spliced in via `__*_PLACEHOLDER__` tokens (see `ARCHITECTURE.md`).
- Rebuild after every change:
  ```
  python3 scripts/build_diy_pca_viewer.py
  ```
- Commit both the script change and the regenerated `main/diy_pca.html` together.

If you find yourself reaching for `main/diy_pca.html` to edit it directly, stop — the next build will overwrite it.

## Tracked vs. ignored

`scripts/` and `data/` are gitignored except for explicit allow-list entries in `.gitignore`:

- Tracked scripts: `build_diy_pca_viewer.py`, `build_lst_overlay.py`, `build_userguide_pdf.sh`.
- Tracked data: only `data/exposure/processed/landsat/` (the LST PNG overlay + bounds JSON).
- Source CSV/GeoJSON inputs to the build pipeline live under `data/` locally and are **not** committed.

## Deployment

Pushing to `main` triggers `.github/workflows/pages.yml`, which rsyncs the repo into `_site/` and publishes to GitHub Pages.

- The rsync rules exclude `data/**` *except* the landsat subtree. Keep the include/exclude order intact when editing — rsync is "first match wins," so the landsat includes must come before the catch-all `data/**` exclude.
- The viewer fetches `data/exposure/processed/landsat/annual_lst_bounds.json` at runtime; that JSON contains a relative `url:` to the PNG.

## Other build artifacts

- LST overlay (re-run when raw Landsat inputs change):
  ```
  python3 scripts/build_lst_overlay.py
  ```
- User guide PDF (re-run when `userguide/index.html` changes):
  ```
  bash scripts/build_userguide_pdf.sh
  ```

## Do / don't

- **Do** make UI tweaks (CSS, DOM, JS handlers) inside `HTML_TEMPLATE` in the build script.
- **Do** rebuild and commit `main/diy_pca.html` alongside the script change.
- **Don't** edit `main/diy_pca.html` directly.
- **Don't** drop files under `data/` expecting them to deploy unless they're under `data/exposure/processed/landsat/`.
