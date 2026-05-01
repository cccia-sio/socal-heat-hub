# Project conventions

Static GitHub Pages site for an LA County heat-risk viewer. See `ARCHITECTURE.md` for the full repo map.

## `main/diy_pca.html` is generated — never edit it directly

It's built by `scripts/build_diy_pca_viewer.py` from the `HTML_TEMPLATE` string at the bottom of that script, with tract data spliced in via `__*_PLACEHOLDER__` tokens. UI/CSS/JS edits go in the template. Rebuild and commit the regenerated HTML alongside the script change:

```
python3 scripts/build_diy_pca_viewer.py
```

## `scripts/` and `data/` are gitignored except allow-list entries

Tracked scripts: `build_diy_pca_viewer.py`, `build_lst_overlay.py`, `build_userguide_pdf.sh`. Tracked data: only `data/exposure/processed/landsat/`. Source CSV/GeoJSON inputs live under `data/` locally but aren't committed — files dropped elsewhere under `data/` won't deploy.

## Deployment gotcha: rsync rule order

`.github/workflows/pages.yml` excludes `data/**` *except* the landsat subtree. rsync is "first match wins," so the landsat includes must come before the catch-all `data/**` exclude. Don't reorder.

## Other rebuilds

- LST overlay: `python3 scripts/build_lst_overlay.py` (when raw Landsat inputs change)
- User guide PDF: `bash scripts/build_userguide_pdf.sh` (when `userguide/index.html` changes)
