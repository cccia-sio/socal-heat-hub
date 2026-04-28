"""Reproject the 100m Landsat annual LST raster to EPSG:4326 and render it as
an RGBA PNG suitable for use as an L.imageOverlay in the viewer's compare-tract
mini-maps.

Inputs
------
  ../heat-risk-sd/data/downscaling/annual_lst.tif   (UTM 11N, 100m, °C)

Outputs
-------
  data/exposure/processed/landsat/annual_lst_4326.png   (RGBA, ocean = transparent)
  data/exposure/processed/landsat/annual_lst_bounds.json (bounds + colormap meta)

The viewer's build_diy_pca_viewer.py reads the JSON and injects bounds + scale
into the HTML; the PNG is fetched directly by the browser at runtime.
"""

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT.parent / "heat-risk-sd" / "data" / "downscaling" / "annual_lst.tif")
OUT_DIR = ROOT / "data" / "exposure" / "processed" / "landsat"
OUT_PNG = OUT_DIR / "annual_lst_4326.png"
OUT_JSON = OUT_DIR / "annual_lst_bounds.json"

# Colormap range — clamp display so the visible spread matches what users care
# about. SD Landsat annual-mean LST clusters around 20–45 °C. Stretch slightly
# beyond to keep extreme tracts on-scale.
VMIN_C = 20.0
VMAX_C = 45.0
# Default colormap + colorblind-friendly alternative (perceptually uniform,
# safe for deuteranopia / protanopia / tritanopia).
COLORMAPS = {"default": "turbo", "cb": "viridis"}


def main():
    if not SRC.exists():
        raise SystemExit(f"Source raster missing: {SRC}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(SRC) as src:
        logger.info(f"Source: {src.shape} {src.crs} res={src.res} bounds={src.bounds}")

        # Reproject UTM → 4326 at native resolution
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
        )
        data_4326 = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data_4326,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        # Bounds (south, west, north, east) of the reprojected raster in lat/lon
        west = transform.c
        north = transform.f
        east = west + transform.a * width
        south = north + transform.e * height
        logger.info(f"4326 bounds: S={south:.4f} W={west:.4f} N={north:.4f} E={east:.4f}")
        logger.info(f"4326 grid: {height}x{width} pixels")

    # Mask invalid (ocean / outside) pixels
    finite = np.isfinite(data_4326)
    logger.info(f"Valid pixels: {finite.sum():,} / {data_4326.size:,}")

    # Render one PNG per palette
    norm = np.clip((data_4326 - VMIN_C) / (VMAX_C - VMIN_C), 0.0, 1.0)
    variants = {}
    for variant, cmap_name in COLORMAPS.items():
        cmap = matplotlib.colormaps[cmap_name]
        rgba = (cmap(norm) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(finite, 220, 0)  # 86% opaque, transparent elsewhere

        suffix = "" if variant == "default" else f"_{variant}"
        png_path = OUT_DIR / f"annual_lst_4326{suffix}.png"
        Image.fromarray(rgba, mode="RGBA").save(png_path, optimize=True)
        size_kb = png_path.stat().st_size // 1024
        logger.info(f"Wrote {png_path} ({size_kb} KB, cmap={cmap_name})")

        legend_stops = []
        for t in np.linspace(0, 1, 5):
            r, g, b, _a = (np.array(cmap(t)) * 255).astype(int)
            legend_stops.append({
                "t": float(t),
                "value": round(float(VMIN_C + t * (VMAX_C - VMIN_C)), 1),
                "rgb": [int(r), int(g), int(b)],
            })
        variants[variant] = {
            "url": f"../data/exposure/processed/landsat/{png_path.name}",
            "colormap": cmap_name,
            "legend": legend_stops,
        }

    meta = {
        # Back-compat: viewer reads top-level url/legend/colormap as default.
        "url": variants["default"]["url"],
        "colormap": variants["default"]["colormap"],
        "legend": variants["default"]["legend"],
        "variants": variants,  # {"default": {...}, "cb": {...}}
        "bounds": {  # Leaflet expects [[south, west], [north, east]]
            "south": float(south), "west": float(west),
            "north": float(north), "east": float(east),
        },
        "vmin_c": VMIN_C, "vmax_c": VMAX_C,
        "source": "Landsat-derived annual mean LST, 100m (USGS/heat-risk-sd)",
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    logger.info(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
