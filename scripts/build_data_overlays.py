"""Build the informational "Data Layers" raster overlays for the viewer.

Renders three context rasters to data/context/overlays/ as EPSG:4326 RGBA PNGs
(default + colorblind variant) plus a *_bounds.json sidecar whose shape matches
build_lst_overlay.py's annual_lst_bounds.json, so the viewer consumes them with
the same code path:

  population  reproducible pull from EPA EnviroAtlas "Dasymetric_2020" ImageServer
              (2020 dasymetric residential population, 30 m), log-scaled
  impervious  heat-risk-sd NLCD impervious_100m.tif  (0-100 %, linear)
  canopy      heat-risk-sd NLCD treecanopy_100m.tif  (0-100 %, linear)

Population is self-contained (downloads from EPA). Impervious/canopy read the
sibling heat-risk-sd repo, located via $HEAT_RISK_SD_DIR or a few candidate
paths. Outputs (PNG + JSON) are the only committed/deployed artifacts; the
downloaded population GeoTIFF is cached locally under data/context/population/.

Rebuild with:  python3 scripts/build_data_overlays.py
"""

import json
import logging
import os
import socket
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "context" / "overlays"          # committed/deployed PNG+JSON
POP_CACHE = ROOT / "data" / "context" / "population" / "sd_dasymetric_pop_2020_30m_utm11n.tif"
TRACTS_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "tracts_sd.geojson"
URL_REL = "../data/context/overlays"                       # PNG url relative to main/heat_risk_map.html

# EPA EnviroAtlas 2020 dasymetric population ImageServer (exportImage clips by bbox)
EPA_EXPORT = ("https://enviroatlas.epa.gov/arcgis/rest/services/Rasters/"
              "Dasymetric_2020/ImageServer/exportImage")
# San Diego County bbox (WGS84), padded a hair past the county edges
SD_BBOX_4326 = (-117.62, 32.50, -116.05, 33.52)

LEGEND_STOPS = [0.0, 0.25, 0.5, 0.75, 1.0]


def resolve_heat_risk_sd():
    """Locate the sibling heat-risk-sd repo (source of the NLCD rasters)."""
    cands = [
        os.environ.get("HEAT_RISK_SD_DIR"),
        ROOT.parent / "heat-risk-sd",
        ROOT.parent.parent / "heat-risk-sd",
        Path.home() / "Documents" / "GitHub" / "heat-risk-sd",
    ]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def load_land_geoms():
    """SD County land outline (EPSG:4326) as a one-element geometry list.

    Overlays are county-rectangle rasters, so they can carry offshore noise (the
    NLCD canopy 'ocean tract' off Camp Pendleton) and out-of-county spillover.
    Masking to this outline forces everything outside the county transparent.
    It's the union of tracts with ALAND >= 100k m^2 — the same land-tract filter
    the viewer uses — so the clip matches the analysis footprint. unary_union
    dissolves the internal tract borders so the mask has no interior gaps.
    """
    gdf = gpd.read_file(TRACTS_PATH)
    gdf["ALAND"] = gdf["ALAND"].astype(float)
    gdf = gdf[gdf["ALAND"] >= 100_000].to_crs(4326)
    return [gdf.union_all()]


def download_population_clip(out_tif):
    """Tile-download the SD dasymetric population grid (UTM 11N, 30 m) from EPA.

    Tiled because a single full-county exportImage reprojection times out
    server-side; tiles are placed on a shared 30 m grid so they reassemble with
    no resampling (population counts stay faithful).
    """
    if out_tif.exists():
        logger.info(f"population clip cached: {out_tif.relative_to(ROOT)}")
        return
    socket.setdefaulttimeout(120)
    res = 30.0
    W, S, E, N = SD_BBOX_4326
    tf = Transformer.from_crs(4326, 32611, always_xy=True)
    xs, ys = zip(*[tf.transform(lon, lat) for lon in (W, E) for lat in (S, N)])
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    w = round((xmax - xmin) / res)
    h = round((ymax - ymin) / res)
    ytop = ymin + h * res
    A = np.full((h, w), np.nan, dtype=np.float32)

    def fetch(xL, yB, xR, yT, tw, th):
        url = (f"{EPA_EXPORT}?bbox={xL},{yB},{xR},{yT}&bboxSR=32611&imageSR=32611"
               f"&size={tw},{th}&format=tiff&pixelType=F32"
               f"&interpolation=RSP_NearestNeighbor&f=image")
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(url).read()
                with MemoryFile(data) as mf, mf.open() as ds:
                    return ds.read(1).astype(np.float32)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)

    tile = 1500
    logger.info(f"downloading population grid {w}x{h} px (UTM 11N, 30 m) in tiles...")
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            c1, r1 = min(c0 + tile, w), min(r0 + tile, h)
            A[r0:r1, c0:c1] = fetch(xmin + c0 * res, ytop - r1 * res,
                                    xmin + c1 * res, ytop - r0 * res, c1 - c0, r1 - r0)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", dtype="float32", count=1, width=w, height=h,
                crs="EPSG:32611", transform=from_origin(xmin, ytop, res, res),
                nodata=-9999.0, compress="deflate", tiled=True)
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(np.where(np.isfinite(A), A, -9999.0).astype(np.float32), 1)
    logger.info(f"wrote {out_tif.relative_to(ROOT)}")


def render_overlay(src_tif, *, name, label, units, scale, cmap_default, cmap_cb,
                   source, land_geoms=None, vmin=0.0, vmax=None, max_width=1600):
    """Reproject a UTM raster to 4326 and write default+cb PNGs + a bounds JSON.

    scale='linear' clips to [vmin, vmax]; scale='log' uses log1p (for the very
    skewed population grid). Zero / nodata pixels render fully transparent so the
    layer "lights up" only where the feature is present; if land_geoms is given,
    pixels outside that outline (ocean / out-of-county) are forced transparent
    too. Display width is capped at max_width (county-scale overlay; the
    analysis-grade GeoTIFF is untouched).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_tif) as src:
        arr = src.read(1).astype(np.float32)
        nod = src.nodata
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        if width > max_width:  # downsample for a lightweight, fast-loading overlay
            west0, north0 = transform.c, transform.f
            east0, south0 = west0 + width * transform.a, north0 + height * transform.e
            scl = max_width / width
            width, height = max_width, max(1, round(height * scl))
            transform = rasterio.transform.from_bounds(west0, south0, east0, north0, width, height)
        data = np.full((height, width), np.nan, dtype=np.float32)
        reproject(arr, data, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=dst_crs,
                  src_nodata=nod, dst_nodata=np.nan, resampling=Resampling.average)

    if land_geoms is not None:
        land = geometry_mask(land_geoms, (height, width), transform,
                             invert=True, all_touched=True)
    else:
        land = True
    present = np.isfinite(data) & (data > 0) & land
    if vmax is None:  # auto: 99th percentile of present pixels
        vmax = float(np.percentile(data[present], 99)) if present.any() else 1.0

    if scale == "log":
        disp = np.log1p(np.clip(np.nan_to_num(data, nan=0.0), 0, None))
        dmin, dmax = np.log1p(vmin), np.log1p(vmax)
        inv = lambda t: float(np.expm1(dmin + t * (dmax - dmin)))
    else:
        disp = np.clip(np.nan_to_num(data, nan=vmin), vmin, vmax)
        dmin, dmax = float(vmin), float(vmax)
        inv = lambda t: float(dmin + t * (dmax - dmin))
    norm = np.clip((disp - dmin) / (dmax - dmin + 1e-9), 0, 1)
    alpha = np.where(present, np.clip(45 + norm * 210, 45, 255), 0).astype(np.uint8)

    variants = {}
    for vk, cmname in (("default", cmap_default), ("cb", cmap_cb)):
        cmap = matplotlib.colormaps[cmname]
        rgba = (cmap(norm) * 255).astype(np.uint8)
        rgba[..., 3] = alpha
        suffix = "" if vk == "default" else "_cb"
        Image.fromarray(rgba, "RGBA").save(OUT_DIR / f"{name}_4326{suffix}.png")
        legend = [{"t": t, "value": round(inv(t), 2),
                   "rgb": [int(c) for c in (np.array(cmap(t)[:3]) * 255).round()]}
                  for t in LEGEND_STOPS]
        variants[vk] = {"url": f"{URL_REL}/{name}_4326{suffix}.png",
                        "colormap": cmname, "legend": legend}

    west, north = transform.c, transform.f
    east = west + width * transform.a
    south = north + height * transform.e
    meta = {**variants["default"], "variants": variants,
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "vmin": round(float(vmin), 3), "vmax": round(float(vmax), 3),
            "scale": scale, "label": label, "units": units, "source": source}
    (OUT_DIR / f"{name}_bounds.json").write_text(json.dumps(meta, indent=2))

    npx = int(present.sum())
    logger.info(f"{name}: {width}x{height} 4326 px, {npx:,} populated px, "
                f"vmax={vmax:.1f}, scale={scale} -> {name}_4326.png (+cb)")
    return meta


def build_population(land_geoms):
    download_population_clip(POP_CACHE)
    with rasterio.open(POP_CACHE) as ds:
        a = ds.read(1, masked=True)
        logger.info(f"population total in clip: {float(a[a > 0].sum()):,.0f} people")
    return render_overlay(
        POP_CACHE, name="population",
        label="Population density", units="people per pixel",
        scale="log", cmap_default="inferno", cmap_cb="cividis",
        source="EPA EnviroAtlas 2020 Dasymetric Population (30 m)",
        land_geoms=land_geoms)


def build_impervious(hrs, land_geoms):
    src = hrs / "data" / "context" / "nlcd" / "impervious_100m.tif"
    if not src.exists():
        logger.warning(f"impervious source missing: {src} — skipping")
        return None
    return render_overlay(
        src, name="impervious",
        label="Impervious surface", units="% impervious",
        scale="linear", vmin=0, vmax=100, cmap_default="Purples", cmap_cb="cividis",
        source="NLCD impervious surface, 100 m (heat-risk-sd)",
        land_geoms=land_geoms)


def build_canopy(hrs, land_geoms):
    src = hrs / "data" / "context" / "nlcd" / "treecanopy_100m.tif"
    if not src.exists():
        logger.warning(f"canopy source missing: {src} — skipping")
        return None
    # clip display at 60 % — SD canopy rarely exceeds it, so this keeps contrast.
    return render_overlay(
        src, name="canopy",
        label="Tree canopy", units="% canopy",
        scale="linear", vmin=0, vmax=60, cmap_default="Greens", cmap_cb="cividis",
        source="NLCD tree canopy cover, 100 m (heat-risk-sd)",
        land_geoms=land_geoms)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    land_geoms = load_land_geoms()
    build_population(land_geoms)
    hrs = resolve_heat_risk_sd()
    if hrs is None:
        logger.warning("heat-risk-sd not found (set $HEAT_RISK_SD_DIR) — "
                       "impervious/canopy overlays skipped; population still built.")
    else:
        logger.info(f"heat-risk-sd: {hrs}")
        build_impervious(hrs, land_geoms)
        build_canopy(hrs, land_geoms)
    logger.info(f"done -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
