"""Crosswalk UCLA Heat Maps excess-ER + CalHeatScore thresholds to SD tracts.

The UCLA Heat Maps publish excess heat-related ER visits and the localized
CalHeatScore temperature thresholds per ZIP, but the heat-risk viewer is
tract-based. This script assigns each tract the value of the San Diego ZIP whose
polygon contains the tract's centroid — the same centroid-in-polygon crosswalk
used by build_hosp_tract_xwalk.py (SD tracts are ~7x smaller than ZIPs, so this
resolves nearly all tracts cleanly; the rare miss falls back to nearest ZIP).

Slice pulled (see WHERE): excess-ER as a Rate (per 10,000 residents),
CalHeatScore level 4, 2008-2018. The four lvl*_th thresholds are a property of
the ZIP+period (identical across CHS level / metric), so one query carries both.

Self-contained: fetches live from the public UCLA ArcGIS service.

Reads:   data/exposure/raw/census/tracts_sd.geojson
Writes:  data/context/ucla_heat_tract.csv
           columns: GEOID, excess_er_rate, chs1_th, chs2_th, chs3_th, chs4_th

Rebuild with:  python3 scripts/build_ucla_heat_tract_xwalk.py
"""

import json
import logging
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TRACTS_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "tracts_sd.geojson"
OUT_CSV = ROOT / "data" / "context" / "ucla_heat_tract.csv"

ZIP_QUERY = ("https://services7.arcgis.com/2hYXmwteFLemN5bW/arcgis/rest/"
             "services/zip_ER/FeatureServer/0/query")
WHERE = "cnty_nm='San Diego' AND NumRate='Rate' AND Years='2008-2018' AND categry=4"
UTM11N = 32611  # accurate centroids / distances for SD

TH_COLS = {"lvl1_th": "chs1_th", "lvl2_th": "chs2_th",
           "lvl3_th": "chs3_th", "lvl4_th": "chs4_th"}


def fetch_zip_layer():
    """SD ZIP polygons with excess-ER rate + the four CalHeatScore thresholds."""
    url = ZIP_QUERY + "?" + urlencode({
        "where": WHERE,
        "outFields": "zip,excs_ER," + ",".join(TH_COLS),
        "returnGeometry": "true", "outSR": 4326, "f": "geojson"})
    socket.setdefaulttimeout(120)
    gj = json.loads(urllib.request.urlopen(url).read())
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs=4326)
    logger.info(f"fetched {len(gdf)} San Diego ZIPs from UCLA service")
    return gdf.to_crs(UTM11N)


def main():
    zips = fetch_zip_layer()
    zips = zips.rename(columns={"excs_ER": "excess_er_rate", **TH_COLS})
    val_cols = ["excess_er_rate"] + list(TH_COLS.values())
    zips = zips[["geometry"] + val_cols]

    tr = gpd.read_file(TRACTS_PATH)
    tr["GEOID"] = tr["GEOID"].astype(str).str.zfill(11)
    tr["ALAND"] = tr["ALAND"].astype(float)
    tr = tr[tr["ALAND"] >= 100_000].to_crs(UTM11N)        # viewer's land footprint
    cent = tr[["GEOID"]].copy()
    cent["geometry"] = tr.geometry.centroid
    cent = gpd.GeoDataFrame(cent, geometry="geometry", crs=UTM11N)

    # Primary: tract centroid within a ZIP polygon.
    joined = gpd.sjoin(cent, zips, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # ZIPs shouldn't overlap

    # Fallback: nearest ZIP for centroids that landed in a gap / on water.
    miss = joined["excess_er_rate"].isna()
    n_fallback = int(miss.sum())
    if n_fallback:
        near = gpd.sjoin_nearest(cent.loc[miss.values], zips, how="left")
        near = near[~near.index.duplicated(keep="first")]
        for c in val_cols:
            joined.loc[miss.values, c] = near[c].values

    out = joined[["GEOID"] + val_cols].copy()
    out["excess_er_rate"] = out["excess_er_rate"].round(2)
    for c in TH_COLS.values():
        out[c] = out[c].round(1)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    matched = int(out["excess_er_rate"].notna().sum())
    logger.info(f"matched {matched}/{len(out)} tracts ({n_fallback} via nearest-ZIP fallback)")
    logger.info(f"excess_er_rate range: {out.excess_er_rate.min():.1f}–{out.excess_er_rate.max():.1f} per 10k")
    logger.info(f"CHS1 onset range: {out.chs1_th.min():.1f}–{out.chs1_th.max():.1f} °F")
    logger.info(f"wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
