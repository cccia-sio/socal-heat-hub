"""Per-tract values for three display-only Data Layers in the viewer.

None of these are PCA variables. They're context drawn over the map, like the
UCLA excess-ER layer (build_ucla_heat_tract_xwalk.py).

* Unsheltered people: the Regional Task Force on Homelessness 2026 point-in-time
  count, unsheltered people per 2020 tract. RTFH lists a tract once per city it
  touches, so rows are summed. A tract not listed had nobody counted.
* Electricity-dependent residents: HHS emPOWER, Medicare beneficiaries who rely
  on electricity-dependent medical equipment, per ZIP. Each tract takes the ZIP
  that contains its centroid (nearest ZIP if none), the same crosswalk as the
  UCLA layer. Carried as a rate per 1,000 ZIP residents, because a ZIP count
  copied onto every tract in it would overstate each one. ZIP residents are the
  summed population of the tracts assigned to it. (Per 1,000 beneficiaries was
  nearly flat, 26-40 in most of the county.) HHS reports counts of 1-10 as 11,
  so a ZIP at 11 or fewer gets no rate. The ZIP and its count ride along for
  the popup.
* Power shutoffs: SDG&E Public Safety Power Shutoffs by tract, from its CPUC
  post-season filings (POSTSR 2B). These are the only seasons filed by tract with
  any SDG&E shutoffs: none in 2022 or 2023, and 2021 and earlier weren't filed
  by tract. The 2024 file uses 2010 tracts (CalEnviroScreen 4.0), so it goes
  through the Census 2010->2020 relationship file, splitting accounts by land
  area. Accounts are summed over events, so a meter shut off twice counts twice.

Reads:   data/exposure/raw/census/tracts_sd.geojson
         data/sensitivity/cdc_svi_2022_sd.csv  (E_TOTPOP, the viewer's population)
Writes:  data/context/context_layers_tract.csv
           columns: GEOID, unsheltered_n, empower_rate, empower_n, empower_zip,
                    psps_events, psps_accounts, psps_mbl, psps_max_hrs

Rebuild with:  python3 scripts/build_context_layers.py
"""

import io
import json
import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TRACTS_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "tracts_sd.geojson"
SVI_PATH = ROOT / "data" / "sensitivity" / "cdc_svi_2022_sd.csv"
OUT_CSV = ROOT / "data" / "context" / "context_layers_tract.csv"
UTM11N = 32611

RTFH_URL = ("https://www.rtfhsd.org/wp-content/uploads/2026/05/"
            "2026-PITC-Unsheltered-Census-Tract-per-City.xlsx")

EMPOWER_QUERY = ("https://services2.arcgis.com/ZQ4jTQn6k7VPXEwO/arcgis/rest/services/"
                 "HHS_emPOWER_REST_Service_Public/FeatureServer/1/query")

_CPUC = ("https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/"
         "safety-and-enforcement-division/reports")
# (season, url, sheet, tract vintage). The 2024 "No Meter" sheet is the
# tract-level table; the plain "Template" sheet repeats it per meter.
PSPS_FILES = [
    ("2024", f"{_CPUC}/2024-psps-post-season-reports/sdge_postsr2b_3-3-2025.xlsx",
     "Template - No Meter", 2010),
    ("2025", f"{_CPUC}/2025-psps-post-season-reports/sdge_postsr2b_3-2-2026.xlsx",
     "Template", 2020),
]
TRACT_REL_URL = ("https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/"
                 "tab20_tract20_tract10_st06.txt")


def _get(url):
    # The CPUC and RTFH sites refuse urllib's default user agent.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _geoid(series):
    return series.astype("int64").astype(str).str.zfill(11)


def unsheltered():
    d = pd.read_excel(io.BytesIO(_get(RTFH_URL)), header=2)
    tract = pd.to_numeric(d["2020 Census Tract"], errors="coerce")  # drops "<City> Total" rows
    d = d[tract.notna()]
    geoid = "06073" + (tract[tract.notna()] * 100).round().astype(int).astype(str).str.zfill(6)
    out = d.groupby(geoid)["Unsheltered Client Count"].sum().rename("unsheltered_n")
    logger.info(f"RTFH 2026: {int(out.sum())} unsheltered people in {len(out)} tracts")
    return out


def empower(tracts):
    url = EMPOWER_QUERY + "?" + urlencode({
        "where": "FIPS_Code='06073'",
        "outFields": "Zip_Code,Medicare_Benes,Power_Dependent_Devices_DME",
        "returnGeometry": "true", "outSR": 4326, "f": "geojson"})
    zips = gpd.GeoDataFrame.from_features(json.loads(_get(url))["features"], crs=4326).to_crs(UTM11N)
    zips = zips.rename(columns={"Zip_Code": "empower_zip", "Power_Dependent_Devices_DME": "empower_n"})
    cols = ["empower_n", "empower_zip"]
    zips = zips[["geometry"] + cols]
    logger.info(f"emPOWER: {len(zips)} San Diego ZIPs, {int(zips.empower_n.sum())} beneficiaries")

    cent = gpd.GeoDataFrame(tracts[["GEOID"]], geometry=tracts.geometry.centroid, crs=UTM11N)
    joined = gpd.sjoin(cent, zips, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    miss = joined["empower_zip"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(cent.loc[miss.values], zips, how="left")
        near = near[~near.index.duplicated(keep="first")]
        for c in cols:
            joined.loc[miss.values, c] = near[c].values
    logger.info(f"emPOWER: {int(miss.sum())} tracts via nearest-ZIP fallback")

    out = joined.set_index("GEOID")[cols]
    svi = pd.read_csv(SVI_PATH, usecols=["FIPS", "E_TOTPOP"])
    pop = svi.set_index(_geoid(svi["FIPS"]))["E_TOTPOP"].clip(lower=0)
    zip_pop = pop.reindex(out.index).groupby(out["empower_zip"]).transform("sum")
    rate = 1000 * out["empower_n"] / zip_pop
    out["empower_rate"] = rate.where((out["empower_n"] > 11) & (zip_pop > 0)).round(1)
    return out


def _tract10_to_20():
    rel = pd.read_csv(io.BytesIO(_get(TRACT_REL_URL)), sep="|", dtype=str, encoding="utf-8-sig")
    rel = rel[rel["GEOID_TRACT_10"].str.startswith("06073")]
    part = rel["AREALAND_PART"].astype(float)
    whole = rel.groupby("GEOID_TRACT_10")["AREALAND_PART"].transform(lambda s: s.astype(float).sum())
    rel["share"] = (part / whole).fillna(0)
    return rel[["GEOID_TRACT_10", "GEOID_TRACT_20", "share"]]


def power_shutoffs():
    rows = []
    for season, url, sheet, vintage in PSPS_FILES:
        d = pd.read_excel(io.BytesIO(_get(url)), sheet_name=sheet, header=3)
        d.columns = [" ".join(str(c).split()) for c in d.columns]
        tcol = "TractID" if "TractID" in d.columns else "GEOID"
        d = d[d[tcol].notna()]
        d = pd.DataFrame({
            "event": d["EVENTID"],
            "GEOID": _geoid(d[tcol]),
            "accounts": d["Total number of customer accounts de-energized"],
            "mbl": d["Total number of MBL customer accounts de-energized"],
            "max_min": d["Maximum amount of time de-energized (minutes)"],
        })
        d = d[d["GEOID"].str.startswith("06073")]
        if vintage == 2010:
            d = d.merge(_tract10_to_20(), left_on="GEOID", right_on="GEOID_TRACT_10")
            d["GEOID"] = d["GEOID_TRACT_20"]
            d["accounts"] *= d["share"]
            d["mbl"] *= d["share"]
            d = d[d["accounts"] >= 0.5]  # a sliver of a split tract didn't lose power
        logger.info(f"PSPS {season}: {d.event.nunique()} events, {round(d.accounts.sum())} accounts")
        rows.append(d)
    d = pd.concat(rows)
    out = d.groupby("GEOID").agg(
        psps_events=("event", "nunique"),
        psps_accounts=("accounts", "sum"),
        psps_mbl=("mbl", "sum"),
        psps_max_hrs=("max_min", "max"),
    )
    out["psps_accounts"] = out["psps_accounts"].round().astype(int)
    out["psps_mbl"] = out["psps_mbl"].round().astype(int)
    out["psps_max_hrs"] = (out["psps_max_hrs"] / 60).round(1)
    logger.info(f"PSPS: {len(out)} tracts with at least one shutoff")
    return out


def main():
    tracts = gpd.read_file(TRACTS_PATH)
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    tracts = tracts[tracts["ALAND"].astype(float) >= 100_000].to_crs(UTM11N)  # viewer's footprint

    out = pd.DataFrame(index=pd.Index(tracts["GEOID"], name="GEOID"))
    uns = unsheltered()
    dropped = uns[~uns.index.isin(out.index)]
    if len(dropped):
        logger.warning(f"RTFH people in tracts the viewer doesn't draw: {dropped.to_dict()}")
    out = out.join(uns).join(empower(tracts)).join(power_shutoffs())

    for c in ["unsheltered_n", "psps_events", "psps_accounts", "psps_mbl"]:
        out[c] = out[c].fillna(0).astype(int)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(OUT_CSV, index=False)
    logger.info(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(out)} tracts)")


if __name__ == "__main__":
    main()
