"""Build interactive DIY PCA viewer for heat risk exploration.

Generates a self-contained HTML file where users can assign variables to
pillars (Exposure, Sensitivity, Adaptive Capacity), run PCA client-side
via Jacobi eigendecomposition, and view results on a Leaflet choropleth.

Usage:
    python scripts/build_diy_pca_viewer.py

Output:
    figures/heat_risk/main/diy_pca.html
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "main" / "diy_pca.html"
TRACTS_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "tracts_sd.geojson"
JURIS_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "jurisdictions_sd.geojson"
TRIBAL_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "tribal_lands_sd.geojson"

# ── CMIP6 / LOCA2 projection settings ──────────────────────────────────────
LOCA2_DIR = ROOT / "data" / "exposure" / "raw" / "loca2"
# 30-year windows centered on the labeled decade
FUTURE_PERIODS = {"2050": "2035-2064", "2080": "2065-2094"}
FUTURE_PERIOD_FALLBACK = "2040-2059"  # last-resort short window if 30-yr missing
SSPS = ["ssp245", "ssp370", "ssp585"]
SSP_LABELS = {"ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
LOCA2_VARS = ["tasmax", "tasmin"]

# Variables whose risk polarity is OPPOSITE of their pillar default.
# Pillar defaults: exposure/sensitivity → higher value = higher risk;
# adaptive → higher value = lower risk (more capacity is protective).
# List below contains the exceptions to those defaults.
INVERTED_VARS = {
    # Sensitivity — HPI "Health & Environment" indicators are inverse-normalized
    # (higher percentile = healthier / cleaner = lower risk).
    "leb_pctile",
    "pollution_pctile",
    "ozone_pctile",
    "pm25_pctile",
    "dieselpm_pctile",
    # Adaptive Capacity — these are vulnerabilities, not protections
    "prob_NoAC",
    "pct_pre1980",
    "pct_pre1960",
    "pct_renter",
}

# Which DIY PCA variables get which LOCA2 delta applied (ΔK = ΔC for differences)
DELTA_VAR_MAP = {
    "mean_tmax": "tasmax", "p85_tmax": "tasmax",
    "p99_tmax": "tasmax", "max_tmax": "tasmax",
    "mean_tmin_overnight": "tasmin", "p90_tmin_overnight": "tasmin",
    # Landsat LST — tasmax delta as best-available proxy
    "mean_lst": "tasmax",
    # HRRR-derived WBGT — tasmax delta as first-order proxy
    # (true WBGT delta would need future RH/wind/solar, not in LOCA2)
    "p85_wbgt": "tasmax", "p99_wbgt": "tasmax",
}

# Variable definitions: (column_key, human_label, pillar, subsection)
VARIABLES = [
    # ── Exposure ─────────────────────────────────────────────────────────
    # Heat
    ("mean_tmax", "Mean Daily Maximum Temperature", "exposure", "Heat"),
    ("p85_tmax", "85th Percentile Daily Maximum Temperature", "exposure", "Heat"),
    ("p99_tmax", "99th Percentile Daily Maximum Temperature", "exposure", "Heat"),
    ("max_tmax", "Annual Maximum Temperature", "exposure", "Heat"),
    # ("extreme_day_fraction", "Extreme Heat Day Fraction", "exposure", "Heat"),
    #   Removed pending decision on canonical definition (ETCCDI TX90p vs current
    #   custom DOY-95 / zone-OR rule). Data still computed upstream — re-enable
    #   here once the methodology is finalized.
    ("mean_lst", "Mean Land Surface Temperature (Landsat)", "exposure", "Heat"),
    ("p85_wbgt", "85th Percentile Wet-Bulb Globe Temperature (WBGT)", "exposure", "Heat"),
    ("p99_wbgt", "99th Percentile Wet-Bulb Globe Temperature (WBGT)", "exposure", "Heat"),
    # Nighttime Heat
    ("mean_tmin_overnight", "Mean Overnight Minimum Temperature", "exposure", "Nighttime Heat"),
    ("p90_tmin_overnight", "90th Percentile Overnight Minimum Temperature", "exposure", "Nighttime Heat"),
    ("no_relief_fraction", "No-Relief Night Fraction", "exposure", "Nighttime Heat"),
    # Humidity
    ("mean_rh_hrrr", "Mean Relative Humidity", "exposure", "Humidity"),
    ("mean_rh_hot_hours", "Mean Humidity During Hot Hours", "exposure", "Humidity"),
    ("p95_dewpoint", "95th Percentile Dewpoint Temperature", "exposure", "Humidity"),
    # ── Sensitivity ──────────────────────────────────────────────────────
    # Health — CDC PLACES
    ("bphigh_prev", "High Blood Pressure", "sensitivity", "Health"),
    ("casthma_prev", "Current Asthma", "sensitivity", "Health"),
    ("chd_prev", "Coronary Heart Disease", "sensitivity", "Health"),
    ("cognition_prev", "Cognitive Difficulty", "sensitivity", "Health"),
    ("copd_prev", "COPD", "sensitivity", "Health"),
    ("depression_prev", "Depression", "sensitivity", "Health"),
    ("diabetes_prev", "Diabetes", "sensitivity", "Health"),
    ("stroke_prev", "Stroke", "sensitivity", "Health"),
    ("leb_pctile", "Life Expectancy", "sensitivity", "Health"),
    # Demographics — SVI
    ("EP_AGE65", "Age 65+", "sensitivity", "Demographics"),
    ("EP_AGE17", "Age 17 & Under", "sensitivity", "Demographics"),
    ("EP_DISABL", "Disability", "sensitivity", "Demographics"),
    ("EP_LIMENG", "Limited English", "sensitivity", "Demographics"),
    ("EP_SNGPNT", "Single-Parent HH", "sensitivity", "Demographics"),
    ("EP_POV150", "Below 150% Poverty", "sensitivity", "Demographics"),
    ("EP_UNEMP", "Unemployed", "sensitivity", "Demographics"),
    ("EP_HBURD", "Housing Cost Burden", "sensitivity", "Demographics"),
    ("EP_NOHSDP", "No HS Diploma", "sensitivity", "Demographics"),
    ("EP_UNINSUR", "Uninsured", "sensitivity", "Demographics"),
    ("EP_MINRTY", "Minority", "sensitivity", "Demographics"),
    ("EP_MUNIT", "Multi-Unit Structures", "sensitivity", "Demographics"),
    ("EP_MOBILE", "Mobile Homes", "sensitivity", "Demographics"),
    ("EP_CROWD", "Crowding", "sensitivity", "Demographics"),
    ("EP_NOVEH", "No Vehicle", "sensitivity", "Demographics"),
    ("EP_GROUPQ", "Group Quarters", "sensitivity", "Demographics"),
    ("EP_NOINT", "No Internet", "sensitivity", "Demographics"),
    # SVI Themes
    ("RPL_THEMES", "Overall SVI", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME1", "Socioeconomic Status", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME2", "HH Disability/Language", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME3", "Racial/Ethnic Minority", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME4", "Housing/Transportation", "sensitivity", "Social Vulnerability Index (SVI)"),
    # ── Adaptive Capacity ────────────────────────────────────────────────
    # Air Conditioning
    ("ac_prob", "AC Prevalence", "adaptive", "Air Conditioning"),
    ("prob_NoAC", "Prob. No AC", "adaptive", "Air Conditioning"),
    ("prob_Central", "Prob. Central AC", "adaptive", "Air Conditioning"),
    # Built Environment
    ("treecanopy_pctile", "Tree Canopy", "adaptive", "Built Environment"),
    ("parkaccess_pctile", "Park Access", "adaptive", "Built Environment"),
    ("pct_pre1980", "Pre-1980 Housing (%)", "adaptive", "Built Environment"),
    ("pct_pre1960", "Pre-1960 Housing (%)", "adaptive", "Built Environment"),
    ("pct_renter", "Renter-Occupied (%)", "adaptive", "Built Environment"),
    ("pct_green", "Green Space (%)", "adaptive", "Built Environment"),
    # Socioeconomic — HPI
    ("income_pctile", "Income", "adaptive", "Socioeconomic"),
    ("abovepoverty_pctile", "Above Poverty", "adaptive", "Socioeconomic"),
    ("economic_pctile", "Economic", "adaptive", "Socioeconomic"),
    ("education_pctile", "Education", "adaptive", "Socioeconomic"),
    ("houseing_pctile", "Housing Quality", "adaptive", "Socioeconomic"),
    ("healthcareaccess_pctile", "Healthcare Access", "adaptive", "Socioeconomic"),
    ("transportation_pctile", "Transportation", "adaptive", "Socioeconomic"),
    ("social_pctile", "Social Cohesion", "adaptive", "Socioeconomic"),
    ("neighborhood_pctile", "Neighborhood Quality", "adaptive", "Socioeconomic"),
    ("insured_pctile", "Health Insurance", "adaptive", "Socioeconomic"),
    ("automobile_pctile", "Automobile Access", "adaptive", "Socioeconomic"),
    ("homeownership_pctile", "Homeownership", "adaptive", "Socioeconomic"),
    ("commute_pctile", "Commute Time", "adaptive", "Socioeconomic"),
    ("supermkts_pctile", "Supermarket Access", "adaptive", "Socioeconomic"),
    ("voting_pctile", "Voter Participation", "adaptive", "Socioeconomic"),
    ("retail_pctile", "Retail Food Access", "adaptive", "Socioeconomic"),
    # Environment
    ("pollution_pctile", "Pollution (Inv.)", "sensitivity", "Environment"),
    ("ozone_pctile", "Ozone", "sensitivity", "Environment"),
    ("pm25_pctile", "PM 2.5", "sensitivity", "Environment"),
    ("dieselpm_pctile", "Diesel PM", "sensitivity", "Environment"),
]

# ── Source URLs per variable ─────────────────────────────────────────────
_SRC_MESOWEST = "https://synopticdata.com/"
_SRC_PLACES = "https://www.cdc.gov/places/index.html"
_SRC_SVI = "https://www.atsdr.cdc.gov/place-health/php/svi/index.html"
_SRC_HPI = "https://www.healthyplacesindex.org/"
_SRC_ROMITTI = "https://academic.oup.com/pnasnexus/article/1/4/pgac210/6713629"
_SRC_AHN = "https://doi.org/10.1038/s41597-025-06104-3"
_SRC_ACS_B25034 = "https://censusreporter.org/tables/B25034/"
_SRC_ACS_B25003 = "https://censusreporter.org/tables/B25003/"
_SRC_SANGIS = "https://sdgis-sandag.opendata.arcgis.com/"
_SRC_LANDSAT = "https://www.usgs.gov/landsat-missions"
_SRC_HRRR = "https://rapidrefresh.noaa.gov/hrrr/"

VAR_SOURCES = {
    # Exposure — MesoWest/Synoptic + CIMIS
    "mean_tmax": _SRC_MESOWEST, "p85_tmax": _SRC_MESOWEST,
    "p99_tmax": _SRC_MESOWEST, "max_tmax": _SRC_MESOWEST,
    "extreme_day_fraction": _SRC_MESOWEST,
    "mean_tmin_overnight": _SRC_MESOWEST, "p90_tmin_overnight": _SRC_MESOWEST,
    "no_relief_fraction": _SRC_MESOWEST,
    # Exposure — Landsat LST
    "mean_lst": _SRC_LANDSAT,
    # Exposure — HRRR-derived WBGT (Liljegren/Dimiceli) and humidity
    "p85_wbgt": _SRC_HRRR, "p99_wbgt": _SRC_HRRR,
    "mean_rh_hrrr": _SRC_HRRR, "mean_rh_hot_hours": _SRC_HRRR,
    "p95_dewpoint": _SRC_HRRR,
    # Sensitivity — CDC PLACES
    "bphigh_prev": _SRC_PLACES, "casthma_prev": _SRC_PLACES,
    "chd_prev": _SRC_PLACES, "cognition_prev": _SRC_PLACES,
    "copd_prev": _SRC_PLACES, "depression_prev": _SRC_PLACES,
    "diabetes_prev": _SRC_PLACES, "stroke_prev": _SRC_PLACES,
    # Sensitivity — SVI
    "EP_AGE65": _SRC_SVI, "EP_AGE17": _SRC_SVI, "EP_DISABL": _SRC_SVI,
    "EP_LIMENG": _SRC_SVI, "EP_SNGPNT": _SRC_SVI, "EP_POV150": _SRC_SVI,
    "EP_UNEMP": _SRC_SVI, "EP_HBURD": _SRC_SVI, "EP_NOHSDP": _SRC_SVI,
    "EP_UNINSUR": _SRC_SVI, "EP_MINRTY": _SRC_SVI, "EP_MUNIT": _SRC_SVI,
    "EP_MOBILE": _SRC_SVI, "EP_CROWD": _SRC_SVI, "EP_NOVEH": _SRC_SVI,
    "EP_GROUPQ": _SRC_SVI, "EP_NOINT": _SRC_SVI,
    "RPL_THEMES": _SRC_SVI, "RPL_THEME1": _SRC_SVI, "RPL_THEME2": _SRC_SVI,
    "RPL_THEME3": _SRC_SVI, "RPL_THEME4": _SRC_SVI,
    # Adaptive — Romitti et al. (2022)
    "ac_prob": _SRC_ROMITTI,
    # Adaptive — Ahn & Uejio / COMBUST PLUS
    "prob_NoAC": _SRC_AHN, "prob_Central": _SRC_AHN,
    # Adaptive — ACS
    "pct_pre1980": _SRC_ACS_B25034, "pct_pre1960": _SRC_ACS_B25034,
    "pct_renter": _SRC_ACS_B25003,
    # Adaptive — SanGIS
    "pct_green": _SRC_SANGIS,
    # HPI (adaptive + sensitivity)
    "treecanopy_pctile": _SRC_HPI, "parkaccess_pctile": _SRC_HPI,
    "income_pctile": _SRC_HPI, "abovepoverty_pctile": _SRC_HPI,
    "economic_pctile": _SRC_HPI, "education_pctile": _SRC_HPI,
    "houseing_pctile": _SRC_HPI, "healthcareaccess_pctile": _SRC_HPI,
    "transportation_pctile": _SRC_HPI, "social_pctile": _SRC_HPI,
    "neighborhood_pctile": _SRC_HPI, "insured_pctile": _SRC_HPI,
    "automobile_pctile": _SRC_HPI, "homeownership_pctile": _SRC_HPI,
    "commute_pctile": _SRC_HPI, "supermkts_pctile": _SRC_HPI,
    "voting_pctile": _SRC_HPI, "retail_pctile": _SRC_HPI,
    "leb_pctile": _SRC_HPI, "pollution_pctile": _SRC_HPI,
    "ozone_pctile": _SRC_HPI, "pm25_pctile": _SRC_HPI,
    "dieselpm_pctile": _SRC_HPI,
}

# ── Column groupings for loading ─────────────────────────────────────────
# Columns from tract_heat_risk_all.csv
_EXP_CSV_COLS = [
    "mean_tmax", "p85_tmax", "p99_tmax", "max_tmax", "extreme_day_fraction",
]
# Columns from cdc_places
_PLACES_COLS = [v[0] for v in VARIABLES if v[0].endswith("_prev")]
# Columns from SVI
_SVI_COLS = [v[0] for v in VARIABLES if v[0].startswith(("EP_", "RPL_"))]
# Columns from HPI (2020 tracts file — domain-level)
_HPI_DOMAIN_COLS = [
    "treecanopy_pctile", "parkaccess_pctile", "income_pctile",
    "abovepoverty_pctile", "economic_pctile", "education_pctile",
    "healthcareaccess_pctile", "transportation_pctile", "social_pctile",
    "houseing_pctile", "neighborhood_pctile", "pollution_pctile", "leb_pctile",
]
# Columns from HPI (full file — indicator-level)
_HPI_INDICATOR_COLS = [
    "insured_pctile", "automobile_pctile", "homeownership_pctile",
    "commute_pctile", "supermkts_pctile", "voting_pctile", "retail_pctile",
    "ozone_pctile", "pm25_pctile", "dieselpm_pctile",
]


def _safe_merge(tracts, df, cols, on="GEOID"):
    """Merge only columns that exist in df, warn about missing ones."""
    have = [c for c in cols if c in df.columns]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        logger.warning(f"Columns not found (skipped): {miss}")
    if have:
        tracts = tracts.merge(df[["GEOID"] + have], on=on, how="left")
    return tracts


def load_data():
    """Load all tract-level data, merge, impute NaN, simplify geometry."""
    logger.info("Loading tracts...")
    tracts = gpd.read_file(TRACTS_PATH)
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    tracts["ALAND"] = tracts["ALAND"].astype(float)
    tracts = tracts[tracts["ALAND"] >= 100_000][["GEOID", "geometry"]].copy()

    # ── Exposure (tract_heat_risk_all.csv) ──
    exp = pd.read_csv(ROOT / "data/exposure/processed/extreme_heat/tract_heat_risk_all.csv")
    exp["GEOID"] = exp["GEOID"].astype(str).str.zfill(11)
    tracts = _safe_merge(tracts, exp, _EXP_CSV_COLS)

    # ── Exposure — Landsat LST (mean surface temp) ──
    lst_path = ROOT / "data/exposure/tract_lst_dtr.csv"
    if lst_path.exists():
        lst = pd.read_csv(lst_path, dtype={"GEOID": str})
        lst["GEOID"] = lst["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, lst, ["mean_lst"])

    # ── Exposure — HRRR-derived WBGT (Liljegren/Dimiceli, 2019–2025) ──
    hrrr_path = ROOT / "data/exposure/tract_hrrr_exposure.csv"
    if hrrr_path.exists():
        hrrr = pd.read_csv(hrrr_path, dtype={"GEOID": str})
        hrrr["GEOID"] = hrrr["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, hrrr,
            ["p85_wbgt", "p99_wbgt", "mean_rh_hrrr", "p95_dewpoint", "mean_rh_hot_hours"])

    # ── Exposure — Nighttime heat (IDW from stations) ──
    night_path = ROOT / "data/exposure/processed/extreme_heat/tract_nighttime_heat.csv"
    if night_path.exists():
        night = pd.read_csv(night_path, dtype={"GEOID": str})
        night["GEOID"] = night["GEOID"].str.zfill(11)
        tracts = _safe_merge(
            tracts, night,
            ["mean_tmin_overnight", "p90_tmin_overnight", "no_relief_fraction"],
        )

    # ── Sensitivity — CDC PLACES ──
    places = pd.read_csv(ROOT / "data/sensitivity/cdc_places_sd_tracts.csv")
    places["GEOID"] = places["GEOID"].astype(str).str.zfill(11)
    tracts = _safe_merge(tracts, places, _PLACES_COLS)

    # ── Sensitivity — SVI (also pull E_TOTPOP for population-weighted city aggregations) ──
    svi = pd.read_csv(ROOT / "data/sensitivity/cdc_svi_2022_sd.csv",
                       usecols=lambda c: c in {"FIPS", "E_TOTPOP"} | set(_SVI_COLS)).copy()
    svi["GEOID"] = svi["FIPS"].astype(str).str.zfill(11)
    for col in _SVI_COLS:
        if col in svi.columns:
            svi[col] = pd.to_numeric(svi[col], errors="coerce")
            svi.loc[svi[col] < 0, col] = np.nan
    if "E_TOTPOP" in svi.columns:
        svi["E_TOTPOP"] = pd.to_numeric(svi["E_TOTPOP"], errors="coerce")
        svi.loc[svi["E_TOTPOP"] < 0, "E_TOTPOP"] = np.nan
    tracts = _safe_merge(tracts, svi, _SVI_COLS + ["E_TOTPOP"])

    # ── Adaptive — AC prevalence ──
    ac = pd.read_csv(
        ROOT / "data/adaptive_capacity/aircon/tract_ac_prevalence_sd_summary.csv"
    )
    ac["GEOID"] = ac["GEOID"].astype(str).str.zfill(11)
    tracts = _safe_merge(tracts, ac, ["ac_prob"])

    # ── Adaptive — AC type probabilities (Ahn et al.) ──
    ac_type_path = (ROOT / "data/adaptive_capacity/aircon/ahn_etal"
                    / "Uncertainty_tract/summary_ACprob_tract_2020.csv")
    if ac_type_path.exists():
        act = pd.read_csv(ac_type_path)
        act["GEOID"] = act["GEOID"].astype(str).str.zfill(11)
        # Rename columns with spaces
        act = act.rename(columns={"prob_Evaporative Cooler": "prob_EvapCooler"})
        tracts = _safe_merge(tracts, act, ["prob_NoAC", "prob_Central"])

    # ── Adaptive — HPI domain-level (2020 tracts) ──
    hpi = pd.read_csv(ROOT / "data/sensitivity/hpi/hpi_san_diego_2020tracts.csv")
    hpi["GEOID"] = hpi["GEOID_TRACT_20"].astype(str).str.zfill(11)
    tracts = _safe_merge(tracts, hpi, _HPI_DOMAIN_COLS)

    # ── Adaptive — HPI indicator-level (full file) ──
    hpi_full_path = ROOT / "data/sensitivity/hpi/hpi_san_diego.csv"
    if hpi_full_path.exists():
        hpi_full = pd.read_csv(hpi_full_path, dtype={"geoid": str})
        hpi_full["GEOID"] = hpi_full["geoid"].astype(str).str.zfill(11)
        tracts = _safe_merge(tracts, hpi_full, _HPI_INDICATOR_COLS)

    # ── Adaptive — Building age (ACS B25034) ──
    bldg_path = ROOT / "data/adaptive_capacity/building_age/acs_b25034_year_built_sd_tracts.csv"
    if bldg_path.exists():
        bldg = pd.read_csv(bldg_path, dtype={"GEOID": str})
        bldg["GEOID"] = bldg["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, bldg, ["pct_pre1980", "pct_pre1960"])

    # ── Adaptive — Renter occupancy (ACS demographics) ──
    acs_path = ROOT / "data/exposure/raw/census/acs_demographics_sd.csv"
    if acs_path.exists():
        acs = pd.read_csv(acs_path, dtype={"GEOID": str})
        acs["GEOID"] = acs["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, acs, ["pct_renter"])

    # ── Context — Land use (tract_exploration.csv) ──
    ctx_path = ROOT / "data/context/tract_exploration.csv"
    if ctx_path.exists():
        ctx = pd.read_csv(ctx_path, dtype={"geoid": str})
        ctx["GEOID"] = ctx["geoid"].astype(str).str.zfill(11)
        tracts = _safe_merge(tracts, ctx, ["pct_green"])

    # ── Impute & simplify ──
    var_cols = [v[0] for v in VARIABLES]
    loaded = [c for c in var_cols if c in tracts.columns]
    for col in loaded:
        tracts[col] = tracts[col].fillna(tracts[col].median()).round(2)

    # Population field — used for population-weighted city aggregations in the
    # compare-cities modal. Not a user-facing variable, so kept off VARIABLES.
    if "E_TOTPOP" in tracts.columns:
        tracts["E_TOTPOP"] = tracts["E_TOTPOP"].fillna(0).round().astype(int)
        extra = ["E_TOTPOP"]
    else:
        extra = []

    tracts = tracts.to_crs(epsg=4326)
    tracts["geometry"] = tracts.geometry.simplify(0.0003)

    keep = ["GEOID", "geometry"] + loaded + extra
    tracts = tracts[keep]
    logger.info(f"Loaded {len(tracts)} tracts, {len(loaded)}/{len(var_cols)} variables")
    return tracts


def compute_tract_deltas(tracts):
    """Compute per-tract CMIP6 deltas via centroid interpolation from LOCA2.

    For each future window in FUTURE_PERIODS, loads ensemble-mean NetCDF
    deltas (3km), averages across 12 months to get annual-mean delta, fills
    ocean NaN, and bilinearly interpolates to tract centroids.

    Returns {period_label: {ssp: {loca2_var: [delta_per_tract]}}} or None if
    no period has any usable files. period_label is e.g. "2050" / "2080".
    """
    # Tract centroids in lon/lat (project to UTM for accurate centroids, then back)
    tracts_utm = tracts.to_crs(epsg=32611)
    centroids_4326 = tracts_utm.geometry.centroid.to_crs(epsg=4326)
    c_lon = centroids_4326.x.values
    c_lat = centroids_4326.y.values

    out = {}
    for label, period in FUTURE_PERIODS.items():
        test_path = LOCA2_DIR / "deltas_ensemble" / "ssp370" / f"tasmax_delta_{period}.nc"
        if not test_path.exists():
            alt = FUTURE_PERIOD_FALLBACK
            test_path = LOCA2_DIR / "deltas_ensemble" / "ssp370" / f"tasmax_delta_{alt}.nc"
            if not test_path.exists():
                logger.warning(f"No LOCA2 deltas for period {label} ({period}) — skipping")
                continue
            logger.info(f"Period {label}: using fallback {alt} (30-yr {period} missing)")
            period = alt

        deltas = {}
        for ssp in SSPS:
            deltas[ssp] = {}
            for var in LOCA2_VARS:
                path = LOCA2_DIR / "deltas_ensemble" / ssp / f"{var}_delta_{period}.nc"
                if not path.exists():
                    logger.warning(f"Missing {path.name} — skipping {var}/{ssp}/{label}")
                    continue

                da = xr.open_dataarray(path)
                annual = da.mean(dim="month").values  # (lat, lon)
                lat = da.lat.values
                lon = da.lon.values

                mask = np.isfinite(annual)
                if not mask.all():
                    _, nearest_idx = distance_transform_edt(
                        ~mask, return_distances=True, return_indices=True,
                    )
                    annual = annual[tuple(nearest_idx)]

                interp = RegularGridInterpolator(
                    (lat, lon), annual, method="linear",
                    bounds_error=False, fill_value=None,
                )
                vals = interp(np.column_stack([c_lat, c_lon]))
                deltas[ssp][var] = [round(float(v), 2) for v in vals]

                logger.info(
                    f"  {label}/{ssp}/{var}: mean Δ={np.nanmean(vals):+.2f}°C, "
                    f"range=[{np.nanmin(vals):+.2f}, {np.nanmax(vals):+.2f}]"
                )

        out[label] = deltas

    if not out:
        logger.warning("No LOCA2 delta files found for any period — skipping CMIP6 projections")
        return None
    logger.info(f"Computed tract deltas for periods: {list(out.keys())}, SSPs: {SSPS}")
    return out


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracts = load_data()

    # Compute CMIP6 tract-level deltas (None if files missing)
    deltas = compute_tract_deltas(tracts)

    # Serialize — ensure_ascii so no raw unicode in JS source
    geojson_str = tracts.to_json()
    var_meta_str = json.dumps(
        [{"key": v[0], "label": v[1], "pillar": v[2], "sub": v[3]}
         for v in VARIABLES if v[0] in tracts.columns],
        ensure_ascii=True,
    )
    var_sources_str = json.dumps(
        {k: v for k, v in VAR_SOURCES.items() if k in tracts.columns},
        ensure_ascii=True,
    )
    deltas_str = json.dumps(deltas, ensure_ascii=True) if deltas else "null"
    delta_map_str = json.dumps(
        {k: v for k, v in DELTA_VAR_MAP.items() if k in tracts.columns},
        ensure_ascii=True,
    )
    # Per-variable risk polarity: True = higher value means higher risk (color
    # red when greater than the comparison tract); False = higher means lower
    # risk (color green when greater).
    polarity = {}
    for key, _label, pillar, _sub in VARIABLES:
        if key not in tracts.columns:
            continue
        default_higher_is_worse = pillar in ("exposure", "sensitivity")
        flipped = key in INVERTED_VARS
        polarity[key] = default_higher_is_worse ^ flipped
    polarity_str = json.dumps(polarity, ensure_ascii=True)

    # Load Landsat LST overlay metadata (optional)
    lst_meta_path = ROOT / "data/exposure/processed/landsat/annual_lst_bounds.json"
    lst_str = "null"
    if lst_meta_path.exists():
        lst_str = lst_meta_path.read_text().strip()
        logger.info("Loaded Landsat LST overlay metadata")
    else:
        logger.warning("No LST overlay metadata — compare-mode mini-maps will fall back to outline only")

    # Load city jurisdiction boundaries (optional)
    juris_str = "null"
    if JURIS_PATH.exists():
        juris = gpd.read_file(JURIS_PATH)
        juris = juris[juris["category"] == "City"][["NAME", "geometry"]].copy()
        juris = juris.to_crs(epsg=4326)
        juris["geometry"] = juris.geometry.simplify(0.0005)
        juris_str = juris.to_json()
        logger.info(f"Loaded {len(juris)} city jurisdictions")
    else:
        logger.warning("No jurisdictions file — city boundaries will not be shown")

    # Load tribal lands (optional) — Census TIGER AIANNH 2024, SD County subset
    tribal_str = "null"
    if TRIBAL_PATH.exists():
        tribal_str = TRIBAL_PATH.read_text().strip()
        logger.info(f"Loaded tribal lands from {TRIBAL_PATH.name}")
    else:
        logger.warning("No tribal lands file — tribal overlay will not be shown")

    html = HTML_TEMPLATE
    html = html.replace("__GEOJSON_PLACEHOLDER__", geojson_str)
    html = html.replace("__VARMETA_PLACEHOLDER__", var_meta_str)
    html = html.replace("__VARSOURCES_PLACEHOLDER__", var_sources_str)
    html = html.replace("__DELTAS_PLACEHOLDER__", deltas_str)
    html = html.replace("__DELTAMAP_PLACEHOLDER__", delta_map_str)
    html = html.replace("__JURIS_PLACEHOLDER__", juris_str)
    html = html.replace("__TRIBAL_PLACEHOLDER__", tribal_str)
    html = html.replace("__LST_PLACEHOLDER__", lst_str)
    html = html.replace("__POLARITY_PLACEHOLDER__", polarity_str)

    OUTPUT_PATH.write_text(html)
    size_kb = len(html) // 1024
    logger.info(f"Saved to {OUTPUT_PATH} ({size_kb} KB)")

    rel = OUTPUT_PATH.relative_to(ROOT)
    print(f"\nGitHub Pages URL (after push):")
    print(f"  https://connorjmack.github.io/heat-risk-viewer/{rel}")


# ---------------------------------------------------------------------------
# HTML template — self-contained Leaflet + JS PCA viewer
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>San Diego Heat Risk Explorer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script src="https://unpkg.com/protomaps-leaflet@4.1.1/dist/protomaps-leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{display:flex;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     height:100vh;overflow:hidden;background:#f0f2f5}

#sidebar{width:320px;height:100vh;overflow-y:auto;padding:20px 16px;
  background:#fafafa;border-right:1px solid #e0e0e0;flex-shrink:0;
  scrollbar-width:thin}
#sidebar::-webkit-scrollbar{width:6px}
#sidebar::-webkit-scrollbar-thumb{background:#ccc;border-radius:3px}
h1{font-size:17px;font-weight:700;letter-spacing:-0.3px}
.subtitle{font-size:12px;color:#888;margin:4px 0 16px;line-height:1.4}

#scenario-bar{margin:18px 0 0;padding:14px 0 0;border-top:1px solid #e0e0e0;display:none}
.scenario-label{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;color:#666;margin-bottom:8px}
.scenario-btns{display:flex;flex-direction:column;gap:6px}
.scn-btn{padding:8px 12px;border:1px solid #ddd;border-radius:6px;
  background:#fff;font-size:12px;font-weight:500;cursor:pointer;
  font-family:inherit;transition:all 0.15s;text-align:left;line-height:1.3}
.scn-btn:hover{background:#f0f0f0}
.scn-btn.active{background:#2563eb;color:#fff;border-color:#2563eb}
.scn-period-row{display:flex;gap:6px}
.scn-period-btn{flex:1;padding:8px 6px;border:1px solid #ddd;border-radius:6px;
  background:#fff;font-size:12px;font-weight:500;cursor:pointer;
  font-family:inherit;transition:all 0.15s;text-align:center;line-height:1.2}
.scn-period-btn:hover{background:#f0f0f0}
.scn-period-btn.active{background:#2563eb;color:#fff;border-color:#2563eb}
.scn-period-btn.expanded:not(.active){background:#eef2ff;border-color:#c7d2fe;color:#1e40af}
.scn-ssp-panel{margin:8px 0 0;padding:8px;background:#f8f9fb;
  border-radius:6px;border:1px solid #eef0f4}
.period-group-label{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;color:#888;margin:0 0 5px 2px}
.scenario-info{font-size:10px;color:#888;margin-top:8px;font-style:italic;min-height:14px}
.fixed-tag{font-size:9px;color:#999;font-style:italic;margin-left:auto;white-space:nowrap}

.pillar-section{margin-bottom:14px}
.pillar-hdr{font-size:12px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;padding:6px 0 4px;margin-bottom:2px;border-bottom:2px solid;
  display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.pillar-hdr input[type=checkbox]{width:14px;height:14px;accent-color:currentColor;
  cursor:pointer;flex-shrink:0;margin:0}
details.subsec{margin:0 0 2px}
details.subsec summary{font-size:11.5px;font-weight:600;color:#555;
  padding:4px 0 2px;cursor:pointer;user-select:none;list-style:none}
details.subsec summary::-webkit-details-marker{display:none}
details.subsec summary::before{content:"\\25B6 ";font-size:8px;margin-right:4px;
  display:inline-block;transition:transform 0.15s}
details.subsec[open] summary::before{transform:rotate(90deg)}
.var-row{display:flex;align-items:center;gap:8px;padding:1px 0 1px 12px}
.var-row label{display:flex;align-items:center;gap:6px;font-size:12px;
  color:#444;cursor:pointer;user-select:none}
.var-row input[type=checkbox]{width:14px;height:14px;accent-color:currentColor;
  cursor:pointer;flex-shrink:0}
.src-link{display:none;font-size:10px;color:#2563eb;text-decoration:none;
  margin-left:auto;white-space:nowrap;flex-shrink:0}
.src-link:hover{text-decoration:underline}
.var-row.on .src-link{display:inline}

#compute-btn{width:100%;padding:11px;margin:16px 0 12px;
  background:#2563eb;color:#fff;border:none;border-radius:10px;
  font-size:14px;font-weight:600;cursor:pointer;transition:background 0.2s;
  font-family:inherit}
#compute-btn:hover{background:#1d4ed8}
#compute-btn:active{background:#1e40af}

#results{font-size:12px;color:#555;margin-bottom:12px}
.res-row{margin-bottom:3px}
.res-row b{font-weight:600}
.res-empty{color:#bbb;font-style:italic}

#layer-select{border-top:1px solid #e0e0e0;padding-top:10px;margin-top:4px}
#layer-select h3{font-size:12px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;margin-bottom:6px;color:#666}
.lbtn{display:flex;align-items:center;gap:8px;width:100%;
  padding:6px 10px;border:none;border-radius:8px;background:transparent;
  font-size:12px;font-weight:500;color:#666;cursor:pointer;
  font-family:inherit;text-align:left;transition:background 0.15s}
.lbtn:hover{background:#eee}
.lbtn.active{background:#e8eeff;color:#2563eb;font-weight:600}
.lbtn-txt{flex:1;min-width:0}
.lbtn-pct{color:#999;font-weight:500}
.lbtn.active .lbtn-pct{color:#2563eb}
.ldot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
/* Pattern-button info icon sits flush with the right edge of the sidebar; the
   sidebar's overflow-y:auto clips horizontal overflow, so anchor the popup's
   right edge to the icon and let it grow leftward into the sidebar. */
.lbtn .info-pop{left:auto;right:0}
.lbtn .info-pop::after{left:auto;right:5px}
details.pc-group{margin-top:4px;border-top:1px dashed #e5e5e5;padding-top:4px}
details.pc-group>summary{list-style:none;cursor:pointer;
  padding:6px 10px;border-radius:8px;font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:0.4px;color:#555;
  display:flex;align-items:center;gap:6px;user-select:none;
  transition:background 0.15s}
details.pc-group>summary:hover{background:#eee}
details.pc-group>summary::-webkit-details-marker{display:none}
details.pc-group>summary::before{content:"\\25B8";font-size:10px;color:#999;
  transition:transform 0.15s;display:inline-block}
details.pc-group[open]>summary::before{transform:rotate(90deg)}
.pc-help{font-size:10.5px;color:#888;line-height:1.4;
  padding:2px 10px 4px 10px;font-style:italic}
.pc-help .info-ic-mini{display:inline-flex;align-items:center;
  justify-content:center;width:11px;height:11px;border-radius:50%;
  background:#2563eb;color:#fff;font-size:9px;font-weight:700;font-style:italic;
  font-family:Georgia,'Times New Roman',serif;line-height:1;
  vertical-align:middle}

#map-wrap{flex:1;position:relative;height:100vh}
#map{width:100%;height:100%}

#logos{position:absolute;top:14px;right:14px;z-index:999;
  background:rgba(255,255,255,0.92);padding:8px 14px;border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,0.10);backdrop-filter:blur(12px);
  display:flex;flex-direction:row;align-items:center;gap:14px}
#logo-sio{position:absolute;bottom:30px;right:14px;z-index:999;
  background:rgba(255,255,255,0.92);padding:8px 14px;border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,0.10);backdrop-filter:blur(12px);
  display:flex;align-items:center}
#logos img,#logo-sio img{display:block;height:30px;width:auto}

#legend{position:absolute;top:14px;left:60px;z-index:999;
  background:rgba(255,255,255,0.92);padding:10px 14px;border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,0.10);font-size:11px;
  backdrop-filter:blur(12px);display:none}
#legend .grad{height:12px;width:160px;border-radius:3px;margin:4px 0 2px}
#legend .labels{display:flex;justify-content:space-between;font-size:10px;color:#888}
.dl-row{display:flex;align-items:center;gap:6px;padding:1px 0}
.dl-sw{width:12px;height:12px;border-radius:2px;flex-shrink:0}
.info-ic{display:inline-flex;align-items:center;justify-content:center;
  width:15px;height:15px;border-radius:50%;background:#2563eb;color:#fff;
  font-size:11px;font-weight:700;font-style:italic;
  font-family:Georgia,'Times New Roman',serif;margin-left:6px;cursor:help;
  user-select:none;vertical-align:middle;position:relative;line-height:1;
  box-shadow:0 1px 2px rgba(0,0,0,0.15)}
.info-ic:hover,.info-ic:focus{background:#1d4ed8;outline:none}
.info-pop{display:none;position:absolute;top:calc(100% + 8px);left:0;
  width:220px;background:#1f2937;color:#fff;font-size:11px;font-style:normal;
  font-weight:400;font-family:inherit;line-height:1.45;padding:8px 10px;
  border-radius:6px;box-shadow:0 4px 14px rgba(0,0,0,0.18);z-index:1500;
  text-align:left;pointer-events:none;letter-spacing:normal;white-space:normal}
.info-ic:hover .info-pop,.info-ic:focus .info-pop{display:block}
.info-pop::after{content:"";position:absolute;bottom:100%;left:5px;
  border:5px solid transparent;border-bottom-color:#1f2937}

#prompt{position:absolute;inset:0;z-index:1000;
  display:flex;align-items:center;justify-content:center;
  background:rgba(255,255,255,0.55);font-size:15px;font-weight:500;
  color:#888;pointer-events:none}

#error-bar{position:absolute;top:10px;left:50%;transform:translateX(-50%);
  z-index:1001;background:#fee2e2;color:#991b1b;padding:8px 16px;
  border-radius:8px;font-size:13px;display:none;max-width:80%}
#dl-pdf-btn:hover{background:#f3f4f6;border-color:#9ca3af}
#dl-pdf-btn:disabled{opacity:0.6;cursor:wait}
#pdf-overlay{position:fixed;inset:0;z-index:2500;display:none;
  align-items:center;justify-content:center;background:rgba(255,255,255,0.65);
  backdrop-filter:blur(2px)}
#pdf-overlay.show{display:flex}
#pdf-overlay .box{background:#fff;padding:18px 22px;border-radius:10px;
  box-shadow:0 4px 16px rgba(0,0,0,0.18);font-size:13px;color:#374151;
  display:flex;align-items:center;gap:10px}
#pdf-overlay .spinner{width:14px;height:14px;border:2px solid #d1d5db;
  border-top-color:#2563eb;border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* --- export-data feature --- */
#dl-data-btn{margin-top:8px}
#dl-data-btn:hover{background:#f3f4f6;border-color:#9ca3af}
#export-modal{position:fixed;inset:0;z-index:2200;background:rgba(20,25,40,0.55);
  display:none;align-items:center;justify-content:center;
  backdrop-filter:blur(2px);font-family:inherit}
#export-modal.open{display:flex}
.exp-panel{background:#fff;width:min(460px,94vw);max-height:90vh;
  border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,0.30);
  display:flex;flex-direction:column;overflow:hidden}
.exp-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid #e5e7eb;flex-shrink:0}
.exp-hdr h2{font-size:15px;font-weight:700;margin:0;color:#111}
.exp-close{background:none;border:none;font-size:22px;color:#666;cursor:pointer;
  padding:0 4px;line-height:1;font-family:inherit}
.exp-close:hover{color:#111}
.exp-body{padding:14px 18px;overflow-y:auto;font-size:12px;color:#444;flex:1}
.exp-section{margin-bottom:14px}
.exp-section-label{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;color:#666;margin-bottom:6px}
.exp-radios{display:flex;flex-direction:column;gap:2px}
.exp-radio{display:flex;align-items:center;gap:8px;padding:5px 8px;
  border-radius:6px;cursor:pointer;user-select:none;font-size:12px;color:#333}
.exp-radio:hover{background:#f3f4f6}
.exp-radio input{margin:0;cursor:pointer;flex-shrink:0}
.exp-radio .exp-radio-desc{font-size:10px;color:#888;margin-left:auto}
.exp-juris-actions{display:flex;gap:10px;margin:6px 0 4px;font-size:11px}
.exp-juris-actions button{background:none;border:none;color:#2563eb;cursor:pointer;
  font-family:inherit;font-size:11px;padding:0;text-decoration:underline}
.exp-juris-actions button:disabled{color:#bbb;cursor:not-allowed;text-decoration:none}
.exp-juris-list{padding:8px 10px;border:1px solid #e5e7eb;
  border-radius:8px;background:#fafafa;max-height:160px;overflow-y:auto}
.exp-juris-list.disabled{opacity:0.5;pointer-events:none}
.exp-juris-row{display:flex;align-items:center;gap:6px;padding:2px 0}
.exp-juris-row label{display:flex;align-items:center;gap:6px;font-size:12px;
  color:#444;cursor:pointer;user-select:none;flex:1}
.exp-juris-row input{cursor:pointer;flex-shrink:0}
.exp-note{font-size:11px;color:#888;margin-top:8px;line-height:1.4}
.exp-footer{padding:12px 18px;border-top:1px solid #e5e7eb;
  display:flex;gap:8px;justify-content:flex-end;flex-shrink:0;background:#fafafa}
.exp-btn{padding:8px 14px;border-radius:8px;font-size:12px;font-weight:500;
  cursor:pointer;font-family:inherit;border:1px solid #d1d5db;background:#fff;color:#374151}
.exp-btn:hover{background:#f3f4f6}
.exp-btn-primary{background:#2563eb;color:#fff;border-color:#2563eb}
.exp-btn-primary:hover{background:#1d4ed8}
.exp-btn:disabled{opacity:0.5;cursor:not-allowed}

/* --- compare-tracts feature --- */
#compare-banner{position:absolute;top:10px;left:50%;transform:translateX(-50%);
  z-index:1001;background:#dbeafe;color:#1e3a8a;padding:8px 16px;
  border-radius:8px;font-size:13px;font-weight:500;display:none;
  box-shadow:0 2px 8px rgba(0,0,0,0.10)}
#compare-banner b{font-weight:600}
#compare-banner button{margin-left:12px;background:#2563eb;color:#fff;border:none;
  border-radius:6px;padding:3px 10px;font-size:12px;cursor:pointer;font-family:inherit}
.compare-cursor{cursor:crosshair !important}
.compare-cursor .leaflet-interactive{cursor:crosshair !important}
.popup-compare-btn{display:inline-block;margin-top:6px;background:#2563eb;color:#fff;
  border:none;border-radius:6px;padding:5px 10px;font-size:11px;font-weight:500;
  cursor:pointer;font-family:inherit}
.popup-compare-btn:hover{background:#1d4ed8}

#compare-modal{position:fixed;inset:0;z-index:2000;background:rgba(20,25,40,0.55);
  display:none;align-items:center;justify-content:center;
  backdrop-filter:blur(2px);font-family:inherit}
#compare-modal.open{display:flex}
.cmp-panel{background:#fff;width:min(1100px,96vw);height:min(720px,92vh);
  border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,0.30);
  display:flex;flex-direction:column;overflow:hidden}
.cmp-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid #e5e7eb;flex-shrink:0}
.cmp-hdr h2{font-size:15px;font-weight:700;margin:0}
.cmp-hdr .cmp-sub{font-size:11px;color:#888;margin-left:10px;font-weight:400}
.cmp-close{background:none;border:none;font-size:22px;color:#666;cursor:pointer;
  padding:0 4px;line-height:1;font-family:inherit}
.cmp-close:hover{color:#111}
.cmp-body{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:#e5e7eb;overflow:hidden}
.cmp-col{background:#fff;display:flex;flex-direction:column;overflow:hidden}
.cmp-col-hdr{padding:10px 14px;border-bottom:1px solid #e5e7eb;background:#f9fafb;
  flex-shrink:0}
.cmp-col-hdr .cmp-tract-id{font-size:13px;font-weight:600;color:#111}
.cmp-col-hdr .cmp-tract-meta{font-size:11px;color:#888;margin-top:2px}
.cmp-mini-map{height:200px;flex-shrink:0;background:#eef}
.cmp-vars{flex:1;overflow-y:auto;padding:10px 14px;font-size:12px;
  scrollbar-width:thin}
.cmp-vars::-webkit-scrollbar{width:6px}
.cmp-vars::-webkit-scrollbar-thumb{background:#ccc;border-radius:3px}
.cmp-pillar-hdr{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;padding:8px 0 4px;margin-top:8px;border-bottom:2px solid;
  color:#666}
.cmp-pillar-hdr:first-child{margin-top:0}
.cmp-sub-hdr{font-size:11px;font-weight:600;color:#666;padding:6px 0 2px;
  margin-top:4px}
.cmp-row{display:flex;justify-content:space-between;align-items:center;
  padding:3px 0;gap:8px;border-bottom:1px solid #f3f4f6}
.cmp-row .cmp-label{color:#444;flex:1;min-width:0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cmp-row .cmp-val{font-variant-numeric:tabular-nums;color:#111;flex-shrink:0;
  display:flex;flex-direction:column;align-items:flex-end;line-height:1.18}
.cmp-val-raw{font-weight:500;color:#111}
.cmp-val-pct{font-size:10px;color:#888;font-weight:400;margin-top:1px}
.cmp-row.cmp-higher .cmp-val-raw{color:#c0392b;font-weight:700}
.cmp-row.cmp-lower  .cmp-val-raw{color:#16a085}
/* Colorblind-friendly overrides (Okabe-Ito vermillion + sky blue) */
body.cb-mode .cmp-row.cmp-higher .cmp-val-raw{color:#d55e00}
body.cb-mode .cmp-row.cmp-lower  .cmp-val-raw{color:#0072b2}
.cmp-row .cmp-arrow{font-size:9px;color:#aaa;width:10px;text-align:center}
.cmp-lst-legend{display:none;align-items:center;gap:10px;padding:6px 18px;
  border-bottom:1px solid #e5e7eb;background:#fafafa;font-size:11px;color:#555}
.cmp-lst-title{font-weight:600;color:#333;flex-shrink:0}
.cmp-lst-bar{flex:1;height:10px;border-radius:2px;border:1px solid #ddd;
  min-width:120px;max-width:280px}
.cmp-lst-ticks{display:flex;justify-content:space-between;flex:1;
  max-width:280px;color:#888;font-variant-numeric:tabular-nums}
.cmp-toggle-row{display:flex;align-items:center;gap:14px;padding:6px 18px;
  border-bottom:1px solid #e5e7eb;background:#fafafa;font-size:11px;color:#555;
  flex-wrap:wrap}
.cmp-toggle{display:flex;align-items:center;gap:6px;cursor:pointer;
  font-weight:600;color:#333;user-select:none}
.cmp-toggle input{width:13px;height:13px;accent-color:#4a6fa5;cursor:pointer}
.cmp-toggle-hint{color:#888;font-weight:400}
.cmp-landuse-legend{display:none;flex-wrap:wrap;gap:6px 12px;padding:6px 18px;
  border-bottom:1px solid #e5e7eb;background:#fafafa;font-size:11px;color:#555}
.cmp-landuse-legend.open{display:flex}
.cmp-lu-item{display:flex;align-items:center;gap:5px}
.cmp-lu-swatch{width:11px;height:11px;border-radius:2px;border:1px solid rgba(0,0,0,0.18);
  flex-shrink:0}

#overlay-toggles{border-top:1px solid #e0e0e0;padding-top:10px;margin-top:10px}
#overlay-toggles h3{font-size:12px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;margin-bottom:6px;color:#666}
.ov-row{display:flex;align-items:center;gap:8px;padding:3px 0}
.ov-row label{display:flex;align-items:center;gap:6px;font-size:12px;
  color:#444;cursor:pointer;user-select:none}
.ov-row input[type=checkbox]{width:14px;height:14px;accent-color:#4a6fa5;cursor:pointer}

/* --- compare-scenarios feature --- */
#scn-cmp-btn{width:100%;padding:9px;border:1px solid #2563eb;border-radius:8px;
  background:#eff5ff;font-size:12px;font-weight:600;cursor:pointer;
  font-family:inherit;color:#1d4ed8;transition:all 0.15s;margin-top:10px}
#scn-cmp-btn:hover{background:#dbeafe}
#scn-compare-modal{position:fixed;inset:0;z-index:2000;background:rgba(20,25,40,0.55);
  display:none;align-items:center;justify-content:center;
  backdrop-filter:blur(2px);font-family:inherit}
#scn-compare-modal.open{display:flex}
.scn-cmp-panel{background:#fff;width:min(1200px,96vw);height:min(760px,94vh);
  border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,0.30);
  display:flex;flex-direction:column;overflow:hidden}
.scn-cmp-body{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:#e5e7eb;overflow:hidden;min-height:0}
.scn-cmp-col{background:#fff;display:flex;flex-direction:column;overflow:hidden;min-height:0}
.scn-cmp-picker{padding:10px 14px;border-bottom:1px solid #e5e7eb;background:#f9fafb;
  display:flex;align-items:center;gap:8px;flex-shrink:0}
.scn-cmp-picker label{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;color:#666;flex-shrink:0}
.scn-cmp-picker select{flex:1;padding:6px 8px;font-size:12px;font-family:inherit;
  border:1px solid #d1d5db;border-radius:6px;background:#fff;cursor:pointer;
  min-width:0}
.scn-cmp-map{flex:1;background:#eef;min-height:0}
.scn-cmp-meta{padding:6px 14px;font-size:11px;color:#888;border-top:1px solid #e5e7eb;
  background:#fafafa;flex-shrink:0;min-height:24px}
.scn-cmp-legend{display:flex;align-items:center;gap:18px;padding:8px 18px;
  border-top:1px solid #e5e7eb;background:#fafafa;font-size:11px;color:#555;
  flex-shrink:0;flex-wrap:wrap}
.scn-leg-block{display:flex;align-items:center;gap:8px;flex:1;min-width:260px}
.scn-leg-block .cmp-lst-bar{flex:1;height:10px;border-radius:2px;border:1px solid #ddd;
  min-width:120px;max-width:280px}
.scn-leg-block .cmp-lst-ticks{display:flex;justify-content:space-between;flex:1;
  max-width:280px;color:#888;font-variant-numeric:tabular-nums}

/* --- welcome popup --- */
#welcome-modal{position:fixed;inset:0;z-index:2300;background:rgba(20,25,40,0.55);
  display:none;align-items:center;justify-content:center;
  backdrop-filter:blur(2px);font-family:inherit}
#welcome-modal.open{display:flex}
.wlc-panel{background:#fff;width:min(520px,94vw);max-height:90vh;
  border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,0.30);
  display:flex;flex-direction:column;overflow:hidden}
.wlc-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid #e5e7eb;flex-shrink:0}
.wlc-hdr h2{font-size:15px;font-weight:700;margin:0;color:#111}
.wlc-close{background:none;border:none;font-size:22px;color:#666;cursor:pointer;
  padding:0 4px;line-height:1;font-family:inherit}
.wlc-close:hover{color:#111}
.wlc-body{padding:14px 18px;overflow-y:auto;font-size:13px;color:#333;
  flex:1;line-height:1.55}
.wlc-body p{margin:0 0 10px}
.wlc-body ul{margin:0;padding-left:18px}
.wlc-body li{margin-bottom:7px}
.wlc-body li b{color:#111}
.wlc-footer{padding:12px 18px;border-top:1px solid #e5e7eb;
  display:flex;gap:8px;justify-content:flex-end;flex-shrink:0;background:#fafafa}
.wlc-btn{padding:8px 16px;border-radius:8px;font-size:12px;font-weight:600;
  cursor:pointer;font-family:inherit;border:1px solid #2563eb;
  background:#2563eb;color:#fff}
.wlc-btn:hover{background:#1d4ed8;border-color:#1d4ed8}
</style>
</head>
<body>

<div id="sidebar">
  <h1>San Diego Heat Risk Explorer</h1>
  <div class="subtitle">Toggle variables on/off within each pillar, then
    click <b>Build Composite Risk Map</b>.</div>
  <div id="controls"></div>
  <button id="compute-btn" onclick="compute()">Build Composite Risk Map</button>
  <div id="results"></div>
  <div id="layer-select">
    <h3>Map Layer</h3>
    <div id="layer-buttons"></div>
  </div>
  <div id="overlay-toggles" style="display:none">
    <h3>Jurisdictions</h3>
    <details class="subsec" id="juris-subsec" style="margin-top:2px"><summary>Cities</summary>
      <div class="ov-row" style="margin-left:6px"><label><input type="checkbox" id="cb-juris-all" onchange="toggleJurisAll(this.checked)">Show all cities</label></div>
      <div id="juris-items" style="margin-top:2px"></div>
    </details>
    <details class="subsec" id="tribal-subsec" style="margin-top:6px"><summary>Tribal Lands</summary>
      <div class="ov-row" style="margin-left:6px"><label><input type="checkbox" id="cb-tribal-all" onchange="toggleTribalAll(this.checked)"><span style="color:#8b5a2b;font-weight:600">Show all tribal lands</span></label></div>
      <div id="tribal-items" style="margin-top:2px"></div>
    </details>
  </div>
  <div id="scenario-bar">
    <div class="scenario-label">Climate Scenario</div>
    <div class="scn-period-row">
      <button class="scn-period-btn active" data-scn-period="present" onclick="selectScnPeriod('present')">Present Day</button>
      <button class="scn-period-btn" data-scn-period="2050" data-period-block="2050" onclick="selectScnPeriod('2050')">2050</button>
      <button class="scn-period-btn" data-scn-period="2080" data-period-block="2080" onclick="selectScnPeriod('2080')">2080</button>
    </div>
    <div class="scn-ssp-panel" data-ssp-panel="2050" data-period-block="2050" style="display:none">
      <div class="period-group-label">Mid-Century (2035&ndash;2064 mean)</div>
      <div class="scenario-btns">
        <button class="scn-btn" data-scn="ssp245" data-period="2050" onclick="setScenario('ssp245','2050')">Moderate Emissions (SSP 2-4.5)</button>
        <button class="scn-btn" data-scn="ssp370" data-period="2050" onclick="setScenario('ssp370','2050')">High Emissions (SSP 3-7.0)</button>
        <button class="scn-btn" data-scn="ssp585" data-period="2050" onclick="setScenario('ssp585','2050')">Very High Emissions (SSP 5-8.5)</button>
      </div>
    </div>
    <div class="scn-ssp-panel" data-ssp-panel="2080" data-period-block="2080" style="display:none">
      <div class="period-group-label">End-of-Century (2065&ndash;2094 mean)</div>
      <div class="scenario-btns">
        <button class="scn-btn" data-scn="ssp245" data-period="2080" onclick="setScenario('ssp245','2080')">Moderate Emissions (SSP 2-4.5)</button>
        <button class="scn-btn" data-scn="ssp370" data-period="2080" onclick="setScenario('ssp370','2080')">High Emissions (SSP 3-7.0)</button>
        <button class="scn-btn" data-scn="ssp585" data-period="2080" onclick="setScenario('ssp585','2080')">Very High Emissions (SSP 5-8.5)</button>
      </div>
    </div>
    <div id="scenario-info" class="scenario-info"></div>
    <button id="scn-cmp-btn" onclick="openScnCompareModal()" title="Open a side-by-side composite-risk map (left) plus Heat Exposure change (right) under two climate scenarios">Compare Scenarios Side-by-Side</button>
  </div>
  <div id="accessibility" style="border-top:1px solid #e0e0e0;padding-top:10px;margin-top:14px">
    <h3 style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;color:#666">Accessibility</h3>
    <div class="ov-row"><label title="Swap red/green palettes for Okabe-Ito + viridis (deuteranopia/protanopia/tritanopia safe)"><input type="checkbox" id="cb-cbmode" onchange="toggleCbMode(this.checked)">Colorblind-friendly colors</label></div>
  </div>
  <div id="export-section" style="border-top:1px solid #e0e0e0;padding-top:10px;margin-top:14px">
    <h3 style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;color:#666">Export</h3>
    <button id="dl-pdf-btn" onclick="downloadMapPdf()" style="width:100%;padding:9px;border:1px solid #d1d5db;border-radius:8px;background:#fff;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;color:#374151;transition:all 0.15s">Download Map as PDF</button>
    <button id="dl-data-btn" onclick="openExportModal()" style="width:100%;padding:9px;border:1px solid #d1d5db;border-radius:8px;background:#fff;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;color:#374151;transition:all 0.15s">Download Data</button>
  </div>
</div>

<div id="map-wrap">
  <div id="map"></div>
  <div id="logos">
    <img src="../assets/usd-logo-horizontal.png" alt="University of San Diego">
    <img src="../assets/ccsd_logo.png" alt="Climate Collaborative — San Diego Region">
  </div>
  <div id="logo-sio">
    <img src="../assets/sio_logo.png" alt="UC San Diego — Scripps Institution of Oceanography">
  </div>
  <div id="legend"></div>
  <div id="prompt">Select variables and click Build Composite Risk Map</div>
  <div id="error-bar"></div>
  <div id="compare-banner">
    <span><b>Compare mode:</b> click a second tract to compare with <span id="compare-first-id">—</span>.</span>
    <button onclick="cancelCompare()">Cancel (ESC)</button>
  </div>
</div>

<div id="pdf-overlay">
  <div class="box"><div class="spinner"></div><span id="pdf-overlay-msg">Generating PDF\\u2026</span></div>
</div>

<div id="export-modal" onclick="if(event.target===this)closeExportModal()">
  <div class="exp-panel">
    <div class="exp-hdr">
      <h2>Download Data</h2>
      <button class="exp-close" onclick="closeExportModal()" title="Close (ESC)">&times;</button>
    </div>
    <div class="exp-body">
      <div class="exp-section">
        <div class="exp-section-label">File Format</div>
        <div class="exp-radios">
          <label class="exp-radio"><input type="radio" name="exp-format" value="csv" checked>CSV<span class="exp-radio-desc">spreadsheet (Excel, pandas)</span></label>
          <label class="exp-radio"><input type="radio" name="exp-format" value="geojson">GeoJSON<span class="exp-radio-desc">GIS (QGIS, ArcGIS)</span></label>
        </div>
      </div>
      <div class="exp-section">
        <div class="exp-section-label">Columns</div>
        <label class="exp-radio" style="cursor:pointer"><input type="checkbox" id="exp-incl-pctile" checked>Include within-county percentile ranks<span class="exp-radio-desc">adds <code>&lt;var&gt;_sd_pctile</code></span></label>
        <div class="exp-note">Raw source values are always included. Percentile ranks (0&ndash;100) are computed within San Diego County and match the values used by the map and PCA.</div>
      </div>
      <div class="exp-section">
        <div class="exp-section-label">Scope</div>
        <div class="exp-radios">
          <label class="exp-radio"><input type="radio" name="exp-scope" value="all" checked onchange="_expSyncScope()">Entire map<span class="exp-radio-desc" id="exp-all-count"></span></label>
          <label class="exp-radio"><input type="radio" name="exp-scope" value="juris" onchange="_expSyncScope()">Selected jurisdictions</label>
        </div>
        <div class="exp-juris-actions">
          <button type="button" id="exp-juris-all" onclick="_expJurisSelectAll(true)">Select all</button>
          <button type="button" id="exp-juris-none" onclick="_expJurisSelectAll(false)">Clear</button>
        </div>
        <div class="exp-juris-list" id="exp-juris-list"></div>
        <div class="exp-note">Tracts whose centroid falls inside a selected jurisdiction are included.</div>
      </div>
    </div>
    <div class="exp-footer">
      <button class="exp-btn" onclick="closeExportModal()">Cancel</button>
      <button class="exp-btn exp-btn-primary" id="exp-download-btn" onclick="doExportDownload()">Download</button>
    </div>
  </div>
</div>

<div id="compare-modal" onclick="if(event.target===this)closeCompareModal()">
  <div class="cmp-panel">
    <div class="cmp-hdr">
      <h2>Census Tract Comparison<span class="cmp-sub" id="cmp-context"></span></h2>
      <button class="cmp-close" onclick="closeCompareModal()" title="Close (ESC)">&times;</button>
    </div>
    <div id="cmp-lst-legend" class="cmp-lst-legend"></div>
    <div class="cmp-toggle-row">
      <label class="cmp-toggle"><input type="checkbox" id="cmp-toggle-landuse" onchange="toggleLanduse(this.checked)">Show SANDAG land use parcels</label>
      <span class="cmp-toggle-hint">click any parcel for category</span>
    </div>
    <div id="cmp-landuse-legend" class="cmp-landuse-legend"></div>
    <div class="cmp-body">
      <div class="cmp-col">
        <div class="cmp-col-hdr">
          <div class="cmp-tract-id" id="cmp-tract-a-id">—</div>
          <div class="cmp-tract-meta" id="cmp-tract-a-meta"></div>
        </div>
        <div class="cmp-mini-map" id="cmp-map-a"></div>
        <div class="cmp-vars" id="cmp-vars-a"></div>
      </div>
      <div class="cmp-col">
        <div class="cmp-col-hdr">
          <div class="cmp-tract-id" id="cmp-tract-b-id">—</div>
          <div class="cmp-tract-meta" id="cmp-tract-b-meta"></div>
        </div>
        <div class="cmp-mini-map" id="cmp-map-b"></div>
        <div class="cmp-vars" id="cmp-vars-b"></div>
      </div>
    </div>
  </div>
</div>

<div id="scn-compare-modal" onclick="if(event.target===this)closeScnCompareModal()">
  <div class="cmp-panel scn-cmp-panel">
    <div class="cmp-hdr">
      <h2>Compare Scenarios<span class="cmp-sub">Left = Composite Risk &middot; Right = Δ Heat Exposure (B − A)</span></h2>
      <button class="cmp-close" onclick="closeScnCompareModal()" title="Close (ESC)">&times;</button>
    </div>
    <div class="scn-cmp-body">
      <div class="scn-cmp-col">
        <div class="scn-cmp-picker">
          <label>Left</label>
          <select id="scn-cmp-sel-a" onchange="setScnCompare('a',this.value)"></select>
        </div>
        <div class="scn-cmp-map" id="scn-cmp-map-a"></div>
        <div class="scn-cmp-meta" id="scn-cmp-meta-a"></div>
      </div>
      <div class="scn-cmp-col">
        <div class="scn-cmp-picker">
          <label>Right (Δ)</label>
          <select id="scn-cmp-sel-b" onchange="setScnCompare('b',this.value)"></select>
        </div>
        <div class="scn-cmp-map" id="scn-cmp-map-b"></div>
        <div class="scn-cmp-meta" id="scn-cmp-meta-b"></div>
      </div>
    </div>
    <div class="scn-cmp-legend" id="scn-cmp-legend"></div>
  </div>
</div>

<script>
/* ================================================================
   DATA  (injected by build script)
   ================================================================ */
var GJ  = __GEOJSON_PLACEHOLDER__;
var VM  = __VARMETA_PLACEHOLDER__;
var VSRC = __VARSOURCES_PLACEHOLDER__;
var DELTAS = __DELTAS_PLACEHOLDER__;         // {period:{ssp:{tasmax:[...],tasmin:[...]}}} or null
var DELTA_MAP = __DELTAMAP_PLACEHOLDER__;    // {varKey:"tasmax"|"tasmin"}
var JURIS = __JURIS_PLACEHOLDER__;           // city jurisdiction GeoJSON or null
var TRIBAL = __TRIBAL_PLACEHOLDER__;         // tribal lands GeoJSON or null
var LST   = __LST_PLACEHOLDER__;             // Landsat LST overlay meta or null
var POLARITY = __POLARITY_PLACEHOLDER__;     // {varKey: true if higher=worse(red)}

var PMETA = {
  exposure:    {label:"Heat Exposure",     color:"#c0392b"},
  sensitivity: {label:"Population Sensitivity", color:"#d4a017"},
  adaptive:    {label:"Adaptive Capacity", color:"#16a085"}
};
var SREF = {exposure:"mean_tmax", sensitivity:"bphigh_prev", adaptive:"income_pctile"};

/* ================================================================
   STATE
   ================================================================ */
var enabled = {};   // varKey -> bool
var curLayer = "composite";
var rawVar   = null; // key of raw variable being previewed (before PCA)
var results  = null;
var curScenario = "present"; // "present" | "ssp245" | "ssp370" | "ssp585"
var curPeriod   = "";        // "" (present) | "2050" | "2080"
var cbMode = (function(){try{return localStorage.getItem('hrv-cb-mode')==='1';}catch(e){return false;}})();
VM.forEach(function(v){ enabled[v.key] = true; });

// Pre-compute percentile ranks for every raw variable (for preview rendering)
var rawPct = {}; // varKey -> array of percentile values (0-100), indexed by feature
// Pre-compute present-day min/max per variable (fixed color scale for scenario comparison)
var rawRange = {}; // varKey -> {min, max}
VM.forEach(function(v){
  var vals=GJ.features.map(function(f){return f.properties[v.key]||0;});
  rawPct[v.key]=pctRank(vals);
  var mn=vals[0],mx=vals[0];
  for(var i=1;i<vals.length;i++){if(vals[i]<mn)mn=vals[i];if(vals[i]>mx)mx=vals[i];}
  rawRange[v.key]={min:mn,max:mx};
});
// Also build a lookup: varKey -> pillar
var varPillar={};VM.forEach(function(v){varPillar[v.key]=v.pillar;});
// And varKey -> label
var varLabel={};VM.forEach(function(v){varLabel[v.key]=v.label;});

// Show scenario bar if CMIP6 deltas are available
if(DELTAS){
  document.getElementById('scenario-bar').style.display='block';
  // Hide period blocks for which we have no data (e.g. 2080 not yet downloaded)
  document.querySelectorAll('[data-period-block]').forEach(function(el){
    var p=el.getAttribute('data-period-block');
    if(!DELTAS[p]) el.style.display='none';
  });
}

/* ================================================================
   SCENARIO
   ================================================================ */
// Top-level period buttons: Present Day applies immediately; 2050/2080 just
// expand their SSP picker (user picks an SSP, which calls setScenario directly).
function selectScnPeriod(period){
  if(period==='present'){
    setScenario('present','');
    return;
  }
  var panel=document.querySelector('[data-ssp-panel="'+period+'"]');
  var btn=document.querySelector('.scn-period-btn[data-scn-period="'+period+'"]');
  if(!panel||!btn)return;
  document.querySelectorAll('.scn-ssp-panel').forEach(function(p){
    if(p!==panel)p.style.display='none';
  });
  document.querySelectorAll('.scn-period-btn').forEach(function(b){
    if(b!==btn)b.classList.remove('expanded');
  });
  var open=panel.style.display==='block';
  panel.style.display=open?'none':'block';
  btn.classList.toggle('expanded',!open);
}

function setScenario(ssp,period){
  curScenario=ssp;
  curPeriod=(ssp==='present')?'':(period||'');
  window._adjPct={};  // invalidate adjusted percentile cache
  document.querySelectorAll('.scn-btn').forEach(function(b){
    var match=(b.getAttribute('data-scn')===ssp)&&
              ((b.getAttribute('data-period')||'')===curPeriod);
    b.classList.toggle('active',match);
  });
  var activeKey=(ssp==='present')?'present':curPeriod;
  document.querySelectorAll('.scn-period-btn').forEach(function(b){
    b.classList.toggle('active',b.getAttribute('data-scn-period')===activeKey);
    b.classList.remove('expanded');
  });
  document.querySelectorAll('.scn-ssp-panel').forEach(function(p){
    p.style.display=(p.getAttribute('data-ssp-panel')===activeKey)?'block':'none';
  });
  var labels={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'};
  var pwin={'2050':'2035\\u20132064','2080':'2065\\u20132094'};
  var info=document.getElementById('scenario-info');
  if(ssp==='present'){info.textContent='';}
  else{info.textContent=labels[ssp]+' projection, LOCA2 ensemble mean, '+pwin[curPeriod];}
  updateFixedBadges();
  updateDeltaRange();
  if(results){compute();}
  else if(rawVar){drawTracts();}
}

function updateFixedBadges(){
  VM.forEach(function(v){
    var badge=document.getElementById('fix_'+v.key);
    if(!badge)return;
    badge.style.display=(curScenario!=='present'&&v.pillar==='exposure'&&!DELTA_MAP[v.key])?'inline':'none';
  });
}

// Look up the per-tract delta array for a variable, given current period+scenario.
// Returns null if not applicable.
function getDeltaArr(varKey){
  if(curScenario==='present'||!DELTAS||!curPeriod)return null;
  var perPeriod=DELTAS[curPeriod];
  if(!perPeriod)return null;
  var perSsp=perPeriod[curScenario];
  if(!perSsp)return null;
  var loca=DELTA_MAP[varKey];
  if(!loca)return null;
  return perSsp[loca]||null;
}

// Get adjusted value for a variable at feature index i
function getAdj(varKey,i){
  var v=GJ.features[i].properties[varKey]||0;
  var d=getDeltaArr(varKey);
  if(d)v+=d[i];
  return v;
}

// Format a raw variable value with appropriate units for display in the
// compare-tracts modal. Knows about temperature (°C), 0–1 fractions
// (probabilities, AC-rank), pre-computed percentile vars (HPI _pctile, SVI
// RPL_*) and CDC PLACES / SVI EP_* / pct_* / humidity values that are stored
// as 0–100 percentages.
var _CMP_TEMP_KEYS={mean_tmax:1,p85_tmax:1,p99_tmax:1,max_tmax:1,
  mean_lst:1,p85_wbgt:1,p99_wbgt:1,
  mean_tmin_overnight:1,p90_tmin_overnight:1,p95_dewpoint:1};
var _CMP_FRAC_KEYS={no_relief_fraction:1,ac_prob:1,prob_NoAC:1,
  prob_Central:1};
function fmtCmpRaw(key,val){
  if(val==null||isNaN(val))return '\\u2014';
  var v=+val;
  if(_CMP_TEMP_KEYS[key])return v.toFixed(1)+'\\u00b0C';
  if(_CMP_FRAC_KEYS[key])return (v*100).toFixed(1)+'%';
  if(key.indexOf('RPL_')===0)return v.toFixed(2);
  if(/_pctile$/.test(key))return v.toFixed(1);
  return v.toFixed(1)+'%';
}

// Compute and cache delta min/max for current rawVar + scenario + period
function updateDeltaRange(){
  window._deltaRange=null;
  if(!rawVar)return;
  var darr=getDeltaArr(rawVar);
  if(!darr)return;
  var mn=darr[0],mx=darr[0];
  for(var i=1;i<darr.length;i++){if(darr[i]<mn)mn=darr[i];if(darr[i]>mx)mx=darr[i];}
  window._deltaRange={min:mn,max:mx};
}

/* ================================================================
   MAP
   ================================================================ */
var map = L.map("map").setView([32.85,-117.05],10);
var basemapLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{
  attribution:"CARTO", maxZoom:18}).addTo(map);
L.control.scale().addTo(map);
var tLayer = null;

// City jurisdiction boundary overlays — one layer per city, all rendered in black
var jurisLayers={};
var JURIS_COLOR="#000000";
function jurisCheckboxId(name){return "cb-juris-"+name.replace(/\s+/g,"_");}
if(JURIS){
  function makeJurisStyle(){
    return function(){return{fillColor:"transparent",color:JURIS_COLOR,weight:1.5,fillOpacity:0,dashArray:""}};
  }
  function makeJurisHL(name){
    return function(f,lyr){
      lyr.on("mouseover",function(e){e.target.setStyle({weight:3,color:JURIS_COLOR,fillOpacity:0.05})});
      lyr.on("mouseout",function(e){if(jurisLayers[name])jurisLayers[name].resetStyle(e.target)});
      lyr.on("click",function(e){
        if(L.DomEvent)L.DomEvent.stopPropagation(e);
        // If a tract-compare is in progress, juris polygons are sitting on top of
        // the tract layer and would otherwise swallow the click. Forward to the
        // tract beneath the click point.
        if(compareMode==='awaiting'){
          var pt=[e.latlng.lng,e.latlng.lat];
          for(var i=0;i<GJ.features.length;i++){
            if(_ptInGeometry(pt,GJ.features[i].geometry)){
              var gid=GJ.features[i].properties.GEOID;
              if(gid!==compareFirst)finishCompare(gid);
              return;
            }
          }
          return;
        }
        openCityCompareModal(name);
      });
      lyr.bindTooltip(f.properties.NAME+" \\u2014 click to compare",{sticky:true,className:"juris-tip"});
    };
  }
  // One layer per city, alphabetically sorted
  var jurisNames=JURIS.features.map(function(f){return f.properties.NAME;})
    .filter(function(n,i,a){return n&&a.indexOf(n)===i;})
    .sort(function(a,b){return a.localeCompare(b);});
  jurisNames.forEach(function(name){
    var feats=JURIS.features.filter(function(f){return f.properties.NAME===name;});
    if(feats.length===0)return;
    var gj={type:"FeatureCollection",features:feats};
    jurisLayers[name]=L.geoJSON(gj,{style:makeJurisStyle(),onEachFeature:makeJurisHL(name)});
  });
  // Build per-city checkbox list
  var itemsEl=document.getElementById("juris-items");
  if(itemsEl){
    itemsEl.innerHTML=jurisNames.map(function(name){
      var id=jurisCheckboxId(name);
      var nameAttr=name.replace(/"/g,"&quot;");
      return '<div class="ov-row" style="margin-left:14px"><label>'+
        '<input type="checkbox" id="'+id+'" data-juris-name="'+nameAttr+'" onchange="toggleJurisLayer(this.dataset.jurisName)">'+
        name+
        '</label></div>';
    }).join("");
  }
  document.getElementById("overlay-toggles").style.display="block";
}

function toggleJurisLayer(name){
  var lyr=jurisLayers[name];
  if(!lyr)return;
  var cb=document.getElementById(jurisCheckboxId(name));
  if(cb&&cb.checked){lyr.addTo(map);lyr.bringToFront();}
  else{map.removeLayer(lyr);}
  syncJurisAllCheckbox();
}
function toggleJurisAll(on){
  Object.keys(jurisLayers).forEach(function(name){
    var cb=document.getElementById(jurisCheckboxId(name));
    if(cb)cb.checked=!!on;
    var lyr=jurisLayers[name];
    if(!lyr)return;
    if(on){lyr.addTo(map);lyr.bringToFront();}
    else{map.removeLayer(lyr);}
  });
}
function syncJurisAllCheckbox(){
  var all=document.getElementById("cb-juris-all");
  if(!all)return;
  var names=Object.keys(jurisLayers);
  if(!names.length){all.checked=false;all.indeterminate=false;return;}
  var n=0;
  names.forEach(function(nm){if(map.hasLayer(jurisLayers[nm]))n++;});
  all.checked=(n===names.length);
  all.indeterminate=(n>0&&n<names.length);
}

// Tribal lands overlays — Census TIGER AIANNH 2024 (San Diego County)
// Reservations and off-reservation trust lands are paired under one tribe entry;
// the map distinguishes them with a solid (reservation) vs dashed (trust) outline.
var tribalLayers={};      // keyed by full feature name (each polygon)
var tribalGroups={};      // tribeKey -> [feature names]
var TRIBAL_COLORS={reservation:"#8b5a2b",trust:"#b78050"};
function tribalGroupKey(name){
  return name.replace(/ Reservation$/,"").replace(/ Off-Reservation Trust Land$/,"");
}
function tribalCheckboxId(groupKey){return "cb-tribal-"+groupKey.replace(/\s+/g,"_");}
if(typeof TRIBAL!=='undefined'&&TRIBAL&&TRIBAL.features&&TRIBAL.features.length){
  function makeTribalStyle(type){
    var c=TRIBAL_COLORS[type]||TRIBAL_COLORS.reservation;
    var dash=(type==='trust')?"5,4":"";
    return function(){return{fillColor:c,color:c,weight:1.5,fillOpacity:0.10,dashArray:dash}};
  }
  function makeTribalHL(name){
    return function(f,lyr){
      var t=f.properties.TYPE||'reservation';
      var c=TRIBAL_COLORS[t]||TRIBAL_COLORS.reservation;
      lyr.on("mouseover",function(e){e.target.setStyle({weight:3,color:c,fillOpacity:0.22})});
      lyr.on("mouseout",function(e){if(tribalLayers[name])tribalLayers[name].resetStyle(e.target)});
      lyr.on("click",function(e){
        if(L.DomEvent)L.DomEvent.stopPropagation(e);
        // Forward to underlying tract during compare-awaiting (same reason as juris)
        if(compareMode==='awaiting'){
          var pt=[e.latlng.lng,e.latlng.lat];
          for(var i=0;i<GJ.features.length;i++){
            if(_ptInGeometry(pt,GJ.features[i].geometry)){
              var gid=GJ.features[i].properties.GEOID;
              if(gid!==compareFirst)finishCompare(gid);
              return;
            }
          }
          return;
        }
        showTribalNoData(e.latlng,f.properties.NAME,t);
      });
      var label=f.properties.NAME+(t==='trust'?" (Trust Land)":"");
      lyr.bindTooltip(label,{sticky:true,className:"juris-tip"});
    };
  }
  TRIBAL.features.forEach(function(f){
    var name=f.properties.NAME;
    var gj={type:"FeatureCollection",features:[f]};
    tribalLayers[name]=L.geoJSON(gj,{
      style:makeTribalStyle(f.properties.TYPE),
      onEachFeature:makeTribalHL(name)
    });
    var gk=tribalGroupKey(name);
    if(!tribalGroups[gk])tribalGroups[gk]=[];
    tribalGroups[gk].push(name);
  });
  // Build one checkbox per tribe (group), sorted alphabetically
  var tribalGroupKeys=Object.keys(tribalGroups).sort(function(a,b){return a.localeCompare(b);});
  var itemsEl=document.getElementById("tribal-items");
  if(itemsEl){
    itemsEl.innerHTML=tribalGroupKeys.map(function(gk){
      var id=tribalCheckboxId(gk);
      var keyAttr=gk.replace(/"/g,"&quot;");
      return '<div class="ov-row" style="margin-left:14px"><label>'+
        '<input type="checkbox" id="'+id+'" data-tribal-key="'+keyAttr+'" onchange="toggleTribalLayer(this.dataset.tribalKey)">'+
        '<span style="color:'+TRIBAL_COLORS.reservation+'">'+gk+'</span>'+
        '</label></div>';
    }).join("");
  }
  document.getElementById("overlay-toggles").style.display="block";
}
function toggleTribalLayer(groupKey){
  var feats=tribalGroups[groupKey];
  if(!feats)return;
  var cb=document.getElementById(tribalCheckboxId(groupKey));
  var on=!!(cb&&cb.checked);
  feats.forEach(function(name){
    var lyr=tribalLayers[name];
    if(!lyr)return;
    if(on){lyr.addTo(map);lyr.bringToFront();}
    else{map.removeLayer(lyr);}
  });
  syncTribalAllCheckbox();
}
function toggleTribalAll(on){
  Object.keys(tribalGroups).forEach(function(gk){
    var cb=document.getElementById(tribalCheckboxId(gk));
    if(cb)cb.checked=!!on;
    tribalGroups[gk].forEach(function(name){
      var lyr=tribalLayers[name];
      if(!lyr)return;
      if(on){lyr.addTo(map);lyr.bringToFront();}
      else{map.removeLayer(lyr);}
    });
  });
}
function syncTribalAllCheckbox(){
  var all=document.getElementById("cb-tribal-all");
  if(!all)return;
  var keys=Object.keys(tribalGroups);
  if(!keys.length){all.checked=false;all.indeterminate=false;return;}
  var n=0;
  keys.forEach(function(gk){
    var anyOn=tribalGroups[gk].some(function(name){return map.hasLayer(tribalLayers[name]);});
    if(anyOn)n++;
  });
  all.checked=(n===keys.length);
  all.indeterminate=(n>0&&n<keys.length);
}

/* ================================================================
   PCA  — Jacobi eigendecomposition
   ================================================================ */
function jacobiEigen(A){
  var n=A.length, i, j, k, p, q;
  if(n===0) return {ev:[],V:[]};
  if(n===1) return {ev:[A[0][0]],V:[[1]]};
  var S=[],V=[];
  for(i=0;i<n;i++){S[i]=A[i].slice();V[i]=[];for(j=0;j<n;j++)V[i][j]=(i===j)?1:0;}
  for(var sw=0;sw<100;sw++){
    var off=0;
    for(i=0;i<n;i++)for(j=i+1;j<n;j++)off+=S[i][j]*S[i][j];
    if(off<1e-20)break;
    for(p=0;p<n-1;p++)for(q=p+1;q<n;q++){
      if(Math.abs(S[p][q])<1e-15)continue;
      var tau=(S[q][q]-S[p][p])/(2*S[p][q]),t;
      if(Math.abs(tau)>1e15)t=1/(2*tau);
      else t=(tau>=0?1:-1)/(Math.abs(tau)+Math.sqrt(1+tau*tau));
      var c=1/Math.sqrt(1+t*t),s=t*c,Spq=S[p][q];
      S[p][p]-=t*Spq;S[q][q]+=t*Spq;S[p][q]=S[q][p]=0;
      for(var r=0;r<n;r++){if(r===p||r===q)continue;
        var rp=S[r][p],rq=S[r][q];
        S[r][p]=S[p][r]=c*rp-s*rq;S[r][q]=S[q][r]=s*rp+c*rq;}
      for(i=0;i<n;i++){var vp=V[i][p],vq=V[i][q];V[i][p]=c*vp-s*vq;V[i][q]=s*vp+c*vq;}
    }
  }
  var ev=[];for(i=0;i<n;i++)ev.push(S[i][i]);
  return {ev:ev,V:V};
}

function corrXY(x,y){
  var n=x.length,mx=0,my=0,i;
  for(i=0;i<n;i++){mx+=x[i];my+=y[i];}mx/=n;my/=n;
  var num=0,dx=0,dy=0;
  for(i=0;i<n;i++){var a=x[i]-mx,b=y[i]-my;num+=a*b;dx+=a*a;dy+=b*b;}
  return num/(Math.sqrt(dx*dy)+1e-15);
}

function pctRank(arr){
  var n=arr.length,idx=[],i;
  for(i=0;i<n;i++)idx.push(i);
  idx.sort(function(a,b){return arr[a]-arr[b]});
  var ranks=new Array(n);i=0;
  while(i<n){var j=i;
    while(j<n-1&&arr[idx[j+1]]===arr[idx[j]])j++;
    var avg=(i+j)/2+1;for(var k=i;k<=j;k++)ranks[idx[k]]=Math.round(avg/n*1000)/10;
    i=j+1;}
  return ranks;
}

function pillarPCA(pid,keys){
  var feats=GJ.features,N=feats.length,P=keys.length,i,j,k;
  if(P===0)return null;
  var X=[];
  for(i=0;i<N;i++){X[i]=[];for(j=0;j<P;j++){
    var v=feats[i].properties[keys[j]];if(v==null)v=0;
    var d=getDeltaArr(keys[j]);
    if(d)v+=d[i];
    X[i][j]=v;}}
  // z-score
  var mu=[],sd=[];
  for(j=0;j<P;j++){mu[j]=0;for(i=0;i<N;i++)mu[j]+=X[i][j];mu[j]/=N;
    sd[j]=0;for(i=0;i<N;i++)sd[j]+=(X[i][j]-mu[j])*(X[i][j]-mu[j]);
    sd[j]=Math.sqrt(sd[j]/(N-1));if(sd[j]<1e-10)sd[j]=1;
    for(i=0;i<N;i++)X[i][j]=(X[i][j]-mu[j])/sd[j];}
  // Sign-align each column to its pillar's risk semantics so vulnerability
  // variables (INVERTED_VARS) point the same way as their peers. Without this,
  // a P=1 selection of e.g. prob_NoAC would skip alignment and end up with the
  // 100-s adaptive inversion applied to an already-inverted score. flip[j] is
  // ±1 — used later to convert PCA loadings back to original-variable space
  // for tooltip readability.
  var pillarHigherIsRisk=(pid==="exposure"||pid==="sensitivity");
  var flip=[];
  for(j=0;j<P;j++){
    var keyHigherIsRisk=POLARITY[keys[j]];
    if(keyHigherIsRisk!==pillarHigherIsRisk){
      flip.push(-1);
      for(i=0;i<N;i++)X[i][j]=-X[i][j];
    }else{
      flip.push(1);
    }
  }
  if(P===1){var raw=[];for(i=0;i<N;i++)raw.push(X[i][0]);
    return{scores:pctRank(raw),pcs:[{scores:pctRank(raw),pct:100,loadings:[flip[0]]}],nret:1,totvar:100,keys:keys};}
  // cov — init all rows first so C[j][i] works when j>i
  var C=[];for(i=0;i<P;i++){C[i]=[];for(j=0;j<P;j++)C[i][j]=0;}
  for(i=0;i<P;i++)for(j=i;j<P;j++){var s=0;for(k=0;k<N;k++)s+=X[k][i]*X[k][j];C[i][j]=C[j][i]=s/(N-1);}
  var eig=jacobiEigen(C);
  // sort desc
  var ord=[];for(i=0;i<P;i++)ord.push(i);
  ord.sort(function(a,b){return eig.ev[b]-eig.ev[a]});
  var sev=ord.map(function(i){return eig.ev[i]});
  var tot=0;for(i=0;i<P;i++)tot+=Math.max(sev[i],0);
  var vr=sev.map(function(v){return Math.max(v,0)/tot});
  var nret=0;for(i=0;i<P;i++)if(sev[i]>1)nret++;nret=Math.max(1,nret);
  // project
  var sc=[];for(i=0;i<N;i++){sc[i]=[];for(k=0;k<P;k++){
    var s=0,ei=ord[k];for(j=0;j<P;j++)s+=X[i][j]*eig.V[j][ei];sc[i][k]=s;}}
  // sign ref
  var ref=SREF[pid],ri=keys.indexOf(ref);if(ri<0)ri=0;
  var rc=[];for(i=0;i<N;i++)rc.push(X[i][ri]);
  // weighted composite
  var wts=vr.slice(0,nret),ws=0;for(i=0;i<wts.length;i++)ws+=wts[i];
  var w=new Array(N);for(i=0;i<N;i++){w[i]=0;for(k=0;k<nret;k++)w[i]+=sc[i][k]*wts[k]/ws;}
  if(corrXY(rc,w)<0)for(i=0;i<N;i++)w[i]=-w[i];
  // individual PCs (also capture loadings, sign-aligned with scores)
  var nex=Math.min(Math.max(3,nret),P),pcs=[];
  for(k=0;k<nex;k++){
    var raw=[];for(i=0;i<N;i++)raw.push(sc[i][k]);
    var ld=[];for(j=0;j<P;j++)ld.push(eig.V[j][ord[k]]);
    if(corrXY(rc,raw)<0){
      for(i=0;i<N;i++)raw[i]=-raw[i];
      for(j=0;j<P;j++)ld[j]=-ld[j];
    }
    // Convert loadings back to original-variable space for the "Driven by" tooltip:
    // a + sign should mean "high original value pushes the pattern up."
    var ldOrig=[];for(j=0;j<P;j++)ldOrig.push(ld[j]*flip[j]);
    pcs.push({scores:pctRank(raw),pct:Math.round(vr[k]*1000)/10,loadings:ldOrig});
  }
  var tv=0;for(i=0;i<nret;i++)tv+=vr[i];
  return{scores:pctRank(w),pcs:pcs,nret:nret,totvar:Math.round(tv*1000)/10,keys:keys};
}

/* ================================================================
   COMPUTE
   ================================================================ */
function compute(){
  try{
    document.getElementById("prompt").style.display="none";
    document.getElementById("error-bar").style.display="none";
    // collect enabled vars per pillar
    var pv={exposure:[],sensitivity:[],adaptive:[]};
    VM.forEach(function(v){if(enabled[v.key])pv[v.pillar].push(v.key);});
    var total=pv.exposure.length+pv.sensitivity.length+pv.adaptive.length;
    console.log("Pillar vars:",JSON.stringify({e:pv.exposure.length,s:pv.sensitivity.length,a:pv.adaptive.length}));
    if(total===0){document.getElementById("prompt").style.display="flex";
      document.getElementById("prompt").textContent="Turn on at least one variable, then click Build Composite Risk Map";return;}
    rawVar=null; // switch from raw preview to PCA results mode
    // PCA
    var res={};
    ["exposure","sensitivity","adaptive"].forEach(function(pid){res[pid]=pillarPCA(pid,pv[pid]);});
    // write to features
    var N=GJ.features.length;
    for(var i=0;i<N;i++){
      var pr=GJ.features[i].properties,sum=0,cnt=0;
      ["exposure","sensitivity","adaptive"].forEach(function(pid){
        if(!res[pid])return;
        pr["_"+pid]=res[pid].scores[i];
        for(var k=0;k<res[pid].pcs.length;k++)
          pr["_"+pid+"_pc"+(k+1)]=res[pid].pcs[k].scores[i];
        var s=res[pid].scores[i];
        if(pid==="adaptive")s=100-s;
        sum+=s;cnt++;
      });
      pr._comp=cnt>0?Math.round(sum/cnt*10)/10:50;
      var eS=res.exposure?res.exposure.scores[i]:0;
      var sS=res.sensitivity?res.sensitivity.scores[i]:0;
      var aD=res.adaptive?(100-res.adaptive.scores[i]):0;
      pr._drv=(eS>=sS&&eS>=aD)?"exposure":(sS>=aD)?"sensitivity":"adaptive";
      pr._drvI=pr._comp/100;
    }
    results=res;
    drawTracts();
    showResults();
    showLayerBtns();
    console.log("Compute done, features:",N);
  }catch(err){
    console.error(err);
    var eb=document.getElementById("error-bar");
    eb.textContent="Error: "+err.message;
    eb.style.display="block";
  }
}

/* ================================================================
   COLORS
   ================================================================ */
function lerp(a,b,t){return a+(b-a)*t}
function interpC(cs,t){t=Math.max(0,Math.min(1,t));
  var idx=t*(cs.length-1),lo=Math.floor(idx),hi=Math.min(lo+1,cs.length-1),f=idx-lo;
  return[Math.round(lerp(cs[lo][0],cs[hi][0],f)),Math.round(lerp(cs[lo][1],cs[hi][1],f)),Math.round(lerp(cs[lo][2],cs[hi][2],f))];}
function toHex(rgb){return"#"+rgb.map(function(c){return Math.max(0,Math.min(255,c)).toString(16).padStart(2,"0")}).join("")}

// ── Default palettes (ColorBrewer sequential — individually CB-safe; the
//    pillar set still uses red/orange/green together which can be confusing)
var CR=[[255,255,178],[254,204,92],[253,141,60],[240,59,32],[189,0,38]];
var CE=[[254,229,217],[252,174,145],[251,106,74],[203,24,29]];
var CS=[[242,240,247],[203,201,226],[158,154,200],[106,81,163]];
var CA=[[229,245,224],[161,217,155],[65,171,93],[0,109,44]];
var CD=[[255,255,204],[255,237,160],[254,178,76],[253,141,60],[252,78,42],[227,26,28],[177,0,38]];
var DRGB={exposure:[192,57,43],sensitivity:[212,160,23],adaptive:[22,160,82]};

// ── Colorblind-friendly palettes ─────────────────────────────────────────
// Composite + delta: viridis (perceptually uniform, CB-safe across all forms)
// Pillars: Okabe-Ito hue families (vermillion, yellow, blue) — each varied
// in lightness so any single pillar reads as a low-to-high sequential ramp.
var CR_CB=[[68,1,84],[59,82,139],[33,144,140],[94,201,98],[253,231,37]];
var CE_CB=[[253,221,182],[249,180,90],[228,123,15],[180,68,4]];   // vermillion family
var CS_CB=[[247,231,189],[235,196,108],[210,150,30],[150,100,12]];// yellow family
var CA_CB=[[211,229,243],[124,182,221],[40,118,189],[0,72,138]];  // blue family
var CD_CB=[[253,231,37],[154,213,87],[33,180,124],[24,143,141],[55,100,144],[68,49,124],[68,1,84]].reverse();
var DRGB_CB={exposure:[213,94,0],sensitivity:[230,159,0],adaptive:[0,114,178]};

var PCC={exposure:CE,sensitivity:CS,adaptive:CA};
var PCC_CB={exposure:CE_CB,sensitivity:CS_CB,adaptive:CA_CB};

// Active palette accessors — read cbMode at call time
function pal(name){
  if(!cbMode)return{CR:CR,CE:CE,CS:CS,CA:CA,CD:CD,DRGB:DRGB,PCC:PCC}[name];
  return{CR:CR_CB,CE:CE_CB,CS:CS_CB,CA:CA_CB,CD:CD_CB,DRGB:DRGB_CB,PCC:PCC_CB}[name];
}

function fcolor(pr,fi){
  var pcc=pal('PCC'),cr=pal('CR'),cd=pal('CD'),drgb=pal('DRGB');
  // Raw variable preview mode
  if(rawVar&&rawPct[rawVar]){
    var darr=getDeltaArr(rawVar);
    if(darr){
      var d=darr[fi];
      var dr=window._deltaRange;
      if(!dr)return"#ccc";
      var span=dr.max-dr.min;if(span<1e-6)span=1;
      var t=(d-dr.min)/span;
      return toHex(interpC(cd,Math.max(0,Math.min(1,t))));
    }
    var pct=rawPct[rawVar][fi];
    var cs=pcc[varPillar[rawVar]]||cr;
    return toHex(interpC(cs,(pct-2)/96));
  }
  if(curLayer==="composite")return toHex(interpC(cr,(pr._comp-2)/96));
  if(curLayer==="driver"){var b=drgb[pr._drv]||[128,128,128];
    var intn=0.15+0.85*(pr._drvI||0.5);
    return toHex(b.map(function(c){return Math.round(255-intn*(255-c))}));}
  var m=curLayer.match(/^(exposure|sensitivity|adaptive)_pc(\\d+)$/);
  if(m){var v=pr["_"+m[1]+"_pc"+m[2]];if(v==null)return"#ccc";
    return toHex(interpC(pcc[m[1]]||cr,(v-2)/96));}
  return"#ccc";
}

/* ================================================================
   MAP DRAW
   ================================================================ */
function drawTracts(){
  if(tLayer){map.removeLayer(tLayer);tLayer=null;}
  // Build GEOID->index lookup for raw percentile access
  if(!window._gIdx){window._gIdx={};GJ.features.forEach(function(f,i){window._gIdx[f.properties.GEOID]=i;});}
  tLayer=L.geoJSON(GJ,{
    style:function(f){var idx=window._gIdx[f.properties.GEOID]||0;return{fillColor:fcolor(f.properties,idx),color:"#fff",weight:0.5,fillOpacity:0.8}},
    onEachFeature:function(f,lyr){
      lyr.on("mouseover",function(e){e.target.setStyle({weight:2,color:"#333"})});
      lyr.on("mouseout",function(e){if(tLayer)tLayer.resetStyle(e.target)});
      var p=f.properties;
      var fi=window._gIdx[p.GEOID]||0;
      // In compare-awaiting mode, intercept clicks before the popup opens
      lyr.on("click",function(e){
        if(compareMode==='awaiting'){
          L.DomEvent.stopPropagation(e);
          if(p.GEOID===compareFirst){return;} // ignore re-click of the same tract
          finishCompare(p.GEOID);
        }
      });
      var h='<div style="font-family:sans-serif;font-size:13px;line-height:1.6">'+
        '<b>Tract '+p.GEOID.slice(-6)+'</b><br>';
      if(rawVar){
        var rv=p[rawVar];
        var dArrTip=getDeltaArr(rawVar);
        if(dArrTip){
          var delta=dArrTip[fi];
          h+='\\u0394 '+(varLabel[rawVar]||rawVar)+': <b style="color:#c0392b">'+
            (delta>=0?'+':'')+delta.toFixed(2)+'\\u00b0C</b><br>';
          h+='<span style="color:#888;font-size:11px">Present: '+(rv!=null?rv:'—')+
            ' \\u2192 Projected: '+getAdj(rawVar,fi).toFixed(1)+'</span><br>';
        }else{
          h+=(varLabel[rawVar]||rawVar)+': '+(rv!=null?rv:'—')+'<br>';
        }
      }else{
        if(p._comp!=null) h+='Composite: '+p._comp.toFixed(1)+'<br>';
        if(p._exposure!=null) h+='Heat Exposure: '+p._exposure.toFixed(1)+'<br>';
        if(p._sensitivity!=null) h+='Population Sensitivity: '+p._sensitivity.toFixed(1)+'<br>';
        if(p._adaptive!=null) h+='Adaptive Cap: '+p._adaptive.toFixed(1)+'<br>';
      }
      h+='<button class="popup-compare-btn" onclick="startCompare(\\''+p.GEOID+'\\')">Compare with another tract</button>';
      h+='</div>';
      lyr.bindPopup(h);
    }
  }).addTo(map);
  // Keep jurisdiction boundaries on top of tracts
  Object.keys(jurisLayers).forEach(function(k){if(map.hasLayer(jurisLayers[k]))jurisLayers[k].bringToFront();});
  if(typeof tribalLayers!=='undefined'){
    Object.keys(tribalLayers).forEach(function(k){if(map.hasLayer(tribalLayers[k]))tribalLayers[k].bringToFront();});
  }
  showLegend();
}

// Hover-tooltip helper. Used by showLegend AND showLayerBtns, so it must live
// at module scope — when nested inside showLegend, calls from showLayerBtns
// throw ReferenceError and silently nuke the PC dropdown render.
function ic(t){return ' <span class="info-ic" tabindex="0">i<span class="info-pop">'+t+'</span></span>';}

function showLegend(){
  var el=document.getElementById("legend");
  // Raw variable preview legend
  if(rawVar){
    // Future scenario + projectable var: delta legend
    if(getDeltaArr(rawVar)&&window._deltaRange){
      var dr=window._deltaRange;
      var sl={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'};
      var dstops=pal('CD').map(function(c){return"rgb("+c.join(",")+")"}).join(",");
      el.innerHTML='<div style="font-weight:600;margin-bottom:2px">\\u0394 '+varLabel[rawVar]+ic('Projected change in this variable between the future scenario and the present-day baseline (LOCA2 ensemble mean). Warmer colors indicate larger projected increases.')+'</div>'+
        '<div style="font-size:10px;color:#666;margin-bottom:4px">'+sl[curScenario]+' '+curPeriod+' change from present &nbsp;<a href="https://climatedata.ca/resource/understanding-shared-socio-economic-pathways-ssps/" target="_blank" rel="noopener" style="color:#2563eb;text-decoration:underline">What are SSPs?</a></div>'+
        '<div class="grad" style="background:linear-gradient(to right,'+dstops+')"></div>'+
        '<div class="labels"><span>+'+dr.min.toFixed(1)+'\\u00b0C</span><span>+'+dr.max.toFixed(1)+'\\u00b0C</span></div>';
      el.style.display="block";return;
    }
    var cs=pal('PCC')[varPillar[rawVar]]||pal('CR');
    var stops=cs.map(function(c){return"rgb("+c.join(",")+")"}).join(",");
    el.innerHTML='<div style="font-weight:600;margin-bottom:2px">'+varLabel[rawVar]+ic('Each tract is colored by its percentile rank for this variable across all San Diego County tracts. Low = bottom of the distribution, High = top.')+'</div>'+
      '<div class="grad" style="background:linear-gradient(to right,'+stops+')"></div>'+
      '<div class="labels"><span>Low</span><span>High</span></div>';
    el.style.display="block";return;
  }
  if(curLayer==="composite"){
    var cstops=pal('CR').map(function(c){return"rgb("+c.join(",")+")"}).join(",");
    el.innerHTML='<div style="font-weight:600;margin-bottom:2px">Composite Risk'+ic('Mean of the three pillar scores: Heat Exposure, Population Sensitivity, and low Adaptive Capacity. Each pillar is a PCA-weighted composite of the variables you turned on, percentile-ranked across San Diego County tracts. Higher = greater overall heat risk.')+'</div>'+
      '<div class="grad" style="background:linear-gradient(to right,'+cstops+')"></div>'+
      '<div class="labels"><span>Low</span><span>High</span></div>';
    el.style.display="block";
  }else if(curLayer==="driver"){
    var dr=pal('DRGB');
    var rgb=function(arr){return"rgb("+arr.join(",")+")"};
    el.innerHTML='<div style="font-weight:600;margin-bottom:4px">Dominant Driver'+ic('Each tract is colored by whichever pillar contributes most to its risk: Heat Exposure, Population Sensitivity, or low Adaptive Capacity. This identifies the leading driver of heat risk for each tract.')+'</div>'+
      '<div class="dl-row"><div class="dl-sw" style="background:'+rgb(dr.exposure)+'"></div>Heat Exposure</div>'+
      '<div class="dl-row"><div class="dl-sw" style="background:'+rgb(dr.sensitivity)+'"></div>Population Sensitivity</div>'+
      '<div class="dl-row"><div class="dl-sw" style="background:'+rgb(dr.adaptive)+'"></div>Adaptive Capacity</div>'+
      '<div style="font-size:10px;color:#999;margin-top:3px;font-style:italic">Lighter = lower risk</div>';
    el.style.display="block";
  }else{
    var m=curLayer.match(/^(exposure|sensitivity|adaptive)_pc(\\d+)$/);
    if(m){var cs=pal('PCC')[m[1]]||pal('CR');
      var stops=cs.map(function(c){return"rgb("+c.join(",")+")"}).join(",");
      var pidM=m[1],kM=parseInt(m[2],10)-1;
      var rM=results&&results[pidM];
      var blurbM=(rM&&rM.pcs&&rM.pcs[kM])?_patternBlurb(rM,kM):'';
      var subM=blurbM?'<div style="font-size:10px;color:#666;margin-top:2px;line-height:1.35">'+blurbM+'</div>':'';
      el.innerHTML='<div style="font-weight:600;margin-bottom:2px">'+PMETA[pidM].label+' &mdash; Pattern '+m[2]+ic('A primary spatial pattern from PCA on the enabled variables in this pillar. Pattern 1 captures the largest shared pattern of variance across tracts; later patterns capture the next-largest, mutually independent patterns. Sign-aligned so higher = higher risk.')+'</div>'+
        subM+
        '<div class="grad" style="background:linear-gradient(to right,'+stops+')"></div>'+
        '<div class="labels"><span>Low</span><span>High</span></div>';
      el.style.display="block";
    }else el.style.display="none";
  }
}

/* ================================================================
   UI — CONTROLS
   ================================================================ */
function buildUI(){
  var el=document.getElementById("controls"),html="";
  ["exposure","sensitivity","adaptive"].forEach(function(pid){
    var pm=PMETA[pid];
    html+='<div class="pillar-section">';
    html+='<label class="pillar-hdr" style="border-color:'+pm.color+';color:'+pm.color+'" title="Toggle all variables in this pillar">'+
      '<input type="checkbox" id="pcb_'+pid+'" onchange="togPillar(\\''+pid+'\\',this.checked)">'+
      pm.label+'</label>';
    // Group vars by subsection, preserving order
    var subs=[],seen={};
    VM.forEach(function(v){
      if(v.pillar!==pid)return;
      if(!seen[v.sub]){seen[v.sub]=[];subs.push(v.sub);}
      seen[v.sub].push(v);
    });
    subs.forEach(function(sub){
      html+='<details class="subsec"><summary>'+sub+'</summary>';
      seen[sub].forEach(function(v){
        var chk=enabled[v.key]?" checked":"";
        var cls=enabled[v.key]?"var-row on":"var-row";
        var src=VSRC[v.key]?'<a class="src-link" href="'+VSRC[v.key]+'" target="_blank">(Source)</a>':"";
        var fix=(DELTAS&&v.pillar==='exposure'&&!DELTA_MAP[v.key])?
          '<span class="fixed-tag" id="fix_'+v.key+'" style="display:none">held fixed</span>':"";
        html+='<div class="'+cls+'" id="vr_'+v.key+'"><label style="color:'+pm.color+'">'+
          '<input type="checkbox" onchange="tog(\\''+v.key+'\\')" '+chk+'>'+
          '<span style="color:#444">'+v.label+'</span></label>'+fix+src+'</div>';
      });
      html+='</details>';
    });
    html+='</div>';
  });
  el.innerHTML=html;
  ["exposure","sensitivity","adaptive"].forEach(syncPillarCb);
}

function tog(key){
  enabled[key]=!enabled[key];
  var row=document.getElementById("vr_"+key);
  if(row){if(enabled[key])row.classList.add("on");else row.classList.remove("on");}
  if(enabled[key]){
    // Show this raw variable on the map
    rawVar=key;
    updateDeltaRange();
    document.getElementById("prompt").style.display="none";
    drawTracts();
  }else{
    // If this was the displayed variable, clear or show another enabled one
    if(rawVar===key){
      rawVar=null;
      // Find another enabled variable to show, or clear
      VM.forEach(function(v){if(enabled[v.key]&&!rawVar)rawVar=v.key;});
      updateDeltaRange();
      if(rawVar)drawTracts();
      else if(tLayer){map.removeLayer(tLayer);tLayer=null;
        document.getElementById("legend").style.display="none";
        document.getElementById("prompt").style.display="flex";
        document.getElementById("prompt").textContent="Select variables and click Build Composite Risk Map";}
    }
  }
  syncPillarCb(varPillar[key]);
}

// Set the pillar header checkbox to checked / unchecked / indeterminate
// based on how many of its child variables are currently enabled.
function syncPillarCb(pid){
  var cb=document.getElementById("pcb_"+pid);if(!cb)return;
  var n=0,on=0;
  VM.forEach(function(v){if(v.pillar===pid){n++;if(enabled[v.key])on++;}});
  cb.checked=(n>0&&on===n);
  cb.indeterminate=(on>0&&on<n);
}

// Master checkbox handler: turn every variable in a pillar on or off in one
// shot, then update the displayed raw layer the same way tog() does on a
// single-variable change.
function togPillar(pid,on){
  VM.forEach(function(v){
    if(v.pillar!==pid)return;
    enabled[v.key]=on;
    var row=document.getElementById("vr_"+v.key);
    if(row){
      var cb=row.querySelector('input[type=checkbox]');
      if(cb)cb.checked=on;
      if(on)row.classList.add("on");else row.classList.remove("on");
    }
  });
  // Reconcile the displayed raw variable with the new enabled set.
  if(!on && rawVar && varPillar[rawVar]===pid){
    rawVar=null;
    VM.forEach(function(v){if(enabled[v.key]&&!rawVar)rawVar=v.key;});
    updateDeltaRange();
    if(rawVar)drawTracts();
    else if(tLayer){map.removeLayer(tLayer);tLayer=null;
      document.getElementById("legend").style.display="none";
      document.getElementById("prompt").style.display="flex";
      document.getElementById("prompt").textContent="Select variables and click Build Composite Risk Map";}
  }else if(on && !rawVar){
    VM.forEach(function(v){if(enabled[v.key]&&!rawVar)rawVar=v.key;});
    if(rawVar){
      updateDeltaRange();
      document.getElementById("prompt").style.display="none";
      drawTracts();
    }
  }
  syncPillarCb(pid);
}

function showResults(){
  var el=document.getElementById("results");
  if(!results){el.innerHTML="";return;}
  var html="";
  ["exposure","sensitivity","adaptive"].forEach(function(pid){
    var r=results[pid],pm=PMETA[pid];
    if(r)html+='<div class="res-row"><b style="color:'+pm.color+'">'+pm.label+'</b>: '+
      r.totvar+'% var ('+r.nret+' PC'+(r.nret>1?'s':'')+')</div>';
    else html+='<div class="res-row res-empty">'+pm.label+': no variables</div>';
  });
  el.innerHTML=html;
}

// Look up a variable's pretty label from its key.
function _vmLabel(key){
  for(var i=0;i<VM.length;i++)if(VM[i].key===key)return VM[i].label;
  return key;
}

// Build a "driven by ..." string for one PC: top 3 vars by |loading|, with
// signs that indicate whether higher values of that variable push the
// pattern up (+) or down (-).
function _patternBlurb(r,k){
  var pc=r.pcs[k],ld=pc.loadings,keys=r.keys;
  if(!ld||!keys)return"";
  var idx=[];for(var j=0;j<keys.length;j++)idx.push(j);
  idx.sort(function(a,b){return Math.abs(ld[b])-Math.abs(ld[a])});
  var top=idx.slice(0,Math.min(3,idx.length));
  var parts=top.map(function(j){
    var sign=ld[j]>=0?'+':'−';
    return sign+' '+_vmLabel(keys[j]);
  });
  return 'Driven by: '+parts.join(', ')+'.';
}

function showLayerBtns(){
  var el=document.getElementById("layer-buttons"),html="";
  [{id:"composite",label:"Composite Risk",dot:"#bd0026"},
   {id:"driver",label:"Dominant Driver",dot:"#2c3e50"}].forEach(function(b){
    html+='<button class="lbtn'+(curLayer===b.id?" active":"")+'" onclick="setLyr(\\''+b.id+'\\')">'+
      '<span class="ldot" style="background:'+b.dot+'"></span>'+b.label+'</button>';});
  if(results){
    var anyPC=false;
    ["exposure","sensitivity","adaptive"].forEach(function(pid){
      if(results[pid]&&results[pid].pcs&&results[pid].pcs.length)anyPC=true;});
    if(anyPC){
      // Inner active flag controls open-state so the dropdown stays expanded
      // when the active layer is one of the PCs.
      var pcOpen=/^(exposure|sensitivity|adaptive)_pc\\d+$/.test(curLayer);
      html+='<details class="pc-group"'+(pcOpen?' open':'')+'>'+
        '<summary>Primary Spatial Patterns</summary>'+
        '<div class="pc-help">Underlying patterns from PCA on the variables in each pillar. Hover the <span class="info-ic-mini">i</span> for top contributors.</div>';
      ["exposure","sensitivity","adaptive"].forEach(function(pid){
        var r=results[pid];if(!r||!r.pcs||!r.pcs.length)return;
        var pm=PMETA[pid];
        for(var k=0;k<r.pcs.length;k++){
          var lid=pid+"_pc"+(k+1);
          var blurb=_patternBlurb(r,k).replace(/"/g,'&quot;');
          var info=blurb?ic(blurb):'';
          html+='<button class="lbtn'+(curLayer===lid?" active":"")+'" onclick="setLyr(\\''+lid+'\\')">'+
            '<span class="ldot" style="background:'+pm.color+'"></span>'+
            '<span class="lbtn-txt">'+pm.label+' &mdash; Pattern '+(k+1)+
            ' <span class="lbtn-pct">('+r.pcs[k].pct+'% var)</span></span>'+
            info+'</button>';
        }
      });
      html+='</details>';
    }
  }
  el.innerHTML=html;
}

function setLyr(id){rawVar=null;curLayer=id;drawTracts();showLayerBtns();}

/* ================================================================
   COMPARE TRACTS
   ================================================================ */
var compareMode  = "off";  // "off" | "awaiting" | "shown"
var compareFirst = null;
var compareSecond= null;
var cmpMaps      = {a:null,b:null}; // mini-map instances

function startCompare(geoid){
  compareFirst=geoid;compareMode="awaiting";
  map.closePopup();
  document.getElementById('compare-first-id').textContent=geoid.slice(-6);
  document.getElementById('compare-banner').style.display='block';
  document.getElementById('map-wrap').classList.add('compare-cursor');
}

function cancelCompare(){
  compareMode="off";compareFirst=null;compareSecond=null;
  document.getElementById('compare-banner').style.display='none';
  document.getElementById('map-wrap').classList.remove('compare-cursor');
}

function finishCompare(geoid){
  compareSecond=geoid;compareMode="shown";
  document.getElementById('compare-banner').style.display='none';
  document.getElementById('map-wrap').classList.remove('compare-cursor');
  openCompareModal(compareFirst,compareSecond);
}

function closeCompareModal(){
  document.getElementById('compare-modal').classList.remove('open');
  ['a','b'].forEach(function(s){
    if(cmpMaps[s]){cmpMaps[s].remove();cmpMaps[s]=null;}
    _landuseLayers[s]=null; _cmpLST[s]=null;
  });
  compareMode="off";compareFirst=null;compareSecond=null;
}

document.addEventListener('keydown',function(e){
  if(e.key==='Escape'){
    var cityModal=document.getElementById('city-compare-modal');
    if(document.getElementById('scn-compare-modal').classList.contains('open')){closeScnCompareModal();}
    else if(document.getElementById('compare-modal').classList.contains('open')){closeCompareModal();}
    else if(cityModal&&cityModal.classList.contains('open')){closeCityCompareModal();}
    else if(compareMode==='awaiting'){cancelCompare();}
  }
});

function openCompareModal(geoidA,geoidB){
  var idxA=window._gIdx[geoidA],idxB=window._gIdx[geoidB];
  var fA=GJ.features[idxA],fB=GJ.features[idxB];
  if(!fA||!fB){return;}

  document.getElementById('cmp-tract-a-id').textContent='Tract '+geoidA.slice(-6);
  document.getElementById('cmp-tract-b-id').textContent='Tract '+geoidB.slice(-6);
  document.getElementById('cmp-tract-a-meta').textContent='GEOID '+geoidA;
  document.getElementById('cmp-tract-b-meta').textContent='GEOID '+geoidB;

  // Context label — current scenario / period
  var ctx='';
  if(curScenario==='present'){ctx='Present Day';}
  else{
    var sl={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'};
    ctx=sl[curScenario]+' \\u2022 '+(curPeriod==='2050'?'Mid-Century 2050':'End-of-Century 2080');
  }
  document.getElementById('cmp-context').textContent=ctx;

  // Build variable lists for each tract, organized like the sidebar
  var subsBy={};
  VM.forEach(function(v){
    var k=v.pillar+'||'+v.sub;
    (subsBy[k]=subsBy[k]||[]).push(v);
  });
  var pillarOrder=['exposure','sensitivity','adaptive'];
  var rowsA=[],rowsB=[];
  pillarOrder.forEach(function(pid){
    var pmHdr=PMETA[pid];
    var pillarHasContent=false;
    var aChunk='',bChunk='';
    Object.keys(subsBy).forEach(function(k){
      var pp=k.split('||')[0],ss=k.split('||')[1];
      if(pp!==pid)return;
      var vars=subsBy[k];
      var aSubChunk='',bSubChunk='';
      vars.forEach(function(v){
        var vA=fA.properties[v.key],vB=fB.properties[v.key];
        // For projectable exposure vars, displayed raw is the projected value
        // under the current scenario; for everything else, the stored value.
        var dArr=getDeltaArr(v.key);
        var rawA=(vA==null||isNaN(vA))?null:(dArr?getAdj(v.key,idxA):+vA);
        var rawB=(vB==null||isNaN(vB))?null:(dArr?getAdj(v.key,idxB):+vB);
        var dispRawA=fmtCmpRaw(v.key,rawA);
        var dispRawB=fmtCmpRaw(v.key,rawB);
        // Within-county percentile rank (0–100) — uses present-day pre-computed
        // ranks; for projected vars the relative ordering across SD tracts is
        // approximately preserved under uniform-ish CMIP6 deltas.
        var pA=(rawPct[v.key]&&rawPct[v.key][idxA]!=null)?rawPct[v.key][idxA]:null;
        var pB=(rawPct[v.key]&&rawPct[v.key][idxB]!=null)?rawPct[v.key][idxB]:null;
        var dispPctA=(pA!=null)?(Math.round(pA)+'th pct'):'';
        var dispPctB=(pB!=null)?(Math.round(pB)+'th pct'):'';
        // Highlight which is higher RISK (red) vs lower RISK (green) using
        // per-variable polarity. POLARITY[key]===false means higher value =
        // lower risk (e.g. tree canopy, AC prevalence).
        var clsA='cmp-row',clsB='cmp-row';
        if(rawA!=null&&rawB!=null){
          var higherIsWorse=(POLARITY&&POLARITY[v.key]!==false);
          if(rawA!==rawB){
            var aIsWorse=higherIsWorse?(rawA>rawB):(rawA<rawB);
            if(aIsWorse){clsA+=' cmp-higher';clsB+=' cmp-lower';}
            else        {clsB+=' cmp-higher';clsA+=' cmp-lower';}
          }
        }
        aSubChunk+='<div class="'+clsA+'"><span class="cmp-label">'+v.label+'</span><span class="cmp-val"><span class="cmp-val-raw">'+dispRawA+'</span><span class="cmp-val-pct">'+dispPctA+'</span></span></div>';
        bSubChunk+='<div class="'+clsB+'"><span class="cmp-label">'+v.label+'</span><span class="cmp-val"><span class="cmp-val-raw">'+dispRawB+'</span><span class="cmp-val-pct">'+dispPctB+'</span></span></div>';
      });
      if(aSubChunk){
        aChunk+='<div class="cmp-sub-hdr">'+ss+'</div>'+aSubChunk;
        bChunk+='<div class="cmp-sub-hdr">'+ss+'</div>'+bSubChunk;
        pillarHasContent=true;
      }
    });
    if(pillarHasContent){
      var hdr='<div class="cmp-pillar-hdr" style="border-color:'+pmHdr.color+';color:'+pmHdr.color+'">'+pmHdr.label+'</div>';
      rowsA.push(hdr+aChunk);
      rowsB.push(hdr+bChunk);
    }
  });
  // Add PCA pillar / composite scores at top if available
  var hasPca=(fA.properties._comp!=null);
  if(hasPca){
    var pcaA='<div class="cmp-pillar-hdr" style="border-color:#444;color:#444">PCA Scores</div>';
    var pcaB=pcaA;
    [['_comp','Composite Risk'],['_exposure','Heat Exposure'],['_sensitivity','Population Sensitivity'],['_adaptive','Adaptive Capacity']].forEach(function(p){
      var aV=fA.properties[p[0]],bV=fB.properties[p[0]];
      if(aV==null||bV==null)return;
      var clsA='cmp-row',clsB='cmp-row';
      if(aV>bV){clsA+=' cmp-higher';clsB+=' cmp-lower';}
      else if(bV>aV){clsB+=' cmp-higher';clsA+=' cmp-lower';}
      pcaA+='<div class="'+clsA+'"><span class="cmp-label">'+p[1]+'</span><span class="cmp-val"><span class="cmp-val-raw">'+aV.toFixed(1)+'</span></span></div>';
      pcaB+='<div class="'+clsB+'"><span class="cmp-label">'+p[1]+'</span><span class="cmp-val"><span class="cmp-val-raw">'+bV.toFixed(1)+'</span></span></div>';
    });
    rowsA.unshift(pcaA);rowsB.unshift(pcaB);
  }
  document.getElementById('cmp-vars-a').innerHTML=rowsA.join('');
  document.getElementById('cmp-vars-b').innerHTML=rowsB.join('');

  // Show modal first so the map containers have layout (Leaflet needs sizes)
  document.getElementById('compare-modal').classList.add('open');
  renderLstLegend();
  var _luCb=document.getElementById('cmp-toggle-landuse');
  if(_luCb)_luCb.checked=_landuseShown;
  renderLanduseLegend();
  setTimeout(function(){buildMiniMap('cmp-map-a','a',fA);
                        buildMiniMap('cmp-map-b','b',fB);},20);
}

function lstVariant(){
  if(!LST)return null;
  if(LST.variants&&LST.variants[cbMode?'cb':'default'])return LST.variants[cbMode?'cb':'default'];
  return {url:LST.url,legend:LST.legend,colormap:LST.colormap};
}

function buildMiniMap(elId,slot,feat){
  if(cmpMaps[slot]){cmpMaps[slot].remove();}
  _landuseLayers[slot]=null; _cmpLST[slot]=null;
  var m=L.map(elId,{zoomControl:false,attributionControl:false,
    scrollWheelZoom:true,dragging:true});
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{maxZoom:19}).addTo(m);
  // Custom pane so the protomaps PMTiles canvas (a GridLayer) renders ABOVE
  // the LST imageOverlay. Without this it lands in tilePane (z=200) under
  // overlayPane (z=400) and the LST visually swamps the parcels.
  if(!m.getPane('landusePane')){
    m.createPane('landusePane');
    m.getPane('landusePane').style.zIndex=450;
  }
  // Landsat 100m LST overlay — clipped visually by zooming to the tract bounds.
  // Opacity drops when the land-use overlay is on so parcels stay legible.
  var lv=lstVariant();
  if(lv){
    var b=LST.bounds;
    var initOp=_landuseShown?0.25:0.85;
    _cmpLST[slot]=L.imageOverlay(lv.url,[[b.south,b.west],[b.north,b.east]],{opacity:initOp}).addTo(m);
  }
  // Tract outline (no fill, so the LST shows through)
  var poly=L.geoJSON(feat,{style:{color:'#111',weight:2,fill:false}}).addTo(m);
  m.fitBounds(poly.getBounds(),{padding:[18,18]});
  cmpMaps[slot]=m;
  if(_landuseShown)_addLanduseToMap(slot,m);
}

function renderLstLegend(){
  var el=document.getElementById('cmp-lst-legend');
  var lv=lstVariant();
  if(!el||!lv){if(el)el.style.display='none';return;}
  var stops=lv.legend.map(function(s){return"rgb("+s.rgb.join(",")+")"}).join(",");
  var ticks='';
  lv.legend.forEach(function(s){ticks+='<span>'+s.value+'&deg;</span>';});
  el.innerHTML='<span class="cmp-lst-title">Landsat LST 100m (annual mean &deg;C):</span>'+
    '<span class="cmp-lst-bar" style="background:linear-gradient(to right,'+stops+')"></span>'+
    '<span class="cmp-lst-ticks">'+ticks+'</span>';
  el.style.display='flex';
}

/* ================================================================
   COMPARE CITIES
   Population-weighted aggregations across census tracts whose
   centroid falls inside each city's polygon. Reservations are
   intentionally excluded — see showTribalNoData() below.
   ================================================================ */
var cityCmpMaps={a:null,b:null};
var cityCmpSel ={a:null,b:null};
var _jurisTractCache=null;

function _ensureJurisTractCache(){
  if(_jurisTractCache)return;
  _ensureTractJuris();
  _jurisTractCache={};
  for(var i=0;i<_tractJurisName.length;i++){
    var n=_tractJurisName[i];
    if(!n)continue;
    (_jurisTractCache[n]=_jurisTractCache[n]||[]).push(i);
  }
}
function _jurisTractIdxs(name){
  _ensureJurisTractCache();
  return _jurisTractCache[name]||[];
}
function _jurisTotalPop(name){
  var idxs=_jurisTractIdxs(name),s=0;
  for(var i=0;i<idxs.length;i++){s+=(+GJ.features[idxs[i]].properties.E_TOTPOP||0);}
  return s;
}
function _jurisPopWMean(name,key,opts){
  opts=opts||{};
  var idxs=_jurisTractIdxs(name);
  var dArr=opts.useDelta?getDeltaArr(key):null;
  var wsum=0,vsum=0,n=0;
  for(var i=0;i<idxs.length;i++){
    var idx=idxs[i];
    var pr=GJ.features[idx].properties;
    var v=dArr?getAdj(key,idx):pr[key];
    if(v==null||isNaN(v))continue;
    var w=+pr.E_TOTPOP||0;
    if(w<=0)continue;
    wsum+=w;vsum+=w*(+v);n++;
  }
  if(wsum<=0||n===0)return null;
  return {value:vsum/wsum,pop:wsum,n:n};
}
function _jurisDriverBreakdown(name){
  if(!results)return null;
  var idxs=_jurisTractIdxs(name);
  var counts={exposure:0,sensitivity:0,adaptive:0};
  for(var i=0;i<idxs.length;i++){
    var d=GJ.features[idxs[i]].properties._drv;
    if(d&&counts[d]!==undefined)counts[d]++;
  }
  return counts;
}
function _isRankVar(key){
  return /_pctile$/.test(key) || /^RPL_/.test(key);
}

function openCityCompareModal(initialName){
  if(typeof JURIS==='undefined'||!JURIS){return;}
  _ensureJurisTractCache();
  var allNames=Object.keys(_jurisTractCache).sort(function(a,b){return a.localeCompare(b);});
  if(!allNames.length){return;}
  ['a','b'].forEach(function(s){
    var sel=document.getElementById('city-cmp-sel-'+s);
    sel.innerHTML='<option value="">\\u2014 Pick a city \\u2014</option>'+
      allNames.map(function(n){
        return '<option value="'+n.replace(/"/g,'&quot;')+'">'+n+'</option>';
      }).join('');
  });
  cityCmpSel.a=(initialName&&_jurisTractCache[initialName])?initialName:'';
  cityCmpSel.b='';
  for(var i=0;i<allNames.length;i++){
    if(allNames[i]!==cityCmpSel.a){cityCmpSel.b=allNames[i];break;}
  }
  document.getElementById('city-cmp-sel-a').value=cityCmpSel.a;
  document.getElementById('city-cmp-sel-b').value=cityCmpSel.b;
  var ctx='';
  if(curScenario==='present'){ctx='Present Day';}
  else{
    var sl={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'};
    ctx=sl[curScenario]+' \\u2022 '+(curPeriod==='2050'?'Mid-Century 2050':'End-of-Century 2080');
  }
  document.getElementById('city-cmp-context').textContent=ctx;
  document.getElementById('city-compare-modal').classList.add('open');
  setTimeout(function(){_renderCityCompareSide('a');_renderCityCompareSide('b');
                        _renderCityCompareBody(cityCmpSel.a,cityCmpSel.b);},20);
}

function setCityCompare(side,name){
  cityCmpSel[side]=name;
  _renderCityCompareSide(side);
  _renderCityCompareBody(cityCmpSel.a,cityCmpSel.b);
}

function _renderCityCompareSide(side){
  var name=cityCmpSel[side];
  var nameEl=document.getElementById('city-cmp-name-'+side);
  var metaEl=document.getElementById('city-cmp-meta-'+side);
  if(!name){
    nameEl.textContent='\\u2014';
    metaEl.textContent='';
    if(cityCmpMaps[side]){cityCmpMaps[side].remove();cityCmpMaps[side]=null;}
    return;
  }
  var idxs=_jurisTractIdxs(name);
  var pop=_jurisTotalPop(name);
  nameEl.textContent=name;
  metaEl.textContent=idxs.length+' tract'+(idxs.length===1?'':'s')+' \\u2022 '+pop.toLocaleString()+' residents';
  buildCityMiniMap('city-cmp-map-'+side,side,name);
}

function buildCityMiniMap(elId,slot,name){
  if(cityCmpMaps[slot]){cityCmpMaps[slot].remove();}
  var m=L.map(elId,{zoomControl:false,attributionControl:false,
    scrollWheelZoom:true,dragging:true});
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{maxZoom:19}).addTo(m);
  var idxs=_jurisTractIdxs(name);
  if(idxs.length){
    var feats=idxs.map(function(i){return GJ.features[i];});
    L.geoJSON({type:'FeatureCollection',features:feats},{
      style:{color:'#888',weight:0.5,fillColor:'#3b82f6',fillOpacity:0.15}
    }).addTo(m);
  }
  var jurisFeats=JURIS.features.filter(function(f){return f.properties.NAME===name;});
  var poly=L.geoJSON({type:'FeatureCollection',features:jurisFeats},{
    style:{color:'#111',weight:2,fill:false}
  }).addTo(m);
  try{m.fitBounds(poly.getBounds(),{padding:[10,10]});}
  catch(_){m.setView([32.85,-117.05],10);}
  cityCmpMaps[slot]=m;
}

function _renderCityCompareBody(nameA,nameB){
  var elA=document.getElementById('city-cmp-vars-a');
  var elB=document.getElementById('city-cmp-vars-b');
  if(!nameA||!nameB){
    elA.innerHTML=nameA?'<div style="padding:20px;color:#888;font-style:italic">Pick a city on the right to compare</div>':'';
    elB.innerHTML=nameB?'<div style="padding:20px;color:#888;font-style:italic">Pick a city on the left to compare</div>':'';
    return;
  }
  var subsBy={};
  VM.forEach(function(v){
    var k=v.pillar+'||'+v.sub;
    (subsBy[k]=subsBy[k]||[]).push(v);
  });
  var pillarOrder=['exposure','sensitivity','adaptive'];
  var rowsA=[],rowsB=[];
  pillarOrder.forEach(function(pid){
    var pmHdr=PMETA[pid];
    var pillarHasContent=false;
    var aChunk='',bChunk='';
    Object.keys(subsBy).forEach(function(k){
      var pp=k.split('||')[0],ss=k.split('||')[1];
      if(pp!==pid)return;
      var vars=subsBy[k];
      var aSubChunk='',bSubChunk='';
      vars.forEach(function(v){
        var dArr=getDeltaArr(v.key);
        var aR=_jurisPopWMean(nameA,v.key,{useDelta:!!dArr});
        var bR=_jurisPopWMean(nameB,v.key,{useDelta:!!dArr});
        if(aR===null&&bR===null)return;
        var rawA=aR?aR.value:null, rawB=bR?bR.value:null;
        var dispA=(rawA!=null)?fmtCmpRaw(v.key,rawA):'\\u2014';
        var dispB=(rawB!=null)?fmtCmpRaw(v.key,rawB):'\\u2014';
        var noteSuffix=_isRankVar(v.key)?' <span style="font-size:10px;color:#999">(mean tract rank)</span>':'';
        var label=v.label+noteSuffix;
        var clsA='cmp-row',clsB='cmp-row';
        if(rawA!=null&&rawB!=null&&rawA!==rawB){
          var higherIsWorse=(POLARITY&&POLARITY[v.key]!==false);
          var aIsWorse=higherIsWorse?(rawA>rawB):(rawA<rawB);
          if(aIsWorse){clsA+=' cmp-higher';clsB+=' cmp-lower';}
          else        {clsB+=' cmp-higher';clsA+=' cmp-lower';}
        }
        aSubChunk+='<div class="'+clsA+'"><span class="cmp-label">'+label+'</span><span class="cmp-val"><span class="cmp-val-raw">'+dispA+'</span></span></div>';
        bSubChunk+='<div class="'+clsB+'"><span class="cmp-label">'+label+'</span><span class="cmp-val"><span class="cmp-val-raw">'+dispB+'</span></span></div>';
      });
      if(aSubChunk){
        aChunk+='<div class="cmp-sub-hdr">'+ss+'</div>'+aSubChunk;
        bChunk+='<div class="cmp-sub-hdr">'+ss+'</div>'+bSubChunk;
        pillarHasContent=true;
      }
    });
    if(pillarHasContent){
      var hdr='<div class="cmp-pillar-hdr" style="border-color:'+pmHdr.color+';color:'+pmHdr.color+'">'+pmHdr.label+'</div>';
      rowsA.push(hdr+aChunk);
      rowsB.push(hdr+bChunk);
    }
  });
  if(results){
    var pcaA='<div class="cmp-pillar-hdr" style="border-color:#444;color:#444">PCA Scores</div>';
    var pcaB=pcaA;
    var rankNote=' <span style="font-size:10px;color:#999">(mean tract rank)</span>';
    [['_comp','Composite Risk'],['_exposure','Heat Exposure'],
     ['_sensitivity','Population Sensitivity'],['_adaptive','Adaptive Capacity']].forEach(function(p){
      var aR=_jurisPopWMean(nameA,p[0]);
      var bR=_jurisPopWMean(nameB,p[0]);
      if(aR===null||bR===null)return;
      var clsA='cmp-row',clsB='cmp-row';
      if(aR.value>bR.value){clsA+=' cmp-higher';clsB+=' cmp-lower';}
      else if(bR.value>aR.value){clsB+=' cmp-higher';clsA+=' cmp-lower';}
      pcaA+='<div class="'+clsA+'"><span class="cmp-label">'+p[1]+rankNote+'</span><span class="cmp-val"><span class="cmp-val-raw">'+aR.value.toFixed(1)+'</span></span></div>';
      pcaB+='<div class="'+clsB+'"><span class="cmp-label">'+p[1]+rankNote+'</span><span class="cmp-val"><span class="cmp-val-raw">'+bR.value.toFixed(1)+'</span></span></div>';
    });
    var dA=_jurisDriverBreakdown(nameA), dB=_jurisDriverBreakdown(nameB);
    if(dA&&dB){
      var fmtDrv=function(d){return 'Exp '+d.exposure+' / Sens '+d.sensitivity+' / Adapt '+d.adaptive;};
      pcaA+='<div class="cmp-row"><span class="cmp-label">Dominant Driver <span style="font-size:10px;color:#999">(tract count)</span></span><span class="cmp-val"><span class="cmp-val-raw" style="font-size:11px">'+fmtDrv(dA)+'</span></span></div>';
      pcaB+='<div class="cmp-row"><span class="cmp-label">Dominant Driver <span style="font-size:10px;color:#999">(tract count)</span></span><span class="cmp-val"><span class="cmp-val-raw" style="font-size:11px">'+fmtDrv(dB)+'</span></span></div>';
    }
    rowsA.unshift(pcaA);rowsB.unshift(pcaB);
  }
  elA.innerHTML=rowsA.join('');
  elB.innerHTML=rowsB.join('');
}

function closeCityCompareModal(){
  document.getElementById('city-compare-modal').classList.remove('open');
  ['a','b'].forEach(function(s){
    if(cityCmpMaps[s]){cityCmpMaps[s].remove();cityCmpMaps[s]=null;}
  });
}

// Soft popup shown when a user clicks a tribal-lands polygon. Census tract
// demographics don't represent reservation-resident populations, so rather
// than fabricate a number we offer to compare the surrounding tract or its
// containing city instead.
function showTribalNoData(latlng,fullName,type){
  var pt=[latlng.lng,latlng.lat];
  var hit=-1;
  for(var i=0;i<GJ.features.length;i++){
    if(_ptInGeometry(pt,GJ.features[i].geometry)){hit=i;break;}
  }
  _ensureTractJuris();
  var jurisName=(hit>=0&&_tractJurisName[hit])?_tractJurisName[hit]:null;
  var typeLabel=(type==='trust')?'off-reservation trust land':'reservation';
  var html='<div style="font-size:12px;line-height:1.45;max-width:240px">'+
    '<div style="font-weight:700;margin-bottom:4px">'+fullName+'</div>'+
    '<div style="color:#666;margin-bottom:8px">Census tract demographics don\\u2019t represent populations on this '+typeLabel+', so we don\\u2019t aggregate stats here.</div>';
  if(hit>=0){
    var geoid=GJ.features[hit].properties.GEOID;
    html+='<div style="font-size:11px;color:#666;margin-bottom:4px">Compare the surrounding tract instead:</div>'+
      '<button class="popup-compare-btn" onclick="map.closePopup();startCompare(\\''+geoid+'\\')">Compare Tract '+geoid.slice(-6)+'</button>';
  }
  if(jurisName){
    var safeName=jurisName.replace(/'/g,"\\\\'");
    html+='<div style="margin-top:6px"><button class="popup-compare-btn" onclick="map.closePopup();openCityCompareModal(\\''+safeName+'\\')">Compare '+jurisName+'</button></div>';
  }
  html+='</div>';
  L.popup({offset:[0,-4],className:'tribal-popup'}).setLatLng(latlng).setContent(html).openOn(map);
}

/* ----------------------------------------------------------------
   SANDAG land-use parcels (PMTiles overlay in compare-modal mini-maps)
   ---------------------------------------------------------------- */
var LANDUSE_URL="../data/context/sandag_landuse.pmtiles";
var LU_CATS=[
  {code:'1',name:'Residential',color:'#f4a460'},
  {code:'2',name:'Industrial',color:'#8b4513'},
  {code:'4',name:'Transportation / Utilities',color:'#888888'},
  {code:'5',name:'Commercial / Retail',color:'#e8442a'},
  {code:'6',name:'Public / Institutional',color:'#2166ac'},
  {code:'7',name:'Recreation / Open Space',color:'#1a7a1a'},
  {code:'8',name:'Agriculture',color:'#8fbc00'},
  {code:'9',name:'Vacant / Other',color:'#cccccc'}
];
var LU_WATER_NAMES={'Water':1,'Lake/Reservoir/Large Pond':1,'Bay or Lagoon':1};
var LU_WATER_CAT={name:'Water',color:'#1e90ff'};
var LU_DEFAULT_COLOR='#cccccc';

function _luLookup(props){
  if(!props)return null;
  if(props.Landuse&&LU_WATER_NAMES[props.Landuse])return LU_WATER_CAT;
  var lu=(props.lu||'').toString().replace(/[^0-9]/g,'');
  if(!lu)return null;
  var prefix=lu.charAt(0);
  for(var i=0;i<LU_CATS.length;i++){if(LU_CATS[i].code===prefix)return LU_CATS[i];}
  return null;
}
function _luCategoryColor(props){var c=_luLookup(props);return c?c.color:LU_DEFAULT_COLOR;}
function _luCategoryName(props){var c=_luLookup(props);return c?c.name:'Other';}

var _landuseShown=false;
var _landuseLayers={a:null,b:null};
var _cmpLST={a:null,b:null};

function _buildLanduseLayer(){
  if(!window.protomapsL)return null;
  try{
    return protomapsL.leafletLayer({
      url:LANDUSE_URL,
      pane:'landusePane',
      paintRules:[{
        dataLayer:'landuse',
        symbolizer:new protomapsL.PolygonSymbolizer({
          fill:function(z,f){return _luCategoryColor(f.props);},
          opacity:0.78,
          stroke:'#222',
          width:0.5
        })
      }]
    });
  }catch(e){console.warn('protomaps-leaflet init failed',e);return null;}
}

function _addLanduseToMap(slot,map){
  if(_landuseLayers[slot])return;
  var layer=_buildLanduseLayer();
  if(!layer)return;
  layer.addTo(map);
  _landuseLayers[slot]=layer;
  // Click → popup with the parcel's land-use category. protomaps-leaflet v4
  // exposes queryTileFeaturesDebug(lng,lat,brushPx); returns Map<source, [{feature,layerName}]>.
  map.on('click',function(e){
    if(!_landuseShown)return;
    var L_=_landuseLayers[slot];
    if(!L_||typeof L_.queryTileFeaturesDebug!=='function')return;
    var hits=null;
    try{hits=L_.queryTileFeaturesDebug(e.latlng.lng,e.latlng.lat,16);}catch(err){return;}
    if(!hits||!hits.size)return;
    var feats=null;
    hits.forEach(function(arr){if(!feats&&arr&&arr.length)feats=arr;});
    if(!feats||!feats.length)return;
    var p=(feats[0].feature&&feats[0].feature.props)||{};
    var cat=_luCategoryName(p);
    var label=p.Landuse||cat;
    var lu=p.lu||'';
    L.popup({offset:[0,-4]}).setLatLng(e.latlng)
      .setContent('<div style="font-size:11px;line-height:1.35"><b>'+label+'</b><br>'+
        '<span style="color:#888">'+cat+(lu?' &middot; LU '+lu:'')+'</span></div>')
      .openOn(map);
  });
}

function _removeLanduseFromMap(slot,map){
  if(_landuseLayers[slot]){
    try{map.removeLayer(_landuseLayers[slot]);}catch(e){}
    _landuseLayers[slot]=null;
  }
}

function toggleLanduse(on){
  _landuseShown=!!on;
  ['a','b'].forEach(function(s){
    var m=cmpMaps[s];
    if(!m)return;
    if(_cmpLST[s])_cmpLST[s].setOpacity(_landuseShown?0.25:0.85);
    if(_landuseShown){_addLanduseToMap(s,m);}else{_removeLanduseFromMap(s,m);}
  });
  renderLanduseLegend();
}

function renderLanduseLegend(){
  var el=document.getElementById('cmp-landuse-legend');
  if(!el)return;
  if(!_landuseShown){el.classList.remove('open');el.innerHTML='';return;}
  var cats=LU_CATS.concat([LU_WATER_CAT]);
  el.innerHTML=cats.map(function(c){
    return '<span class="cmp-lu-item"><span class="cmp-lu-swatch" style="background:'+c.color+'"></span>'+c.name+'</span>';
  }).join('');
  el.classList.add('open');
}

/* ================================================================
   COMPARE SCENARIOS  (whole-map juxtaposition under two climates)
   Left  = full Composite Risk under scenario A (sequential CR ramp)
   Right = Δ Heat-Exposure pillar (B − A) under matched variable selection
           (diverging RdBu palette, symmetric around zero)

   Only the exposure pillar moves between scenarios — demographics,
   AC, etc. are held at today's values — so the right map isolates
   the climate signal instead of diluting it through unchanged inputs.
   ================================================================ */
var scnCmpMaps   = {a:null, b:null};
var scnCmpLayers = {a:null, b:null};
var scnCmpScn    = {a:'present|', b:'ssp245|2050'};
var scnCmpComp   = {a:null, b:null};   // cached full composite per side (left uses 'a')
var scnCmpExp    = {a:null, b:null};   // cached exposure-pillar percentiles per side
var _scnCmpSyncing = false;
var _scnCmpFitted  = false;

// Diverging palette (ColorBrewer RdBu reversed): blue = lower exposure, red = higher
// — already colorblind-safe across deuteranopia/protanopia/tritanopia.
var CDIV=[[33,102,172],[67,147,195],[146,197,222],[209,229,240],
          [247,247,247],
          [253,219,199],[244,165,130],[214,96,77],[178,24,43]];

// Available scenario/period options, filtered by what's actually in DELTAS.
function _scnCompareOptions(){
  var opts=[{val:'present|', label:'Present Day'}];
  var sLab={'ssp245':'Moderate Emissions (SSP 2-4.5)',
            'ssp370':'High Emissions (SSP 3-7.0)',
            'ssp585':'Very High Emissions (SSP 5-8.5)'};
  var pLab={'2050':'Mid-Century 2050','2080':'End-of-Century 2080'};
  ['2050','2080'].forEach(function(p){
    if(!DELTAS||!DELTAS[p])return;
    ['ssp245','ssp370','ssp585'].forEach(function(s){
      if(!DELTAS[p][s])return;
      opts.push({val:s+'|'+p, label:pLab[p]+' — '+sLab[s]});
    });
  });
  return opts;
}

// Compute full composite-risk percentile per feature for an arbitrary
// (scenario, period). Uses the current `enabled` variable set.
// Does NOT mutate GJ.features. Returns null if no variables enabled.
function computeCompositeFor(scenario, period){
  var savedScn=curScenario, savedPer=curPeriod;
  curScenario=scenario;
  curPeriod=(scenario==='present')?'':(period||'');
  try{
    var pv={exposure:[],sensitivity:[],adaptive:[]};
    VM.forEach(function(v){if(enabled[v.key])pv[v.pillar].push(v.key);});
    var total=pv.exposure.length+pv.sensitivity.length+pv.adaptive.length;
    if(total===0)return null;
    var res={};
    ['exposure','sensitivity','adaptive'].forEach(function(pid){
      res[pid]=pillarPCA(pid,pv[pid]);
    });
    var N=GJ.features.length, comp=new Array(N);
    for(var i=0;i<N;i++){
      var sum=0,cnt=0;
      ['exposure','sensitivity','adaptive'].forEach(function(pid){
        if(!res[pid])return;
        var s=res[pid].scores[i];
        if(pid==='adaptive')s=100-s;
        sum+=s;cnt++;
      });
      comp[i]=cnt>0?Math.round(sum/cnt*10)/10:50;
    }
    return comp;
  } finally {
    curScenario=savedScn;
    curPeriod=savedPer;
  }
}

// Compute the exposure-pillar percentile per feature for an arbitrary
// (scenario, period), using only the exposure variables the user has
// turned on. Returns null if no exposure variables are enabled.
function computeExposureFor(scenario, period){
  var savedScn=curScenario, savedPer=curPeriod;
  curScenario=scenario;
  curPeriod=(scenario==='present')?'':(period||'');
  try{
    var keys=[];
    VM.forEach(function(v){if(enabled[v.key]&&v.pillar==='exposure')keys.push(v.key);});
    if(keys.length===0)return null;
    var res=pillarPCA('exposure',keys);
    return res?res.scores:null;
  } finally {
    curScenario=savedScn;
    curPeriod=savedPer;
  }
}

function openScnCompareModal(){
  // Need at least one variable enabled overall, and at least one EXPOSURE
  // variable for the right delta map to mean anything.
  var anyOn=false, anyExposure=false;
  for(var k in enabled){
    if(!enabled[k])continue;
    anyOn=true;
    if(varPillar[k]==='exposure')anyExposure=true;
  }
  if(!anyOn){
    var eb=document.getElementById('error-bar');
    eb.textContent='Turn on at least one variable before comparing scenarios.';
    eb.style.display='block';
    setTimeout(function(){eb.style.display='none';},3500);
    return;
  }
  if(!anyExposure){
    var eb2=document.getElementById('error-bar');
    eb2.textContent='Turn on at least one Heat Exposure variable (e.g. a temperature metric) — that pillar is what changes between scenarios.';
    eb2.style.display='block';
    setTimeout(function(){eb2.style.display='none';},5000);
    return;
  }
  // Defaults: left = Present Day, right = currently selected (or first non-present option)
  var opts=_scnCompareOptions();
  var defaultRight=null;
  if(curScenario!=='present'&&curPeriod){defaultRight=curScenario+'|'+curPeriod;}
  if(!defaultRight){
    for(var i=0;i<opts.length;i++){if(opts[i].val!=='present|'){defaultRight=opts[i].val;break;}}
  }
  scnCmpScn={a:'present|', b:defaultRight||'present|'};
  ['a','b'].forEach(function(side){
    var sel=document.getElementById('scn-cmp-sel-'+side);
    sel.innerHTML=opts.map(function(o){
      return '<option value="'+o.val+'"'+(o.val===scnCmpScn[side]?' selected':'')+'>'+o.label+'</option>';
    }).join('');
  });
  document.getElementById('scn-compare-modal').classList.add('open');
  setTimeout(function(){
    renderScnCompareSide('a');
    renderScnCompareSide('b');
    _refreshScnCompareLegend();
    setTimeout(_initScnSync,40);
  },30);
}

function closeScnCompareModal(){
  document.getElementById('scn-compare-modal').classList.remove('open');
  ['a','b'].forEach(function(s){
    if(scnCmpMaps[s]){scnCmpMaps[s].remove();scnCmpMaps[s]=null;}
    scnCmpLayers[s]=null;
    scnCmpComp[s]=null;
    scnCmpExp[s]=null;
  });
  _scnCmpFitted=false;
  window._scnDeltaMax=null;window._scnDeltaArr=null;
}

function setScnCompare(side, val){
  scnCmpScn[side]=val;
  // Right side renders Δ exposure (B − A), so changing A also changes the right map.
  if(side==='a'){
    renderScnCompareSide('a');
    renderScnCompareSide('b');
  } else {
    renderScnCompareSide('b');
  }
  _refreshScnCompareLegend();
}

function renderScnCompareSide(side){
  var parts=scnCmpScn[side].split('|');
  var scn=parts[0], period=parts[1]||'';
  var elId='scn-cmp-map-'+side;
  var metaEl=document.getElementById('scn-cmp-meta-'+side);

  // Lazy init Leaflet
  if(!scnCmpMaps[side]){
    var m=L.map(elId,{zoomControl:true,attributionControl:false}).setView([32.85,-117.05],9);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{maxZoom:18}).addTo(m);
    scnCmpMaps[side]=m;
  }
  var map2=scnCmpMaps[side];
  if(scnCmpLayers[side]){map2.removeLayer(scnCmpLayers[side]);}

  var styleFn, tipFn;
  if(side==='a'){
    var comp=computeCompositeFor(scn, period);
    if(!comp){if(metaEl)metaEl.textContent='No variables selected.';return;}
    scnCmpComp.a=comp;
    // Pre-cache exposure for A so the right side doesn't have to re-run it
    scnCmpExp.a=computeExposureFor(scn, period);
    var cr=pal('CR');
    styleFn=function(f){
      var idx=window._gIdx[f.properties.GEOID]||0;
      var c=comp[idx];
      return {fillColor:toHex(interpC(cr,(c-2)/96)),color:'#fff',weight:0.4,fillOpacity:0.85};
    };
    tipFn=function(f,lyr){
      var idx=window._gIdx[f.properties.GEOID]||0;
      lyr.bindTooltip('Tract '+f.properties.GEOID.slice(-6)+': '+comp[idx].toFixed(1),
        {sticky:true,className:'juris-tip'});
    };
  } else {
    var expB=computeExposureFor(scn, period);
    if(!expB){if(metaEl)metaEl.textContent='No exposure variables selected.';return;}
    scnCmpExp.b=expB;
    if(!scnCmpExp.a){
      var partsA=scnCmpScn.a.split('|');
      scnCmpExp.a=computeExposureFor(partsA[0],partsA[1]||'');
    }
    var expA=scnCmpExp.a||[];
    var delta=new Array(expB.length);
    var maxAbs=0;
    for(var i=0;i<expB.length;i++){
      var d=expB[i]-(expA[i]!=null?expA[i]:0);
      delta[i]=d;
      if(Math.abs(d)>maxAbs)maxAbs=Math.abs(d);
    }
    if(maxAbs<1)maxAbs=1; // floor so an all-zero scale doesn't render as 0/0
    window._scnDeltaMax=maxAbs;
    window._scnDeltaArr=delta;
    styleFn=function(f){
      var idx=window._gIdx[f.properties.GEOID]||0;
      var d=delta[idx];
      var t=(d+maxAbs)/(2*maxAbs);
      return {fillColor:toHex(interpC(CDIV,Math.max(0,Math.min(1,t)))),
              color:'#fff',weight:0.4,fillOpacity:0.85};
    };
    tipFn=function(f,lyr){
      var idx=window._gIdx[f.properties.GEOID]||0;
      var d=delta[idx];
      var sign=(d>0?'+':'');
      lyr.bindTooltip('Tract '+f.properties.GEOID.slice(-6)+
        '<br>Δ Heat Exposure: <b>'+sign+d.toFixed(1)+'</b>'+
        '<br><span style="color:#888">A: '+(expA[idx]!=null?expA[idx].toFixed(1):'—')+
        ' → B: '+expB[idx].toFixed(1)+'</span>',
        {sticky:true,className:'juris-tip'});
    };
  }

  scnCmpLayers[side]=L.geoJSON(GJ,{style:styleFn,onEachFeature:tipFn}).addTo(map2);

  // Fit bounds once per modal open so user zoom isn't lost on every re-render.
  if(side==='a'&&!_scnCmpFitted&&scnCmpLayers.a){
    try{map2.fitBounds(scnCmpLayers.a.getBounds(),{padding:[10,10]});_scnCmpFitted=true;}catch(e){}
  }

  // Meta line
  if(metaEl){
    var lab=null,opts=_scnCompareOptions();
    for(var j=0;j<opts.length;j++){if(opts[j].val===scnCmpScn[side]){lab=opts[j].label;break;}}
    if(side==='a'){
      metaEl.textContent=lab||'';
    } else {
      var aLab=null;
      for(var k2=0;k2<opts.length;k2++){if(opts[k2].val===scnCmpScn.a){aLab=opts[k2].label;break;}}
      metaEl.innerHTML='Δ Heat Exposure vs. <b>'+(aLab||scnCmpScn.a)+'</b> &middot; '+(lab||'');
    }
  }
}

function _refreshScnCompareLegend(){
  var lel=document.getElementById('scn-cmp-legend');
  if(!lel)return;
  var cr=pal('CR');
  var crStops=cr.map(function(c){return'rgb('+c.join(',')+')'}).join(',');
  var divStops=CDIV.map(function(c){return'rgb('+c.join(',')+')'}).join(',');
  var dMax=(window._scnDeltaMax!=null?window._scnDeltaMax:0).toFixed(1);
  lel.innerHTML=
    '<div class="scn-leg-block">'+
      '<span class="cmp-lst-title">Left — Composite Risk:</span>'+
      '<span class="cmp-lst-bar" style="background:linear-gradient(to right,'+crStops+')"></span>'+
      '<span class="cmp-lst-ticks"><span>Lower</span><span>Higher</span></span>'+
    '</div>'+
    '<div class="scn-leg-block">'+
      '<span class="cmp-lst-title">Right — Δ Heat Exposure (B − A):</span>'+
      '<span class="cmp-lst-bar" style="background:linear-gradient(to right,'+divStops+')"></span>'+
      '<span class="cmp-lst-ticks"><span>−'+dMax+'</span><span>0</span><span>+'+dMax+'</span></span>'+
    '</div>';
}

function _initScnSync(){
  if(!scnCmpMaps.a||!scnCmpMaps.b)return;
  function bind(src,dst){
    src.on('move zoom',function(){
      if(_scnCmpSyncing)return;
      _scnCmpSyncing=true;
      dst.setView(src.getCenter(),src.getZoom(),{animate:false});
      _scnCmpSyncing=false;
    });
  }
  bind(scnCmpMaps.a,scnCmpMaps.b);
  bind(scnCmpMaps.b,scnCmpMaps.a);
  try{
    _scnCmpSyncing=true;
    scnCmpMaps.b.setView(scnCmpMaps.a.getCenter(),scnCmpMaps.a.getZoom(),{animate:false});
  } finally { _scnCmpSyncing=false; }
  setTimeout(function(){scnCmpMaps.a.invalidateSize();scnCmpMaps.b.invalidateSize();},80);
}

/* ================================================================
   PDF EXPORT
   ================================================================ */
function _pdfShow(msg){
  var el=document.getElementById('pdf-overlay');
  if(msg)document.getElementById('pdf-overlay-msg').textContent=msg;
  el.classList.add('show');
}
function _pdfHide(){document.getElementById('pdf-overlay').classList.remove('show');}

function _loadScript(src){
  return new Promise(function(resolve,reject){
    var s=document.createElement('script');
    s.src=src;s.crossOrigin='anonymous';
    s.onload=resolve;s.onerror=function(){reject(new Error('Failed to load '+src));};
    document.head.appendChild(s);
  });
}

function _ensurePdfLibs(){
  var needs=[];
  if(!window.jspdf)needs.push(_loadScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js'));
  return Promise.all(needs);
}

function _layerLabel(){
  if(rawVar)return varLabel[rawVar]||rawVar;
  if(curLayer==='composite')return'Composite Risk';
  if(curLayer==='driver')return'Dominant Driver';
  var m=curLayer.match(/^(exposure|sensitivity|adaptive)_pc(\\d+)$/);
  if(m)return PMETA[m[1]].label+' — Pattern '+m[2];
  return curLayer;
}
function _scenarioLabel(){
  if(curScenario==='present')return'Present Day';
  var sl={'ssp245':'SSP2-4.5 (Moderate)','ssp370':'SSP3-7.0 (High)','ssp585':'SSP5-8.5 (Very High)'};
  var pl={'2050':'Mid-Century (2035\\u20132064)','2080':'End-of-Century (2065\\u20132094)'};
  return sl[curScenario]+' \\u2022 '+(pl[curPeriod]||'');
}

// Build a publication-quality figure of the study area without html2canvas.
//
// Approach: render to a 3200x2400 canvas ourselves using Leaflet only as a
// projector. Polygons (tracts, city outlines) are projected through the same
// latLng->pixel function as the basemap tiles, so they're guaranteed to align.
// The previous html2canvas pipeline could not reliably composite Leaflet's
// tile pane and SVG vector pane: each pane lives under its own CSS
// translate3d, and html2canvas captured them at different transforms,
// producing the misaligned tracts-vs-basemap output we used to ship.
function _captureStudyAreaMap(){
  return new Promise(function(resolve,reject){
    // Logical figure size; canvas is rendered at 2x for retina-print output.
    var TARGET_W=1600, TARGET_H=1200;
    var SCALE=2;
    var TILE_SIZE=256;

    // Off-screen Leaflet map used solely for projection math (zoom-fit +
    // latLngToContainerPoint). It carries no tile layer and is never captured.
    var projDiv=document.createElement('div');
    projDiv.style.cssText='position:absolute;top:-99999px;left:-99999px;'+
      'width:'+TARGET_W+'px;height:'+TARGET_H+'px;';
    document.body.appendChild(projDiv);

    var projMap=L.map(projDiv,{zoomControl:false,attributionControl:false,
      fadeAnimation:false,zoomAnimation:false});

    // Frame the study area (tract bounds, fall back to county bbox)
    var bounds=null;
    if(tLayer&&tLayer.getBounds){var b=tLayer.getBounds();if(b&&b.isValid())bounds=b;}
    if(!bounds)bounds=L.latLngBounds([32.53,-117.62],[33.51,-116.08]);
    projMap.fitBounds(bounds,{padding:[20,20],animate:false});

    var zoom=projMap.getZoom();
    var pixelBounds=projMap.getPixelBounds();

    var canvas=document.createElement('canvas');
    canvas.width=TARGET_W*SCALE;
    canvas.height=TARGET_H*SCALE;
    var ctx=canvas.getContext('2d');
    ctx.fillStyle='#ffffff';
    ctx.fillRect(0,0,canvas.width,canvas.height);

    // latLng -> physical canvas pixel (after retina scaling).
    function project(lat,lng){
      var pt=projMap.latLngToContainerPoint([lat,lng]);
      return [pt.x*SCALE, pt.y*SCALE];
    }

    function cleanup(){
      try{projMap.remove();}catch(e){}
      try{projDiv.parentNode.removeChild(projDiv);}catch(e){}
    }

    // ── Basemap tiles ──────────────────────────────────────────
    // Tile coords cover the container's pixel range at the chosen zoom.
    var minTX=Math.floor(pixelBounds.min.x/TILE_SIZE);
    var maxTX=Math.floor(pixelBounds.max.x/TILE_SIZE);
    var minTY=Math.floor(pixelBounds.min.y/TILE_SIZE);
    var maxTY=Math.floor(pixelBounds.max.y/TILE_SIZE);
    var nWorld=Math.pow(2,zoom);
    var subdomains=['a','b','c'];

    function loadImage(src){
      return new Promise(function(res){
        var img=new Image();
        img.crossOrigin='anonymous';
        img.onload=function(){res(img);};
        img.onerror=function(){res(null);};
        img.src=src;
      });
    }

    var tileTasks=[];
    for(var tx=minTX;tx<=maxTX;tx++){
      for(var ty=minTY;ty<=maxTY;ty++){
        if(ty<0||ty>=nWorld)continue;
        var wrapTx=((tx%nWorld)+nWorld)%nWorld;
        var sd=subdomains[Math.abs(tx+ty)%subdomains.length];
        var url='https://'+sd+'.basemaps.cartocdn.com/light_all/'+
                zoom+'/'+wrapTx+'/'+ty+'@2x.png';
        var dx=(tx*TILE_SIZE-pixelBounds.min.x)*SCALE;
        var dy=(ty*TILE_SIZE-pixelBounds.min.y)*SCALE;
        (function(x,y,src){
          tileTasks.push(loadImage(src).then(function(img){
            if(img)ctx.drawImage(img,x,y,TILE_SIZE*SCALE,TILE_SIZE*SCALE);
          }));
        })(dx,dy,url);
      }
    }

    // ── Vector overlays ────────────────────────────────────────
    function addRings(rings){
      rings.forEach(function(ring){
        for(var i=0;i<ring.length;i++){
          var pt=project(ring[i][1],ring[i][0]); // GeoJSON: [lng,lat]
          if(i===0)ctx.moveTo(pt[0],pt[1]);
          else ctx.lineTo(pt[0],pt[1]);
        }
        ctx.closePath();
      });
    }
    function drawFeature(feat,fillColor,strokeColor,strokeWidth,fillOpacity){
      var g=feat.geometry;
      if(!g)return;
      ctx.beginPath();
      if(g.type==='Polygon')addRings(g.coordinates);
      else if(g.type==='MultiPolygon')g.coordinates.forEach(addRings);
      else return;
      if(fillColor){
        ctx.globalAlpha=fillOpacity;
        ctx.fillStyle=fillColor;
        ctx.fill('evenodd');
      }
      if(strokeColor&&strokeWidth>0){
        ctx.globalAlpha=1;
        ctx.lineJoin='round';
        ctx.lineCap='round';
        ctx.strokeStyle=strokeColor;
        ctx.lineWidth=strokeWidth;
        ctx.stroke();
      }
      ctx.globalAlpha=1;
    }

    // ── Decorations ────────────────────────────────────────────
    function drawScaleBar(){
      var c=projMap.getCenter();
      var p1=projMap.latLngToContainerPoint(c);
      var p2=projMap.latLngToContainerPoint([c.lat,c.lng+0.01]);
      var dxc=p2.x-p1.x;
      if(dxc<=0)return;
      var mPerPx=(0.01*111320*Math.cos(c.lat*Math.PI/180))/dxc;
      function pickD(maxM){
        var p=Math.pow(10,Math.floor(Math.log10(maxM)));
        if(5*p<=maxM)return 5*p;
        if(2*p<=maxM)return 2*p;
        return p;
      }
      var maxBar=200; // logical px — long enough to read at print scale
      var dM=pickD(maxBar*mPerPx);
      var dMi=pickD(maxBar*(mPerPx/1609.344));
      var barM=dM/mPerPx;
      var barMi=(dMi*1609.344)/mPerPx;
      var labelM=(dM>=1000)?(dM/1000)+' km':dM+' m';
      var labelMi=(dMi>=1)?dMi+' mi':(Math.round(dMi*5280))+' ft';

      var pad=14, font=14, tickH=6;
      var totalW=Math.max(barM,barMi)+pad*2;
      var totalH=72;
      var bx=18, by=TARGET_H-totalH-18;

      ctx.save();
      ctx.fillStyle='rgba(255,255,255,0.95)';
      ctx.fillRect(bx*SCALE, by*SCALE, totalW*SCALE, totalH*SCALE);
      ctx.strokeStyle='rgba(15,23,42,0.20)';
      ctx.lineWidth=1*SCALE;
      ctx.strokeRect(bx*SCALE, by*SCALE, totalW*SCALE, totalH*SCALE);

      ctx.strokeStyle='#0f172a';
      ctx.lineWidth=2*SCALE;
      ctx.lineCap='butt';

      var x0=bx+pad, yMet=by+14, yImp=by+44;
      ctx.beginPath();
      ctx.moveTo(x0*SCALE,yMet*SCALE); ctx.lineTo((x0+barM)*SCALE,yMet*SCALE);
      ctx.moveTo(x0*SCALE,(yMet-tickH)*SCALE); ctx.lineTo(x0*SCALE,(yMet+tickH)*SCALE);
      ctx.moveTo((x0+barM)*SCALE,(yMet-tickH)*SCALE); ctx.lineTo((x0+barM)*SCALE,(yMet+tickH)*SCALE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x0*SCALE,yImp*SCALE); ctx.lineTo((x0+barMi)*SCALE,yImp*SCALE);
      ctx.moveTo(x0*SCALE,(yImp-tickH)*SCALE); ctx.lineTo(x0*SCALE,(yImp+tickH)*SCALE);
      ctx.moveTo((x0+barMi)*SCALE,(yImp-tickH)*SCALE); ctx.lineTo((x0+barMi)*SCALE,(yImp+tickH)*SCALE);
      ctx.stroke();

      ctx.fillStyle='#0f172a';
      ctx.font='600 '+(font*SCALE)+'px -apple-system,Segoe UI,Roboto,sans-serif';
      ctx.textBaseline='top'; ctx.textAlign='left';
      ctx.fillText(labelM, x0*SCALE, (yMet+10)*SCALE);
      ctx.fillText(labelMi, x0*SCALE, (yImp+10)*SCALE);
      ctx.restore();
    }

    function drawLegend(){
      // Mirror the live in-page legend (showLegend) for the active view state.
      var sl={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'};
      var title='', subtitle='', stops=null;
      var lowLabel='Low', highLabel='High';
      var driverRows=null;

      if(rawVar){
        if(typeof getDeltaArr==='function'&&getDeltaArr(rawVar)&&window._deltaRange){
          var dr=window._deltaRange;
          title='Δ '+(varLabel[rawVar]||rawVar);
          subtitle=(sl[curScenario]||curScenario)+' '+curPeriod+' change from present';
          stops=pal('CD');
          lowLabel='+'+dr.min.toFixed(1)+'°C';
          highLabel='+'+dr.max.toFixed(1)+'°C';
        }else{
          title=varLabel[rawVar]||rawVar;
          subtitle=_scenarioLabel();
          stops=pal('PCC')[varPillar[rawVar]]||pal('CR');
        }
      }else if(curLayer==='composite'){
        title='Composite Risk';
        subtitle=_scenarioLabel();
        stops=pal('CR');
      }else if(curLayer==='driver'){
        title='Dominant Driver';
        subtitle=_scenarioLabel();
        var d=pal('DRGB');
        driverRows=[
          {label:'Heat Exposure',rgb:d.exposure},
          {label:'Population Sensitivity',rgb:d.sensitivity},
          {label:'Adaptive Capacity',rgb:d.adaptive}
        ];
      }else{
        var m=curLayer.match(/^(exposure|sensitivity|adaptive)_pc(\\d+)$/);
        if(m){
          title=PMETA[m[1]].label+' — Pattern '+m[2];
          subtitle=_scenarioLabel();
          stops=pal('PCC')[m[1]]||pal('CR');
        }
      }
      if(!title)return;

      var cityRows=[];
      if(typeof JURIS!=='undefined'&&JURIS&&typeof jurisLayers!=='undefined'){
        Object.keys(jurisLayers).sort(function(a,b){return a.localeCompare(b);}).forEach(function(k){
          if(!map.hasLayer(jurisLayers[k]))return;
          cityRows.push({label:k, color:JURIS_COLOR});
        });
      }
      var tribalRows=[];
      if(typeof TRIBAL!=='undefined'&&TRIBAL&&typeof tribalGroups!=='undefined'){
        Object.keys(tribalGroups).sort(function(a,b){return a.localeCompare(b);}).forEach(function(gk){
          var anyOn=tribalGroups[gk].some(function(name){return map.hasLayer(tribalLayers[name]);});
          if(!anyOn)return;
          tribalRows.push({label:gk,color:TRIBAL_COLORS.reservation,dashed:false});
        });
      }

      var pad=14, gradW=200, gradH=14, swW=22, swH=14, rowH=22;
      var titleSize=18, subSize=12, bodySize=14, sectionGap=10;
      var fonts={
        title:'700 '+(titleSize*SCALE)+'px -apple-system,Segoe UI,Roboto,sans-serif',
        sub:(subSize*SCALE)+'px -apple-system,Segoe UI,Roboto,sans-serif',
        body:(bodySize*SCALE)+'px -apple-system,Segoe UI,Roboto,sans-serif',
        sect:'600 '+(bodySize*SCALE)+'px -apple-system,Segoe UI,Roboto,sans-serif',
        note:'italic '+(subSize*SCALE)+'px -apple-system,Segoe UI,Roboto,sans-serif'
      };

      function tw(text,font){ctx.font=font;return ctx.measureText(text).width/SCALE;}
      var maxLabelW=0;
      if(driverRows)driverRows.forEach(function(r){var w=tw(r.label,fonts.body);if(w>maxLabelW)maxLabelW=w;});
      cityRows.forEach(function(r){var w=tw(r.label,fonts.body);if(w>maxLabelW)maxLabelW=w;});
      tribalRows.forEach(function(r){var w=tw(r.label,fonts.body);if(w>maxLabelW)maxLabelW=w;});
      var titleW=tw(title,fonts.title);
      var subW=subtitle?tw(subtitle,fonts.sub):0;
      var showSspLink=!!(subtitle && curScenario && curScenario!=='present');
      var sspText='What are SSPs?';
      var sspW=showSspLink?tw(sspText,fonts.sub):0;
      var contentW=Math.max(gradW, titleW, subW, sspW, swW+10+maxLabelW);
      var boxW=contentW+pad*2;

      var plan=[], y=pad;
      plan.push({op:'title',y:y});            y+=titleSize+6;
      if(subtitle){plan.push({op:'sub',y:y}); y+=subSize+10;} else { y+=2; }
      if(showSspLink){plan.push({op:'ssplink',y:y}); y+=subSize+10;}
      if(stops){
        plan.push({op:'grad',y:y}); y+=gradH+4;
        plan.push({op:'gradLabels',y:y}); y+=bodySize+4;
      }
      if(driverRows){
        driverRows.forEach(function(r){plan.push({op:'drow',y:y,label:r.label,rgb:r.rgb});y+=rowH;});
        plan.push({op:'note',y:y+2,text:'Lighter = lower risk'});
        y+=subSize+8;
      }
      if(cityRows.length){
        y+=sectionGap;
        plan.push({op:'sect',y:y,text:'Jurisdictions'});
        y+=bodySize+8;
        cityRows.forEach(function(r){plan.push({op:'crow',y:y,label:r.label,color:r.color});y+=rowH;});
      }
      if(tribalRows.length){
        y+=sectionGap;
        plan.push({op:'sect',y:y,text:'Tribal Lands'});
        y+=bodySize+8;
        tribalRows.forEach(function(r){plan.push({op:'crow',y:y,label:r.label,color:r.color,dashed:r.dashed});y+=rowH;});
      }
      var boxH=y+pad-4;

      var bx=TARGET_W-boxW-18, by=18;

      ctx.save();
      ctx.fillStyle='rgba(255,255,255,0.95)';
      ctx.fillRect(bx*SCALE, by*SCALE, boxW*SCALE, boxH*SCALE);
      ctx.strokeStyle='rgba(15,23,42,0.20)';
      ctx.lineWidth=1*SCALE;
      ctx.strokeRect(bx*SCALE, by*SCALE, boxW*SCALE, boxH*SCALE);

      ctx.textBaseline='top'; ctx.textAlign='left';
      plan.forEach(function(L){
        var px=bx+pad, py=by+L.y;
        if(L.op==='title'){
          ctx.fillStyle='#0f172a'; ctx.font=fonts.title;
          ctx.fillText(title, px*SCALE, py*SCALE);
        }else if(L.op==='sub'){
          ctx.fillStyle='#475569'; ctx.font=fonts.sub;
          ctx.fillText(subtitle, px*SCALE, py*SCALE);
        }else if(L.op==='ssplink'){
          ctx.font=fonts.sub; ctx.fillStyle='#2563eb';
          ctx.fillText(sspText, px*SCALE, py*SCALE);
          var lwPx=ctx.measureText(sspText).width;
          var ulYpx=(py+subSize+1)*SCALE;
          ctx.strokeStyle='#2563eb'; ctx.lineWidth=1*SCALE;
          ctx.beginPath();
          ctx.moveTo(px*SCALE, ulYpx);
          ctx.lineTo(px*SCALE+lwPx, ulYpx);
          ctx.stroke();
          canvas._sspLinkRect={x:px*SCALE, y:py*SCALE, w:lwPx, h:(subSize+3)*SCALE};
        }else if(L.op==='grad'){
          var grad=ctx.createLinearGradient(px*SCALE,0,(px+gradW)*SCALE,0);
          stops.forEach(function(c,i){grad.addColorStop(i/(stops.length-1),'rgb('+c.join(',')+')');});
          ctx.fillStyle=grad;
          ctx.fillRect(px*SCALE, py*SCALE, gradW*SCALE, gradH*SCALE);
          ctx.strokeStyle='#94a3b8'; ctx.lineWidth=0.6*SCALE;
          ctx.strokeRect(px*SCALE, py*SCALE, gradW*SCALE, gradH*SCALE);
        }else if(L.op==='gradLabels'){
          ctx.fillStyle='#475569'; ctx.font=fonts.body;
          ctx.textAlign='left';  ctx.fillText(lowLabel, px*SCALE, py*SCALE);
          ctx.textAlign='right'; ctx.fillText(highLabel, (px+gradW)*SCALE, py*SCALE);
          ctx.textAlign='left';
        }else if(L.op==='drow'){
          ctx.fillStyle='rgb('+L.rgb.join(',')+')';
          ctx.fillRect(px*SCALE, py*SCALE, swW*SCALE, swH*SCALE);
          ctx.strokeStyle='#94a3b8'; ctx.lineWidth=0.5*SCALE;
          ctx.strokeRect(px*SCALE, py*SCALE, swW*SCALE, swH*SCALE);
          ctx.fillStyle='#0f172a'; ctx.font=fonts.body;
          ctx.fillText(L.label, (px+swW+10)*SCALE, py*SCALE);
        }else if(L.op==='note'){
          ctx.fillStyle='#94a3b8'; ctx.font=fonts.note;
          ctx.fillText(L.text, px*SCALE, py*SCALE);
        }else if(L.op==='sect'){
          ctx.fillStyle='#0f172a'; ctx.font=fonts.sect;
          ctx.fillText(L.text, px*SCALE, py*SCALE);
        }else if(L.op==='crow'){
          ctx.save();
          ctx.lineWidth=2*SCALE; ctx.strokeStyle=L.color;
          if(L.dashed)ctx.setLineDash([5*SCALE,4*SCALE]);
          ctx.strokeRect(px*SCALE, py*SCALE, swW*SCALE, swH*SCALE);
          ctx.restore();
          ctx.fillStyle='#0f172a'; ctx.font=fonts.body;
          ctx.fillText(L.label, (px+swW+10)*SCALE, py*SCALE);
        }
      });
      ctx.restore();
    }

    function drawAttribution(){
      var text='© OpenStreetMap contributors  © CARTO';
      ctx.font=(10*SCALE)+'px -apple-system,Segoe UI,Roboto,sans-serif';
      var tw=ctx.measureText(text).width;
      var pad=6*SCALE;
      var bx=TARGET_W*SCALE-tw-pad-8*SCALE;
      var by=TARGET_H*SCALE-6*SCALE;
      ctx.fillStyle='rgba(255,255,255,0.85)';
      ctx.fillRect(bx-pad,by-13*SCALE,tw+pad*2,17*SCALE);
      ctx.fillStyle='#444';
      ctx.textAlign='left';
      ctx.textBaseline='alphabetic';
      ctx.fillText(text,bx,by);
    }

    Promise.all(tileTasks).then(function(){
      // Tracts — colored using the live `fcolor()` against current pillar/var state
      if(tLayer&&typeof GJ!=='undefined'&&GJ&&GJ.features){
        GJ.features.forEach(function(f){
          var idx=window._gIdx[f.properties.GEOID]||0;
          drawFeature(f,fcolor(f.properties,idx),'#ffffff',0.5*SCALE,0.85);
        });
      }
      // Replicate visible jurisdiction outlines
      if(typeof JURIS!=='undefined'&&JURIS){
        Object.keys(jurisLayers).forEach(function(k){
          if(!map.hasLayer(jurisLayers[k]))return;
          var feats=JURIS.features.filter(function(f){return f.properties.NAME===k;});
          feats.forEach(function(f){drawFeature(f,null,JURIS_COLOR,1.8*SCALE,0);});
        });
      }
      // Replicate visible tribal land outlines
      if(typeof TRIBAL!=='undefined'&&TRIBAL&&typeof tribalLayers!=='undefined'){
        Object.keys(tribalLayers).forEach(function(k){
          if(!map.hasLayer(tribalLayers[k]))return;
          var f=TRIBAL.features.filter(function(x){return x.properties.NAME===k;})[0];
          if(!f)return;
          var t=f.properties.TYPE||'reservation';
          var c=TRIBAL_COLORS[t]||TRIBAL_COLORS.reservation;
          var dashed=(t==='trust');
          if(dashed){ctx.save();ctx.setLineDash([10*SCALE,8*SCALE]);}
          drawFeature(f,c,c,1.8*SCALE,0.10);
          if(dashed)ctx.restore();
        });
      }
      drawLegend();
      drawScaleBar();
      drawAttribution();
      cleanup();
      resolve(canvas);
    }).catch(function(err){
      cleanup();
      reject(err);
    });
  });
}

function _heroPdf(canvas){
  var jsPDF=window.jspdf.jsPDF;
  var pdf=new jsPDF({orientation:'landscape',unit:'pt',format:'letter',compress:true});
  var pw=pdf.internal.pageSize.getWidth();   // 792
  var ph=pdf.internal.pageSize.getHeight();  // 612
  var M=36;

  // ── Title band ─────────────────────────────────────────────
  pdf.setFillColor(247,249,252);
  pdf.rect(0,0,pw,72,'F');
  pdf.setFillColor(37,99,235); // accent line
  pdf.rect(0,72,pw,2.4,'F');

  pdf.setFont('helvetica','bold');
  pdf.setFontSize(20);pdf.setTextColor(15,23,42);
  pdf.text('San Diego Heat Risk Explorer',M,36);

  pdf.setFont('helvetica','normal');
  pdf.setFontSize(11);pdf.setTextColor(71,85,105);
  pdf.text(_layerLabel(),M,55);
  pdf.setTextColor(120,130,148);
  pdf.text('  \\u2022  '+_scenarioLabel(),M+pdf.getTextWidth(_layerLabel()),55);

  // ── Map figure ─────────────────────────────────────────────
  var top=92;
  var bottom=ph-58;
  var maxW=pw-2*M;
  var maxH=bottom-top;
  var ratio=canvas.width/canvas.height;
  var imgW=maxW, imgH=imgW/ratio;
  if(imgH>maxH){imgH=maxH;imgW=imgH*ratio;}
  var imgX=(pw-imgW)/2;
  var imgY=top+(maxH-imgH)/2;

  // Soft drop shadow + border
  pdf.setFillColor(225,228,235);
  pdf.rect(imgX+2,imgY+2,imgW,imgH,'F');
  pdf.addImage(canvas.toDataURL('image/png'),'PNG',imgX,imgY,imgW,imgH,undefined,'FAST');
  pdf.setDrawColor(203,213,225);
  pdf.setLineWidth(0.6);
  pdf.rect(imgX,imgY,imgW,imgH);

  // SSP info hotspot — clickable link area mapped from canvas to PDF coords
  if(canvas._sspLinkRect){
    var r=canvas._sspLinkRect;
    var lx=imgX+(r.x/canvas.width)*imgW;
    var ly=imgY+(r.y/canvas.height)*imgH;
    var lw=(r.w/canvas.width)*imgW;
    var lh=(r.h/canvas.height)*imgH;
    pdf.link(lx,ly,lw,lh,{url:'https://climatedata.ca/resource/understanding-shared-socio-economic-pathways-ssps/'});
  }

  // ── Footer ─────────────────────────────────────────────────
  pdf.setDrawColor(226,232,240);
  pdf.setLineWidth(0.5);
  pdf.line(M,ph-46,pw-M,ph-46);

  pdf.setFontSize(8);pdf.setTextColor(100,116,139);
  var dt=new Date();
  var dStr=dt.toLocaleString(undefined,{year:'numeric',month:'short',day:'2-digit',
                                         hour:'2-digit',minute:'2-digit'});
  pdf.text('Generated '+dStr,M,ph-30);
  pdf.text('Sources: LOCA2-Hybrid CMIP6 \\u00b7 USGS Landsat \\u00b7 NOAA HRRR \\u00b7 CDC PLACES & SVI \\u00b7 HealthyPlacesIndex',
    M,ph-18);

  var url='connorjmack.github.io/heat-risk-viewer';
  pdf.setFont('helvetica','bold');pdf.setTextColor(37,99,235);
  var urlW=pdf.getTextWidth(url);
  pdf.text(url,pw-M-urlW,ph-18);

  var ts=dt.toISOString().replace(/[:.]/g,'-').slice(0,19);
  pdf.save('san_diego_heat_risk_'+ts+'.pdf');
}

function downloadMapPdf(){
  var btn=document.getElementById('dl-pdf-btn');
  btn.disabled=true;
  _pdfShow('Loading PDF library\\u2026');
  _ensurePdfLibs().then(function(){
    _pdfShow('Rendering study-area map\\u2026');
    return _captureStudyAreaMap();
  }).then(function(canvas){
    _pdfShow('Building PDF\\u2026');
    _heroPdf(canvas);
    _pdfHide();
    btn.disabled=false;
  }).catch(function(err){
    console.error('PDF export failed:',err);
    _pdfHide();
    btn.disabled=false;
    var bar=document.getElementById('error-bar');
    bar.textContent='PDF export failed: '+err.message+
      '. If basemap tiles were blocked, allow basemaps.cartocdn.com, or use the browser\\u2019s Print \\u2192 Save as PDF.';
    bar.style.display='block';
    setTimeout(function(){bar.style.display='none';},9000);
  });
}

/* ================================================================
   ACCESSIBILITY
   ================================================================ */
function toggleCbMode(on){
  cbMode=!!on;
  try{localStorage.setItem('hrv-cb-mode',cbMode?'1':'0');}catch(e){}
  document.body.classList.toggle('cb-mode',cbMode);
  // Re-render anything that bakes palette colors into the DOM/canvas
  if(tLayer){drawTracts();}
  // If the compare modal is open, swap LST overlay + legend live
  if(document.getElementById('compare-modal').classList.contains('open')){
    renderLstLegend();
    if(compareFirst&&compareSecond){
      var fA=GJ.features[window._gIdx[compareFirst]];
      var fB=GJ.features[window._gIdx[compareSecond]];
      buildMiniMap('cmp-map-a','a',fA);
      buildMiniMap('cmp-map-b','b',fB);
    }
  }
}

/* ================================================================
   DATA EXPORT (CSV / GeoJSON, by jurisdiction or entire map)
   ================================================================ */
var _tractCentroids=null;
var _tractJurisName=null;
var _expJurisNames=null;

function _ptInRing(pt,ring){
  var x=pt[0],y=pt[1],inside=false;
  for(var i=0,j=ring.length-1;i<ring.length;j=i++){
    var xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1];
    var intersect=((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/((yj-yi)||1e-12)+xi);
    if(intersect)inside=!inside;
  }
  return inside;
}
function _ptInPolygon(pt,coords){
  if(!coords||!coords.length||!_ptInRing(pt,coords[0]))return false;
  for(var i=1;i<coords.length;i++){if(_ptInRing(pt,coords[i]))return false;}
  return true;
}
function _ptInGeometry(pt,geom){
  if(!geom)return false;
  if(geom.type==='Polygon')return _ptInPolygon(pt,geom.coordinates);
  if(geom.type==='MultiPolygon'){
    for(var i=0;i<geom.coordinates.length;i++){
      if(_ptInPolygon(pt,geom.coordinates[i]))return true;
    }
  }
  return false;
}
function _featureCentroid(f){
  var g=f&&f.geometry;if(!g)return null;
  var sx=0,sy=0,n=0;
  function add(ring){for(var i=0;i<ring.length;i++){sx+=ring[i][0];sy+=ring[i][1];n++;}}
  if(g.type==='Polygon'){add(g.coordinates[0]);}
  else if(g.type==='MultiPolygon'){
    var best=null,bestN=0;
    for(var i=0;i<g.coordinates.length;i++){
      var r=g.coordinates[i][0];
      if(r.length>bestN){bestN=r.length;best=r;}
    }
    if(best)add(best);
  }
  return n>0?[sx/n,sy/n]:null;
}
function _ensureTractCentroids(){
  if(_tractCentroids)return;
  _tractCentroids=GJ.features.map(_featureCentroid);
}
function _ensureTractJuris(){
  if(_tractJurisName)return;
  _ensureTractCentroids();
  _tractJurisName=GJ.features.map(function(_,i){
    var pt=_tractCentroids[i];
    if(!pt||typeof JURIS==='undefined'||!JURIS)return "";
    var fs=JURIS.features;
    for(var j=0;j<fs.length;j++){
      if(_ptInGeometry(pt,fs[j].geometry))return fs[j].properties.NAME||"";
    }
    return "";
  });
}
function _expJurisList(){
  if(_expJurisNames)return _expJurisNames;
  if(typeof JURIS==='undefined'||!JURIS){_expJurisNames=[];return _expJurisNames;}
  var seen={};
  JURIS.features.forEach(function(f){
    var n=f.properties&&f.properties.NAME;
    if(n)seen[n]=true;
  });
  _expJurisNames=Object.keys(seen).sort(function(a,b){return a.localeCompare(b);});
  return _expJurisNames;
}

function _expSyncScope(){
  var scope=document.querySelector('input[name=exp-scope]:checked');
  var mode=scope?scope.value:'all';
  var list=document.getElementById('exp-juris-list');
  if(list)list.classList.toggle('disabled',mode!=='juris');
  var allBtn=document.getElementById('exp-juris-all');
  var noneBtn=document.getElementById('exp-juris-none');
  if(allBtn)allBtn.disabled=(mode!=='juris');
  if(noneBtn)noneBtn.disabled=(mode!=='juris');
}
function _expJurisSelectAll(checked){
  var boxes=document.querySelectorAll('#exp-juris-list input[type=checkbox]');
  for(var i=0;i<boxes.length;i++)boxes[i].checked=!!checked;
}
function _expRenderJurisList(){
  var list=document.getElementById('exp-juris-list');
  if(!list)return;
  var names=_expJurisList();
  if(!names.length){
    list.innerHTML='<div style="font-style:italic;color:#999;font-size:11px">No jurisdiction boundaries available.</div>';
    return;
  }
  var html=names.map(function(n){
    var id='exp-jcb-'+n.replace(/[^A-Za-z0-9]+/g,'_');
    return '<div class="exp-juris-row"><label><input type="checkbox" id="'+id+'" value="'+n.replace(/"/g,'&quot;')+'">'+n+'</label></div>';
  }).join('');
  list.innerHTML=html;
}

function openExportModal(){
  _expRenderJurisList();
  var allCount=document.getElementById('exp-all-count');
  if(allCount)allCount.textContent='('+GJ.features.length+' tracts)';
  var allRadio=document.querySelector('input[name=exp-scope][value=all]');
  if(allRadio)allRadio.checked=true;
  _expSyncScope();
  document.getElementById('export-modal').classList.add('open');
}
function closeExportModal(){
  document.getElementById('export-modal').classList.remove('open');
}

function _selectedExportFormat(){
  var r=document.querySelector('input[name=exp-format]:checked');
  return r?r.value:'csv';
}
function _selectedExportScope(){
  var r=document.querySelector('input[name=exp-scope]:checked');
  return r?r.value:'all';
}
function _selectedJurisdictions(){
  var out=[];
  var boxes=document.querySelectorAll('#exp-juris-list input[type=checkbox]:checked');
  for(var i=0;i<boxes.length;i++)out.push(boxes[i].value);
  return out;
}

function _selectedTractIndices(scope,jurisNames){
  if(scope==='all'){
    var idx=[];for(var i=0;i<GJ.features.length;i++)idx.push(i);return idx;
  }
  _ensureTractCentroids();
  var nameSet={};jurisNames.forEach(function(n){nameSet[n]=true;});
  if(typeof JURIS==='undefined'||!JURIS)return [];
  var polysByName={};
  JURIS.features.forEach(function(f){
    var n=f.properties&&f.properties.NAME;
    if(!n||!nameSet[n])return;
    (polysByName[n]=polysByName[n]||[]).push(f.geometry);
  });
  var picked=[];
  for(var k=0;k<GJ.features.length;k++){
    var pt=_tractCentroids[k];if(!pt)continue;
    var hit=false;
    for(var name in polysByName){
      var geoms=polysByName[name];
      for(var g=0;g<geoms.length;g++){
        if(_ptInGeometry(pt,geoms[g])){hit=true;break;}
      }
      if(hit)break;
    }
    if(hit)picked.push(k);
  }
  return picked;
}

function _exportColumnSpec(){
  var cols=[{key:'GEOID',label:'GEOID'},{key:'jurisdiction',label:'jurisdiction'}];
  VM.forEach(function(v){cols.push({key:v.key,label:v.key,vm:true});});
  var inclPctile=true;
  var pctileBox=document.getElementById('exp-incl-pctile');
  if(pctileBox)inclPctile=!!pctileBox.checked;
  if(inclPctile){
    VM.forEach(function(v){
      cols.push({key:v.key+'_sd_pctile',label:v.key+'_sd_pctile',pctOf:v.key});
    });
  }
  var hasScn=(curScenario!=='present'&&curPeriod);
  if(hasScn){
    VM.forEach(function(v){
      if(DELTA_MAP[v.key])cols.push({key:v.key+'_adj',label:v.key+'_adj',adjOf:v.key});
    });
    cols.push({key:'scenario',label:'scenario'});
    cols.push({key:'period',label:'period'});
  }
  if(results){
    cols.push({key:'composite_risk',label:'composite_risk'});
    cols.push({key:'composite_driver',label:'composite_driver'});
    if(results.exposure)cols.push({key:'exposure_score',label:'exposure_score'});
    if(results.sensitivity)cols.push({key:'sensitivity_score',label:'sensitivity_score'});
    if(results.adaptive)cols.push({key:'adaptive_score',label:'adaptive_score'});
  }
  return cols;
}

function _rowValueFor(col,i,pr){
  if(col.key==='GEOID')return pr.GEOID;
  if(col.key==='jurisdiction'){_ensureTractJuris();return _tractJurisName[i];}
  if(col.key==='scenario')return curScenario;
  if(col.key==='period')return curPeriod;
  if(col.key==='composite_risk')return (pr._comp!==undefined&&pr._comp!==null)?pr._comp:null;
  if(col.key==='composite_driver')return pr._drv||null;
  if(col.key==='exposure_score')return (pr._exposure!==undefined)?pr._exposure:null;
  if(col.key==='sensitivity_score')return (pr._sensitivity!==undefined)?pr._sensitivity:null;
  if(col.key==='adaptive_score')return (pr._adaptive!==undefined)?pr._adaptive:null;
  if(col.pctOf){
    var arr=rawPct[col.pctOf];if(!arr)return null;
    var p=arr[i];if(p===undefined||p===null)return null;
    return Math.round(p*10)/10;
  }
  if(col.adjOf){
    var v=pr[col.adjOf];if(v===null||v===undefined)return null;
    var d=getDeltaArr(col.adjOf);if(!d)return v;
    var adj=v+d[i];return Math.round(adj*1000)/1000;
  }
  if(col.vm){var x=pr[col.key];return (x===undefined)?null:x;}
  return null;
}

function _csvEscape(v){
  if(v===null||v===undefined)return '';
  var s=String(v);
  if(/[",\\n\\r]/.test(s))return '"'+s.replace(/"/g,'""')+'"';
  return s;
}
function _featuresToCSV(indices,cols){
  var lines=[cols.map(function(c){return c.label;}).join(',')];
  for(var k=0;k<indices.length;k++){
    var i=indices[k];
    var pr=GJ.features[i].properties;
    var row=cols.map(function(c){return _csvEscape(_rowValueFor(c,i,pr));});
    lines.push(row.join(','));
  }
  return lines.join('\\n')+'\\n';
}
function _featuresToGeoJSON(indices,cols){
  var feats=indices.map(function(i){
    var src=GJ.features[i];
    var pr=src.properties;
    var props={};
    cols.forEach(function(c){props[c.label]=_rowValueFor(c,i,pr);});
    return {type:'Feature',properties:props,geometry:src.geometry};
  });
  return JSON.stringify({type:'FeatureCollection',features:feats});
}

function _slugify(s){return String(s||'').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'');}
function _exportFilename(scope,jurisNames,fmt){
  var parts=['san_diego_heat_risk'];
  if(scope==='juris'&&jurisNames.length){
    var slugs=jurisNames.map(_slugify).filter(Boolean);
    parts.push(slugs.length<=3?slugs.join('-'):(slugs.length+'_cities'));
  }else{
    parts.push('all_tracts');
  }
  var scn=(curScenario==='present')?'present':(curScenario+(curPeriod?'_'+curPeriod:''));
  parts.push(scn);
  return parts.join('_')+'.'+(fmt==='geojson'?'geojson':'csv');
}
function _triggerDownload(text,filename,mime){
  var blob=new Blob([text],{type:mime+';charset=utf-8'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');
  a.href=url;a.download=filename;
  document.body.appendChild(a);a.click();
  setTimeout(function(){document.body.removeChild(a);URL.revokeObjectURL(url);},0);
}

function doExportDownload(){
  var btn=document.getElementById('exp-download-btn');
  var fmt=_selectedExportFormat();
  var scope=_selectedExportScope();
  var jurisNames=(scope==='juris')?_selectedJurisdictions():[];
  if(scope==='juris'&&jurisNames.length===0){
    alert('Select at least one jurisdiction, or switch to "Entire map".');
    return;
  }
  var indices=_selectedTractIndices(scope,jurisNames);
  if(indices.length===0){
    alert('No tracts fall within the selected jurisdictions.');
    return;
  }
  if(btn)btn.disabled=true;
  try{
    var cols=_exportColumnSpec();
    var fname=_exportFilename(scope,jurisNames,fmt);
    if(fmt==='geojson'){
      _triggerDownload(_featuresToGeoJSON(indices,cols),fname,'application/geo+json');
    }else{
      _triggerDownload(_featuresToCSV(indices,cols),fname,'text/csv');
    }
    closeExportModal();
  }catch(err){
    console.error('Data export failed:',err);
    alert('Data export failed: '+err.message);
  }finally{
    if(btn)btn.disabled=false;
  }
}

document.addEventListener('keydown',function(e){
  if(e.key==='Escape'&&document.getElementById('export-modal').classList.contains('open')){
    closeExportModal();
  }
});

/* ================================================================
   INIT
   ================================================================ */
buildUI();
showLayerBtns();
// Restore CB-mode preference (state already loaded above; sync DOM + checkbox)
if(cbMode){
  document.body.classList.add('cb-mode');
  var cbBox=document.getElementById('cb-cbmode');
  if(cbBox)cbBox.checked=true;
}
console.log("DIY PCA loaded. Features:",GJ.features.length,"Vars:",VM.length);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

