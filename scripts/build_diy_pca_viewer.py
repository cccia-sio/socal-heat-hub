"""Build interactive DIY PCA viewer for heat risk exploration.

Generates a self-contained HTML file where users can assign variables to
pillars (Exposure, Sensitivity, Adaptive Capacity), run PCA client-side
via Jacobi eigendecomposition, and view results on a Leaflet choropleth.

Usage:
    python scripts/build_diy_pca_viewer.py

Output:
    main/heat_risk_map.html
"""

import json
import logging
import urllib.parse
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
OUTPUT_PATH = ROOT / "main" / "heat_risk_map.html"
TRACTS_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "tracts_sd.geojson"
JURIS_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "jurisdictions_sd.geojson"
TRIBAL_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "tribal_lands_sd.geojson"
UNINCORP_PATH = ROOT / "data" / "exposure" / "raw" / "census" / "unincorporated_sd.geojson"
COOLZONES_PATH = ROOT / "data" / "cool_zones" / "coolzone_sites.csv"

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
    "pct_pre1980",
    "pct_pre1960",
    "pct_renter",
    "pct_impervious",
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
    # NOTE: Raw humidity (mean_rh_hrrr, mean_rh_hot_hours, p95_dewpoint) is
    # intentionally NOT exposed as exposure variables. Humidity is a hazard
    # *modifier*, already integrated into WBGT (which is the "Humid heat" mode);
    # as a standalone variable it loaded backwards. See the HEAT_MODES design.
    # ── Sensitivity ──────────────────────────────────────────────────────
    # Health — CDC PLACES
    ("bphigh_prev", "High Blood Pressure", "sensitivity", "Health"),
    ("casthma_prev", "Current Asthma", "sensitivity", "Health"),
    ("chd_prev", "Coronary Heart Disease", "sensitivity", "Health"),
    ("cognition_prev", "Cognitive Difficulty", "sensitivity", "Health"),
    ("copd_prev", "Chronic Obstructive Pulmonary Disease (COPD)", "sensitivity", "Health"),
    ("depression_prev", "Depression", "sensitivity", "Health"),
    ("diabetes_prev", "Diabetes", "sensitivity", "Health"),
    ("stroke_prev", "Stroke", "sensitivity", "Health"),
    ("obesity_prev", "Obesity", "sensitivity", "Health"),
    ("mhlth_prev", "Frequent Mental Distress", "sensitivity", "Health"),
    ("leb_pctile", "Life Expectancy", "sensitivity", "Health"),
    # Demographics — SVI
    ("EP_AGE65", "Age 65+", "sensitivity", "Sociodemographics"),
    ("pct_senior_alone", "Seniors Living Alone (%)", "sensitivity", "Sociodemographics"),
    ("EP_AGE17", "Age 17 & Under", "sensitivity", "Sociodemographics"),
    ("EP_DISABL", "Disability", "sensitivity", "Sociodemographics"),
    ("EP_LIMENG", "Limited English", "sensitivity", "Sociodemographics"),
    ("EP_SNGPNT", "Single-Parent Household", "sensitivity", "Sociodemographics"),
    ("EP_POV150", "Below 150% Poverty", "sensitivity", "Sociodemographics"),
    ("EP_UNEMP", "Unemployed", "sensitivity", "Sociodemographics"),
    ("EP_HBURD", "Housing Cost Burden", "sensitivity", "Sociodemographics"),
    ("EP_NOHSDP", "No High School Diploma", "sensitivity", "Sociodemographics"),
    ("EP_UNINSUR", "Uninsured", "sensitivity", "Sociodemographics"),
    ("EP_MINRTY", "Minority", "sensitivity", "Sociodemographics"),
    ("EP_MUNIT", "Multi-Unit Structures", "sensitivity", "Sociodemographics"),
    ("EP_MOBILE", "Mobile Homes", "sensitivity", "Sociodemographics"),
    ("EP_CROWD", "Crowding", "sensitivity", "Sociodemographics"),
    ("EP_NOVEH", "No Vehicle", "sensitivity", "Sociodemographics"),
    ("EP_GROUPQ", "Group Quarters", "sensitivity", "Sociodemographics"),
    ("EP_NOINT", "No Internet", "sensitivity", "Sociodemographics"),
    # Occupational exposure — ACS C24030 (residence-based)
    ("pct_outdoor_workers", "Outdoor Workers (Ag+Const, %)", "sensitivity", "Sociodemographics"),
    ("pct_ag_workers", "Agricultural Workers (%)", "sensitivity", "Sociodemographics"),
    # Social Determinants of Health (SDOH) — CDC PLACES, flagged from the CA
    # Vulnerable Communities Platform; all vulnerabilities (higher → worse risk).
    # No standalone group: the social-isolation measures sit under Health, the
    # material-hardship measures under Sociodemographics.
    ("loneliness_prev", "Loneliness", "sensitivity", "Health"),
    ("emotionspt_prev", "Lacks Social/Emotional Support", "sensitivity", "Health"),
    ("foodinsecu_prev", "Food Insecurity", "sensitivity", "Sociodemographics"),
    ("foodstamp_prev", "Food Stamp / SNAP Receipt", "sensitivity", "Sociodemographics"),
    ("housinsecu_prev", "Housing Insecurity", "sensitivity", "Sociodemographics"),
    ("lacktrpt_prev", "Lacks Reliable Transportation", "sensitivity", "Sociodemographics"),
    ("shututility_prev", "Utility Shut-Off Threat", "sensitivity", "Sociodemographics"),
    # Energy affordability — DOE LEAD (all-households avg energy burden, % income)
    ("energy_burden", "Energy Burden (% income)", "sensitivity", "Sociodemographics"),
    # SVI Themes
    ("RPL_THEMES", "Overall SVI", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME1", "Socioeconomic Status", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME2", "Household Disability/Language", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME3", "Racial/Ethnic Minority", "sensitivity", "Social Vulnerability Index (SVI)"),
    ("RPL_THEME4", "Housing/Transportation", "sensitivity", "Social Vulnerability Index (SVI)"),
    # Heat-attributable hospitalization rate ratios (CalHeatScore-style
    # relative-scale estimates). Per-ZCTA, crosswalked to tract by
    # centroid-in-polygon (see scripts/build_hosp_tract_xwalk.py). Higher
    # value → more hospitalizations on heat days vs typical days → higher
    # sensitivity, matching the sensitivity-pillar polarity default.
    ("heat_hosp_day", "Heat-Day Hospitalization Rate Ratio", "sensitivity", "Health"),
    ("heat_hosp_night", "Warm-Night Hospitalization Rate Ratio", "sensitivity", "Health"),
    # ── Adaptive Capacity ────────────────────────────────────────────────
    # Air Conditioning
    ("ac_prob", "AC Prevalence (LACE)", "adaptive", "Air Conditioning (AC)"),
    ("ac_prob_romitti", "AC Prevalence (Romitti)", "adaptive", "Air Conditioning (AC)"),
    # Built Environment
    ("treecanopy_pctile", "Tree Canopy", "adaptive", "Built Environment"),
    ("parkaccess_pctile", "Park Access", "adaptive", "Built Environment"),
    ("pct_pre1980", "Pre-1980 Housing (%)", "adaptive", "Built Environment"),
    ("pct_pre1960", "Pre-1960 Housing (%)", "adaptive", "Built Environment"),
    ("pct_renter", "Renter-Occupied (%)", "adaptive", "Built Environment"),
    ("pct_green", "Green Space (%)", "adaptive", "Built Environment"),
    ("pct_impervious", "Impervious Surface (%)", "adaptive", "Built Environment"),
    # Socioeconomic — HPI
    ("income_pctile", "Income", "adaptive", "Socioeconomic"),
    ("abovepoverty_pctile", "Above Poverty", "adaptive", "Socioeconomic"),
    ("economic_pctile", "Economic", "adaptive", "Socioeconomic"),
    ("education_pctile", "Education", "adaptive", "Socioeconomic"),
    ("houseing_pctile", "Housing Quality", "adaptive", "Socioeconomic"),
    ("transportation_pctile", "Transportation", "adaptive", "Socioeconomic"),
    ("social_pctile", "Social Cohesion", "adaptive", "Socioeconomic"),
    ("neighborhood_pctile", "Neighborhood Quality", "adaptive", "Socioeconomic"),
    ("insured_pctile", "Health Insurance", "adaptive", "Socioeconomic"),
    ("automobile_pctile", "Automobile Access", "adaptive", "Socioeconomic"),
    ("homeownership_pctile", "Homeownership", "adaptive", "Socioeconomic"),
    ("commute_pctile", "Commute Time", "adaptive", "Socioeconomic"),
    ("voting_pctile", "Voter Participation", "adaptive", "Socioeconomic"),
    ("retail_pctile", "Retail Food Access", "adaptive", "Socioeconomic"),
    # Pollution Burden
    ("pollution_pctile", "Overall Pollution", "sensitivity", "Pollution Burden"),
    ("ozone_pctile", "Ozone", "sensitivity", "Pollution Burden"),
    ("pm25_pctile", "PM 2.5", "sensitivity", "Pollution Burden"),
    ("dieselpm_pctile", "Diesel PM", "sensitivity", "Pollution Burden"),
]

# ── Exposure heat modes ──────────────────────────────────────────────────
# The exposure pillar is NOT a PCA blend — it's a single-select "heat mode."
# Each mode is a distinct physical hazard; its score is the mean of its
# z-scored member layers, percentile-ranked (no PCA). Daytime is the default.
# Rationale: heat hazards are physically distinct (daytime vs. nighttime heat
# are spatially anti-correlated, r=-0.18), so collapsing them via PCA produced
# an inland/coastal geography axis and a backwards humidity loading.
# (key, label, [member layer keys], one-line description)
HEAT_MODES = [
    ("daytime", "Daytime heat",
     ["mean_tmax", "p85_tmax", "p99_tmax", "max_tmax"],
     "Daytime air temperature — typical and extreme highs. Peaks inland."),
    ("nighttime", "Nighttime heat",
     ["mean_tmin_overnight", "p90_tmin_overnight", "no_relief_fraction"],
     "Overnight heat and lack of relief. Peaks in dense urban/coastal tracts."),
    ("wbgt", "Humid heat (WBGT)",
     ["p85_wbgt", "p99_wbgt"],
     "Wet-Bulb Globe Temperature — integrates heat, humidity, sun, and wind."),
    ("surface", "Surface temperature",
     ["mean_lst"],
     "Satellite land-surface temperature — radiant/urban-heat-island signal."),
]
DEFAULT_HEAT_MODE = "daytime"

# ── Source URLs per variable ─────────────────────────────────────────────
_SRC_MESOWEST = "https://synopticdata.com/"
_SRC_PLACES = "https://www.cdc.gov/places/index.html"
_SRC_SVI = "https://www.atsdr.cdc.gov/place-health/php/svi/index.html"
_SRC_HPI = "https://www.healthyplacesindex.org/"
_SRC_LACE = "https://www.census.gov/newsroom/press-releases/2026/2023-lace.html"
# Point at the Harvard Dataverse deposit, not the article: the deposit is the
# redistributable artifact. Paper is CC BY (PNAS Nexus, doi:10.1093/pnasnexus/pgac210).
_SRC_ROMITTI = "https://doi.org/10.7910/DVN/HWFVP6"
_SRC_ACS_B25034 = "https://censusreporter.org/tables/B25034/"
_SRC_ACS_B25003 = "https://censusreporter.org/tables/B25003/"
_SRC_ACS_C24030 = "https://censusreporter.org/tables/C24030/"
_SRC_ACS_B11007 = "https://censusreporter.org/tables/B11007/"
_SRC_SANGIS = "https://sdgis-sandag.opendata.arcgis.com/"
_SRC_LANDSAT = "https://www.usgs.gov/landsat-missions"
_SRC_HRRR = "https://rapidrefresh.noaa.gov/hrrr/"
_SRC_CALHEATSCORE = "https://calheatscore.calepa.ca.gov/"
_SRC_LEAD = "https://data.openei.org/submissions/6219"
_SRC_NLCD = "https://www.mrlc.gov/data"

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
    "obesity_prev": _SRC_PLACES, "mhlth_prev": _SRC_PLACES,
    # SDOH measures (CDC PLACES) — social isolation + material hardship
    "loneliness_prev": _SRC_PLACES, "emotionspt_prev": _SRC_PLACES,
    "foodinsecu_prev": _SRC_PLACES, "foodstamp_prev": _SRC_PLACES,
    "housinsecu_prev": _SRC_PLACES, "lacktrpt_prev": _SRC_PLACES,
    "shututility_prev": _SRC_PLACES,
    # Sensitivity — DOE LEAD energy burden
    "energy_burden": _SRC_LEAD,
    # Sensitivity — SVI
    "EP_AGE65": _SRC_SVI, "EP_AGE17": _SRC_SVI, "EP_DISABL": _SRC_SVI,
    "EP_LIMENG": _SRC_SVI, "EP_SNGPNT": _SRC_SVI, "EP_POV150": _SRC_SVI,
    "EP_UNEMP": _SRC_SVI, "EP_HBURD": _SRC_SVI, "EP_NOHSDP": _SRC_SVI,
    "EP_UNINSUR": _SRC_SVI, "EP_MINRTY": _SRC_SVI, "EP_MUNIT": _SRC_SVI,
    "EP_MOBILE": _SRC_SVI, "EP_CROWD": _SRC_SVI, "EP_NOVEH": _SRC_SVI,
    "EP_GROUPQ": _SRC_SVI, "EP_NOINT": _SRC_SVI,
    "RPL_THEMES": _SRC_SVI, "RPL_THEME1": _SRC_SVI, "RPL_THEME2": _SRC_SVI,
    "RPL_THEME3": _SRC_SVI, "RPL_THEME4": _SRC_SVI,
    # Sensitivity — Heat-attributable hospitalization (CalHeatScore-style,
    # per-ZCTA, centroid-crosswalked to tract).
    "heat_hosp_day": _SRC_CALHEATSCORE, "heat_hosp_night": _SRC_CALHEATSCORE,
    # Adaptive — Census LACE 2023 (Local Air Conditioning Estimates)
    "ac_prob": _SRC_LACE,
    # Adaptive — Romitti et al. (2022)
    "ac_prob_romitti": _SRC_ROMITTI,
    # Adaptive — ACS
    "pct_pre1980": _SRC_ACS_B25034, "pct_pre1960": _SRC_ACS_B25034,
    "pct_renter": _SRC_ACS_B25003,
    # Sensitivity — ACS occupational exposure
    "pct_outdoor_workers": _SRC_ACS_C24030,
    "pct_ag_workers": _SRC_ACS_C24030,
    # Sensitivity — ACS seniors living alone (B11007)
    "pct_senior_alone": _SRC_ACS_B11007,
    # Adaptive — SanGIS
    "pct_green": _SRC_SANGIS,
    "pct_impervious": _SRC_NLCD,
    # HPI (adaptive + sensitivity)
    "treecanopy_pctile": _SRC_HPI, "parkaccess_pctile": _SRC_HPI,
    "income_pctile": _SRC_HPI, "abovepoverty_pctile": _SRC_HPI,
    "economic_pctile": _SRC_HPI, "education_pctile": _SRC_HPI,
    "houseing_pctile": _SRC_HPI,
    "transportation_pctile": _SRC_HPI, "social_pctile": _SRC_HPI,
    "neighborhood_pctile": _SRC_HPI, "insured_pctile": _SRC_HPI,
    "automobile_pctile": _SRC_HPI, "homeownership_pctile": _SRC_HPI,
    "commute_pctile": _SRC_HPI,
    "voting_pctile": _SRC_HPI, "retail_pctile": _SRC_HPI,
    "leb_pctile": _SRC_HPI, "pollution_pctile": _SRC_HPI,
    "ozone_pctile": _SRC_HPI, "pm25_pctile": _SRC_HPI,
    "dieselpm_pctile": _SRC_HPI,
}

# Human-readable name for each source URL. The "(Source)" links in the sidebar
# only carry a URL; the exported data dictionary names the provenance in words
# so a downloaded file is self-describing without a network round-trip.
SOURCE_NAMES = {
    _SRC_MESOWEST: "MesoWest / Synoptic Data (+ CIMIS), IDW to tract centroids",
    _SRC_PLACES: "CDC PLACES (model-based small-area estimates)",
    _SRC_SVI: "CDC/ATSDR Social Vulnerability Index 2022",
    _SRC_HPI: "Healthy Places Index (HPI) 3.0",
    _SRC_LACE: "U.S. Census Bureau LACE 2023",
    _SRC_ROMITTI: "Romitti et al. 2022, PNAS Nexus (modeled AC prevalence)",
    _SRC_ACS_B25034: "ACS 5-year, table B25034 (year structure built)",
    _SRC_ACS_B25003: "ACS 5-year, table B25003 (tenure)",
    _SRC_ACS_C24030: "ACS 5-year, table C24030 (industry by sex)",
    _SRC_ACS_B11007: "ACS 5-year, table B11007 (households by age)",
    _SRC_SANGIS: "SANDAG / SanGIS regional land use",
    _SRC_LANDSAT: "USGS Landsat (Collection 2 surface temperature)",
    _SRC_HRRR: "NOAA HRRR 3 km analysis, WBGT via Liljegren/Dimiceli",
    _SRC_CALHEATSCORE: "CalHeatScore (CalEPA / OEHHA), per-ZCTA, crosswalked to tract",
    _SRC_LEAD: "DOE LEAD Tool (2022)",
    _SRC_NLCD: "NLCD / MRLC land cover",
}

# ── Units, for the exported data dictionary ──────────────────────────────
# Resolved per variable so an analyst never has to guess whether a column is
# 0–1, 0–100, or degrees. Keep in sync with fmtCmpRaw() in HTML_TEMPLATE,
# which formats these same values for on-screen display.
_U_C = "degrees Celsius"
_U_PCT = "percent (0-100)"
_UNITS_EXPLICIT = {
    "mean_tmax": _U_C, "p85_tmax": _U_C, "p99_tmax": _U_C, "max_tmax": _U_C,
    "mean_lst": _U_C, "p85_wbgt": _U_C, "p99_wbgt": _U_C,
    "mean_tmin_overnight": _U_C, "p90_tmin_overnight": _U_C,
    "p95_dewpoint": _U_C,
    "no_relief_fraction": "fraction of May-Oct nights above 20 C (0-1)",
    "ac_prob": "share of homes with air conditioning (0-1)",
    "ac_prob_romitti": "share of homes with air conditioning (0-1)",
    "energy_burden": "percent of household income spent on energy",
    "heat_hosp_day": (
        "relative-scale effect estimate, centered near 0 "
        "(higher = more excess hospitalizations on extreme-heat days)"
    ),
    "heat_hosp_night": (
        "relative-scale effect estimate, centered near 0 "
        "(higher = more excess hospitalizations after extreme-warm nights)"
    ),
    "extreme_day_fraction": "fraction of days (0-1)",
}


def _var_units(key: str) -> str:
    """Units string for a variable key, used by the exported data dictionary."""
    if key in _UNITS_EXPLICIT:
        return _UNITS_EXPLICIT[key]
    if key.startswith("RPL_"):
        return "percentile rank as published by CDC/ATSDR (0-1)"
    if key.startswith(("EP_", "pct_")):
        return _U_PCT
    if key.endswith("_prev"):
        return "percent of adults 18+ (0-100)"
    if key.endswith("_pctile"):
        return "HPI percentile (0-100)"
    return ""


# Sources we cannot demonstrate are redistributable. Their raw values are dropped
# from the "Download Data" export so the downloadable dataset stays CC BY-clean;
# they still render on the map, used under each source's own terms with
# attribution. Adding a source here is the ONLY switch needed — NO_EXPORT, the
# data dictionary and tests/test_export_licensing.py all derive from it.
RESTRICTED_EXPORT_SOURCES = {_SRC_HPI, _SRC_CALHEATSCORE}

# Why each restricted source is withheld. Surfaced verbatim in the exported data
# dictionary so an absent column is explained rather than mysterious.
RESTRICTED_REASONS = {
    _SRC_HPI: (
        "Shown on the map but omitted from this download: the Healthy Places "
        "Index is licensed for non-commercial use, which conflicts with the "
        "CC BY 4.0 terms of this file. Get it from HPI directly."
    ),
    _SRC_CALHEATSCORE: (
        "Shown on the map but omitted from this download: these estimates come "
        "from a collaborator-supplied file with no published license or public "
        "download, so we cannot establish a right to redistribute them. "
        "CalHeatScore publishes daily scores and a REST API; per-ZCTA effect "
        "estimates are not part of that public release. Request them from "
        "CalHeatScore@oehha.ca.gov."
    ),
}

# Where user feedback goes. A mailto rather than a form or issue tracker: the
# repo is private (so its Issues page 404s for the public), and any hosted form
# would be a third-party data flow that the no-tracking promise would then have
# to carve out. Subject is prefilled so replies are filterable.
FEEDBACK_EMAIL = "cjmack@ucsd.edu"
FEEDBACK_SUBJECT = "San Diego Heat Risk Explorer - feedback"

# Attribution stamped onto every row of the "Download Data" export — as CSV
# columns and as GeoJSON feature properties — so a file that travels away from
# this site still carries its license and a link to the per-variable source
# table. Keep ASCII-only: the CSV ships without a BOM, and non-ASCII would
# mojibake when opened in Excel.
EXPORT_META = {
    "citation": (
        "San Diego Heat Risk Explorer - Southern California Extreme Heat "
        "Research Hub & San Diego Regional Climate Collaborative. Variables "
        "whose source terms bar redistribution (Healthy Places Index; "
        "CalHeatScore heat-hospitalization estimates) are shown on the map "
        "but omitted from this export - see the data dictionary."
    ),
    "license": "CC BY 4.0",
    "sources_url": "https://cccia-sio.github.io/socal-heat-hub/userguide/#data",
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
# Columns sourced from the HPI 3.0 complete data file (xlsx, 2020-tract xwalk).
# Single file replaces the previous HPI 2.x domain CSV + indicator CSV pair.
# Three columns are renamed on load to preserve our existing keys:
#   percapitaincome_pctile → income_pctile
#   housing_pctile         → houseing_pctile  (keeps our typo'd key)
#   clean_enviro_pctile    → pollution_pctile (HPI 3.0 env composite)
# Two HPI 2.x columns were discontinued in HPI 3.0 and removed from VARIABLES:
#   supermkts_pctile, healthcareaccess_pctile
_HPI_COLS = [
    # Domain-level rollups
    "treecanopy_pctile", "parkaccess_pctile", "income_pctile",
    "abovepoverty_pctile", "economic_pctile", "education_pctile",
    "transportation_pctile", "social_pctile",
    "houseing_pctile", "neighborhood_pctile", "pollution_pctile", "leb_pctile",
    # Indicator-level
    "insured_pctile", "automobile_pctile", "homeownership_pctile",
    "commute_pctile", "voting_pctile", "retail_pctile",
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

    # ── Adaptive — AC prevalence (Census LACE 2023) ──
    ac = pd.read_csv(
        ROOT / "data/adaptive_capacity/aircon/LACE_23_Tract.csv",
        dtype={"GEO_ID": str},
    )
    # GEO_ID like '1400000US06073020811' -> 11-digit GEOID after 'US'
    ac["GEOID"] = ac["GEO_ID"].str.split("US").str[-1].str.zfill(11)
    # Census uses negative "jam" sentinels for suppressed/undefined estimates;
    # valid AC_PE is a percent in [0, 100]. Convert to a 0–1 prevalence fraction
    # to match the existing ac_prob semantics (higher = more AC = more capacity).
    ac["AC_PE"] = pd.to_numeric(ac["AC_PE"], errors="coerce")
    ac.loc[~ac["AC_PE"].between(0, 100), "AC_PE"] = np.nan
    ac["ac_prob"] = ac["AC_PE"] / 100.0
    tracts = _safe_merge(tracts, ac, ["ac_prob"])

    # ── Adaptive — AC prevalence (Romitti et al. 2022) ──
    # Second, independent AC-prevalence estimate kept alongside LACE.
    rom = pd.read_csv(
        ROOT / "data/adaptive_capacity/aircon/tract_ac_prevalence_sd_summary.csv",
        dtype={"GEOID": str},
    )
    rom["GEOID"] = rom["GEOID"].str.zfill(11)
    rom = rom.rename(columns={"ac_prob": "ac_prob_romitti"})
    tracts = _safe_merge(tracts, rom, ["ac_prob_romitti"])

    # ── Adaptive + Sensitivity — HPI 3.0 (complete file, 2020-tract xwalk) ──
    # Single statewide xlsx with all indicator + domain pctile columns.
    # Three columns are renamed to preserve our existing keys so downstream
    # code (VARIABLES, INVERTED_VARS, VAR_SOURCES) doesn't have to change.
    # HPI 3.0 publishes pctiles on a 0–1 scale; HPI 2.x and the rest of the
    # viewer's variables use 0–100, so we rescale on load.
    hpi_path = ROOT / "data/hpi/hpi3_xwalked_complete_file_2020tracts.xlsx"
    if hpi_path.exists():
        hpi = pd.read_excel(hpi_path, sheet_name="HPI 3.0 Data")
        hpi["GEOID"] = hpi["GEO_ID"].astype(str).str.zfill(11)
        hpi = hpi.rename(columns={
            "percapitaincome_pctile": "income_pctile",
            "housing_pctile":         "houseing_pctile",
            "clean_enviro_pctile":    "pollution_pctile",
        })
        for col in _HPI_COLS:
            if col in hpi.columns:
                hpi[col] = pd.to_numeric(hpi[col], errors="coerce") * 100.0
        tracts = _safe_merge(tracts, hpi, _HPI_COLS)

    # ── Adaptive — Building age (ACS B25034) ──
    bldg_path = ROOT / "data/adaptive_capacity/building_age/acs_b25034_year_built_sd_tracts.csv"
    if bldg_path.exists():
        bldg = pd.read_csv(bldg_path, dtype={"GEOID": str})
        bldg["GEOID"] = bldg["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, bldg, ["pct_pre1980", "pct_pre1960"])

    # ── Adaptive — Impervious surface (NLCD, tract zonal mean) ──
    # Mean % impervious cover per tract; see scripts/build_impervious_tracts.py
    # (same NLCD raster as the Data-Layer overlay). More impervious → less
    # natural cooling → lower adaptive capacity (a vulnerability; INVERTED_VARS).
    imperv_path = ROOT / "data/adaptive_capacity/impervious/impervious_pct_sd_tracts.csv"
    if imperv_path.exists():
        imperv = pd.read_csv(imperv_path, dtype={"GEOID": str})
        imperv["GEOID"] = imperv["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, imperv, ["pct_impervious"])

    # ── Adaptive + Sensitivity — ACS demographics (renter occupancy, outdoor-worker share) ──
    acs_path = ROOT / "data/exposure/raw/census/acs_demographics_sd.csv"
    if acs_path.exists():
        acs = pd.read_csv(acs_path, dtype={"GEOID": str})
        acs["GEOID"] = acs["GEOID"].str.zfill(11)
        acs["pct_ag_workers"] = (acs["employed_agriculture"] / acs["total_employed"].replace(0, float("nan"))) * 100
        tracts = _safe_merge(tracts, acs, ["pct_renter", "pct_outdoor_workers", "pct_ag_workers"])

    # ── Sensitivity — ACS seniors living alone (B11007) ──
    senior_path = ROOT / "data/sensitivity/acs_senior_alone_sd_tracts.csv"
    if senior_path.exists():
        senior = pd.read_csv(senior_path, dtype={"GEOID": str})
        senior["GEOID"] = senior["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, senior, ["pct_senior_alone"])

    # ── Sensitivity — Energy burden (DOE LEAD Tool 2022) ──
    # All-households average energy burden (% of income spent on energy) per
    # tract; see scripts/build_energy_burden.py. Higher → harder to afford
    # cooling → greater heat sensitivity (default sensitivity polarity).
    energy_path = ROOT / "data/sensitivity/energy_burden/lead_energy_burden_sd_tracts.csv"
    if energy_path.exists():
        energy = pd.read_csv(energy_path, dtype={"GEOID": str})
        energy["GEOID"] = energy["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, energy, ["energy_burden"])

    # ── Sensitivity — Heat-attributable hospitalization (ZCTA → tract centroid xwalk) ──
    # Generated by scripts/build_hosp_tract_xwalk.py from joined_rel.csv. Two
    # CalHeatScore-style relative-scale rate ratios (extreme daytime / nighttime
    # heat, 1-day lag) assigned per tract from the containing ZCTA.
    hosp_path = ROOT / "data/hospitalizations/tract_heat_hosp.csv"
    if hosp_path.exists():
        hosp = pd.read_csv(hosp_path, dtype={"GEOID": str})
        hosp["GEOID"] = hosp["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, hosp, ["heat_hosp_day", "heat_hosp_night"])

    # ── Context — Land use (tract_exploration.csv) ──
    ctx_path = ROOT / "data/context/tract_exploration.csv"
    if ctx_path.exists():
        ctx = pd.read_csv(ctx_path, dtype={"geoid": str})
        ctx["GEOID"] = ctx["geoid"].astype(str).str.zfill(11)
        tracts = _safe_merge(tracts, ctx, ["pct_green"])

    # ── Exploratory — UCLA Heat Maps excess-ER + CalHeatScore thresholds (ZIP→tract) ──
    # Display-only Data Layers (not PCA variables); built by
    # scripts/build_ucla_heat_tract_xwalk.py (centroid-in-polygon crosswalk).
    ucla_path = ROOT / "data/context/ucla_heat_tract.csv"
    ucla_cols = ["excess_er_rate", "chs1_th", "chs2_th", "chs3_th", "chs4_th"]
    if ucla_path.exists():
        ucla = pd.read_csv(ucla_path, dtype={"GEOID": str})
        ucla["GEOID"] = ucla["GEOID"].str.zfill(11)
        tracts = _safe_merge(tracts, ucla, ucla_cols)

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

    # UCLA exploratory display columns (Data Layers only, not PCA variables).
    extra += [c for c in ucla_cols if c in tracts.columns]

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


def _round_geojson_coords(text, ndigits=6):
    """Round every coordinate in a GeoJSON string; properties are untouched.

    to_json() writes full float precision (15 decimals), which roughly doubled the
    inlined geometry. 6 decimals is ~0.1 m.
    """
    obj = json.loads(text)

    def rnd(c):
        return round(c, ndigits) if isinstance(c, (int, float)) else [rnd(x) for x in c]

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "coordinates":
                    o[k] = rnd(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(obj)
    return json.dumps(obj)


def _script_safe(blob):
    """Escape `<` in data bound for an inline <script>.

    A string containing `</script>` would close the tag and run whatever follows,
    and some strings come from outside services (Cool Zones names are from a County
    feature service). `\\u003c` is the same character to JS and JSON.
    """
    return blob.replace("<", "\\u003c")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracts = load_data()

    # Compute CMIP6 tract-level deltas (None if files missing)
    deltas = compute_tract_deltas(tracts)

    # Serialize — ensure_ascii so no raw unicode in JS source
    geojson_str = _round_geojson_coords(tracts.to_json())
    var_meta_str = json.dumps(
        [{"key": v[0], "label": v[1], "pillar": v[2], "sub": v[3]}
         for v in VARIABLES if v[0] in tracts.columns],
        ensure_ascii=True,
    )
    var_sources_str = json.dumps(
        {k: v for k, v in VAR_SOURCES.items() if k in tracts.columns},
        ensure_ascii=True,
    )
    # url -> plain-English source name, and key -> units. Both exist only to
    # make the "Download Data" dictionary self-describing.
    source_names_str = json.dumps(SOURCE_NAMES, ensure_ascii=True)
    var_units_str = json.dumps(
        {v[0]: _var_units(v[0]) for v in VARIABLES if v[0] in tracts.columns},
        ensure_ascii=True,
    )
    # Variable keys excluded from the Download Data export (restricted source
    # terms). They stay on the map; only the bulk download omits them.
    no_export_str = json.dumps(
        sorted(k for k, v in VAR_SOURCES.items()
               if v in RESTRICTED_EXPORT_SOURCES and k in tracts.columns),
        ensure_ascii=True,
    )
    # varKey -> why it is withheld, for the data dictionary's notes column.
    no_export_reasons_str = json.dumps(
        {k: RESTRICTED_REASONS[v] for k, v in VAR_SOURCES.items()
         if v in RESTRICTED_EXPORT_SOURCES and k in tracts.columns
         and v in RESTRICTED_REASONS},
        ensure_ascii=True,
    )
    export_meta_str = json.dumps(EXPORT_META, ensure_ascii=True)
    # Heat modes: drop any layer missing from the data, then any empty mode.
    heat_modes_str = json.dumps(
        [{"key": k, "label": lbl, "layers": [c for c in cols if c in tracts.columns],
          "desc": desc}
         for k, lbl, cols, desc in HEAT_MODES
         if any(c in tracts.columns for c in cols)],
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

    # Load Data-Layer raster overlay metadata (optional) — population, impervious,
    # canopy. Same JSON shape as the LST overlay; built by build_data_overlays.py.
    overlay_meta = {}
    for _ov in ("population", "impervious", "canopy"):
        _p = ROOT / "data/context/overlays" / f"{_ov}_bounds.json"
        if _p.exists():
            overlay_meta[_ov] = _p.read_text().strip()
            logger.info(f"Loaded {_ov} overlay metadata")
        else:
            overlay_meta[_ov] = "null"
            logger.warning(f"No {_ov} overlay metadata — Data Layers entry will be unavailable")

    # Downscaled-analysis encoded raster + tract GEOID lookup -> base64-embedded so
    # the viewer's "Downscaled" layer recomputes composite/driver per ~100 m pixel
    # (downscaled LST exposure + each tract's LIVE sensitivity/adaptive), masked to
    # the populated footprint. R=exposure pct, G/B=tract index, A=mask. Built by
    # decks/sdrcc_slides/build_hero_downscaled.py; embedding avoids canvas tainting.
    import base64 as _b64
    ds_enc_uri, ds_geoids, ds_dens_uri = "null", "null", "null"
    _ds_png = ROOT / "data/context/overlays/downscaled_encoded.png"
    _ds_gj = ROOT / "data/context/overlays/downscaled_geoids.json"
    _ds_dens = ROOT / "data/context/overlays/downscaled_density.png"
    if _ds_png.exists() and _ds_gj.exists():
        ds_enc_uri = '"data:image/png;base64,' + _b64.b64encode(_ds_png.read_bytes()).decode() + '"'
        ds_geoids = _ds_gj.read_text().strip()
        logger.info("Embedded downscaled encoded raster + GEOID lookup")
    else:
        logger.warning("No downscaled_encoded.png — Downscaled layer unavailable")
    if _ds_dens.exists():
        ds_dens_uri = '"data:image/png;base64,' + _b64.b64encode(_ds_dens.read_bytes()).decode() + '"'
        logger.info("Embedded downscaled population-density raster")
    else:
        logger.warning("No downscaled_density.png — bivariate downscaled view unavailable")

    # Load city jurisdiction boundaries (optional)
    juris_str = "null"
    if JURIS_PATH.exists():
        juris = gpd.read_file(JURIS_PATH)
        juris = juris[juris["category"] == "City"][["NAME", "geometry"]].copy()
        juris = juris.to_crs(epsg=4326)
        juris["geometry"] = juris.geometry.simplify(0.0005)
        juris_str = _round_geojson_coords(juris.to_json())
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

    # Load unincorporated county boundary (optional) — SANGIS JUR_MUNICIPAL
    # "S.D. COUNTY" (code CN) features, dissolved to one region. The unincorporated
    # territory is ~85% of the county by area, so it's drawn as one outlined region.
    unincorp_str = "null"
    if UNINCORP_PATH.exists():
        unincorp_str = _round_geojson_coords(UNINCORP_PATH.read_text())
        logger.info(f"Loaded unincorporated boundary from {UNINCORP_PATH.name}")
    else:
        logger.warning("No unincorporated file — unincorporated overlay will not be shown")

    # Load cool zone sites (optional) — sites designated by the County Cool
    # Zones program as free air-conditioned refuge during extreme heat. Built by
    # scripts/build_coolzones.py from the County's feature service + site-list
    # PDF; the `source` column carries provenance so the popup can attribute
    # each marker. Inlined into the HTML so the static page is self-contained.
    coolzones_str = "null"
    if COOLZONES_PATH.exists():
        import csv as _csv
        import json as _json
        feats = []
        with COOLZONES_PATH.open() as f:
            for row in _csv.DictReader(f):
                try:
                    lat = float(row["lat"]); lon = float(row["lon"])
                except (ValueError, KeyError, TypeError):
                    continue
                feats.append({
                    "name": row.get("organization", "").strip(),
                    "address": row.get("address", "").strip(),
                    "city": row.get("city", "").strip(),
                    "zip": row.get("zip", "").strip(),
                    "phone": row.get("phone", "").strip(),
                    "lat": lat, "lon": lon,
                    "src": row.get("source", "").strip(),
                })
        coolzones_str = _json.dumps(feats, separators=(",", ":"))
        logger.info(f"Loaded {len(feats)} cool-zone sites")
    else:
        logger.warning("No cool-zone file — cool-zone overlay will not be shown")

    html = HTML_TEMPLATE
    html = html.replace("__GEOJSON_PLACEHOLDER__", _script_safe(geojson_str))
    html = html.replace("__VARMETA_PLACEHOLDER__", _script_safe(var_meta_str))
    feedback_mailto = (
        f"mailto:{FEEDBACK_EMAIL}?subject={urllib.parse.quote(FEEDBACK_SUBJECT)}"
    )
    html = html.replace("__FEEDBACK_MAILTO_PLACEHOLDER__", feedback_mailto)
    html = html.replace("__VARSOURCES_PLACEHOLDER__", _script_safe(var_sources_str))
    html = html.replace("__SOURCENAMES_PLACEHOLDER__", _script_safe(source_names_str))
    html = html.replace("__VARUNITS_PLACEHOLDER__", _script_safe(var_units_str))
    html = html.replace("__NOEXPORT_PLACEHOLDER__", _script_safe(no_export_str))
    html = html.replace("__NOEXPORTWHY_PLACEHOLDER__", _script_safe(no_export_reasons_str))
    html = html.replace("__EXPORTMETA_PLACEHOLDER__", _script_safe(export_meta_str))
    html = html.replace("__HEATMODES_PLACEHOLDER__", _script_safe(heat_modes_str))
    html = html.replace("__DELTAS_PLACEHOLDER__", _script_safe(deltas_str))
    html = html.replace("__DELTAMAP_PLACEHOLDER__", _script_safe(delta_map_str))
    html = html.replace("__JURIS_PLACEHOLDER__", _script_safe(juris_str))
    html = html.replace("__TRIBAL_PLACEHOLDER__", _script_safe(tribal_str))
    html = html.replace("__UNINCORP_PLACEHOLDER__", _script_safe(unincorp_str))
    html = html.replace("__COOLZONES_PLACEHOLDER__", _script_safe(coolzones_str))
    html = html.replace("__LST_PLACEHOLDER__", _script_safe(lst_str))
    html = html.replace("__POP_PLACEHOLDER__", _script_safe(overlay_meta["population"]))
    html = html.replace("__DS_ENC_PLACEHOLDER__", _script_safe(ds_enc_uri))
    html = html.replace("__DS_GEOIDS_PLACEHOLDER__", _script_safe(ds_geoids))
    html = html.replace("__DS_DENS_PLACEHOLDER__", _script_safe(ds_dens_uri))
    html = html.replace("__IMPERVIOUS_PLACEHOLDER__", _script_safe(overlay_meta["impervious"]))
    html = html.replace("__CANOPY_PLACEHOLDER__", _script_safe(overlay_meta["canopy"]))
    html = html.replace("__POLARITY_PLACEHOLDER__", _script_safe(polarity_str))

    OUTPUT_PATH.write_text(html)
    size_kb = len(html) // 1024
    logger.info(f"Saved to {OUTPUT_PATH} ({size_kb} KB)")

    rel = OUTPUT_PATH.relative_to(ROOT)
    print(f"\nGitHub Pages URL (after push):")
    print(f"  https://cccia-sio.github.io/socal-heat-hub/{rel}")


# ---------------------------------------------------------------------------
# HTML template — self-contained Leaflet + JS PCA viewer
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>San Diego Heat Risk Explorer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"
      integrity="sha384-o/2yZuJZWGJ4s/adjxVW71R+EO/LyCwdQfP5UWSgX/w87iiTXuvDZaejd3TsN7mf"
      crossorigin="anonymous"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"
        integrity="sha384-okbbMvvx/qfQkmiQKfd5VifbKZ/W8p1qIsWvE1ROPUfHWsDcC8/BnHohF7vPg2T6"
        crossorigin="anonymous"></script>
<script src="https://unpkg.com/protomaps-leaflet@4.1.1/dist/protomaps-leaflet.js"
        integrity="sha384-sWbmruuInggy2e4rWvqqqdbRbmdj81cna00wBtIthh+lEV6YQPjrrmDJfnjJsryv"
        crossorigin="anonymous"></script>
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
.subtitle{font-size:12px;color:#666;margin:4px 0 16px;line-height:1.4}

#scenario-bar{margin:18px 0 0;padding:14px 0 0;border-top:1px solid #e0e0e0;display:none}
#scenario-bar>summary{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;color:#666;cursor:pointer;user-select:none;list-style:none;
  display:flex;align-items:center;gap:6px}
#scenario-bar>summary::-webkit-details-marker{display:none}
#scenario-bar>summary::before{content:"\\25B6";font-size:8px;display:inline-block;
  transition:transform 0.15s;color:#999}
#scenario-bar[open]>summary::before{transform:rotate(90deg)}
.scenario-body{margin-top:10px}
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

/* Heat-exposure single-select mode rows (compact radio list) */
.mode-help{font-size:11px;color:#777;line-height:1.4;margin:1px 0 4px;padding-left:2px}
.mode-row{display:flex;align-items:baseline;gap:7px;padding:2px 0 2px 12px;
  cursor:pointer;user-select:none}
.mode-row input[type=radio]{width:13px;height:13px;accent-color:currentColor;
  cursor:pointer;flex-shrink:0;margin:0;align-self:center}
.mode-txt{display:flex;align-items:baseline;gap:6px;font-size:12px}
.mode-name{color:#444}
.mode-row.on .mode-name{font-weight:600}
.mode-meta{font-size:10px;color:#aaa;cursor:help;
  text-decoration:underline dotted #ccc;text-underline-offset:2px}

#compute-btn{width:100%;padding:11px;margin:16px 0 12px;
  background:#2563eb;color:#fff;border:none;border-radius:10px;
  font-size:14px;font-weight:600;cursor:pointer;transition:background 0.2s;
  font-family:inherit}
#compute-btn:hover{background:#1d4ed8}
#compute-btn:active{background:#1e40af}
/* Stale: the panel has changed since the last build. Amber reads as "action
   needed" rather than "error", and the ring catches the eye without motion. */
#compute-btn.needs-build{background:#d97706;box-shadow:0 0 0 3px rgba(217,119,6,0.28)}
#compute-btn.needs-build:hover{background:#b45309}
#compute-btn.needs-build:active{background:#92400e}

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
.popwt{display:flex;align-items:center;gap:7px;width:100%;margin:4px 0 2px;
  padding:5px 10px;font-size:11.5px;color:#555;user-select:none}
.popwt label{display:flex;align-items:center;gap:7px;flex:1;cursor:pointer;color:inherit}
.popwt input{cursor:pointer;margin:0;flex-shrink:0}
.popwt:hover{color:#222}
.biv-legend{background:rgba(255,255,255,0.92);border:1px solid rgba(0,0,0,0.12);
  border-radius:8px;padding:6px 8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
  font-size:9.5px;color:#555;line-height:1.15;align-self:flex-start}
.biv-title{font-weight:600;color:#333;margin-bottom:3px}
.biv-sw{display:block;width:74px;height:74px;border:1px solid #bbb}
.biv-axes{display:flex;justify-content:space-between;margin-top:2px;font-size:9px;color:#666}
/* Sidebar info bubbles (layer buttons, PCA-pattern buttons, the Populated-areas
   checkbox) sit at the right edge of #sidebar, which clips horizontal overflow
   (overflow-y:auto forces overflow-x:auto). To render the popup to the RIGHT of
   its icon -- out over the map -- promote it to fixed positioning; the coords are
   set from the icon's rect by positionSidebarPop() in JS on hover/focus, and the
   arrow moves to the popup's left edge. On mobile the sidebar is a transformed
   overlay drawer (see the max-width media query), which would trap a fixed
   descendant, so there it reverts to an in-drawer popup that grows leftward. */
#sidebar .info-pop{position:fixed;left:0;top:0;right:auto;transform:translateY(-50%)}
#sidebar .info-pop::after{top:50%;left:-10px;bottom:auto;right:auto;
  transform:translateY(-50%);border:5px solid transparent;border-right-color:#1f2937}
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
.tri-host{display:flex;justify-content:center;margin:5px 0 1px}
.tri-host canvas{display:block}
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
  background:rgba(0,0,0,0.35);font-size:15px;font-weight:600;
  color:#fff;text-shadow:0 1px 3px rgba(0,0,0,0.6);pointer-events:none}

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
#city-focus-bar{position:absolute;top:10px;left:50%;transform:translateX(-50%);
  z-index:1001;background:#fff;color:#1f2937;padding:7px 8px 7px 14px;
  border-radius:10px;font-size:13px;display:none;align-items:center;gap:10px;
  box-shadow:0 2px 10px rgba(0,0,0,0.14);border:1px solid #e5e7eb}
#city-focus-bar.show{display:flex}
#city-focus-bar .cf-name{font-weight:700;color:#111827}
#city-focus-bar .cf-hint{color:#6b7280;font-size:11px}
#city-focus-bar button{background:#fff;border:1px solid #d1d5db;border-radius:7px;
  padding:5px 10px;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;
  color:#374151;transition:all 0.15s}
#city-focus-bar button:hover{background:#f3f4f6;border-color:#9ca3af}
#city-focus-bar button.cf-primary{background:#2563eb;border-color:#2563eb;color:#fff}
#city-focus-bar button.cf-primary:hover{background:#1d4ed8}
#city-focus-bar button.cf-primary:disabled{opacity:0.6;cursor:wait}
#city-focus-bar button.cf-exit{padding:5px 9px;color:#6b7280;font-weight:700}

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
/* Per-map legend beneath each comparison map (replaces the old shared footer) */
.scn-cmp-leg{padding:7px 14px 8px;border-top:1px solid #eef0f2;background:#fafafa;
  flex-shrink:0;min-height:76px}
.scn-leg-name{font-size:11px;font-weight:600;color:#374151;margin-bottom:5px}
.scn-leg-name .scn-leg-sub{font-weight:400;color:#9ca3af}
.scn-leg-dirs,.scn-leg-ends,.scn-leg-ticks{display:flex;justify-content:space-between;
  font-size:10px;color:#9ca3af;font-variant-numeric:tabular-nums}
.scn-leg-dirs{margin-bottom:3px;color:#6b7280;font-weight:500}
.scn-leg-bar{height:11px;border-radius:3px;border:1px solid #ddd}
.scn-leg-ends,.scn-leg-ticks{margin-top:3px}
.scn-leg-unit{font-size:9px;color:#b0b4ba;text-align:center;margin-top:2px;
  text-transform:uppercase;letter-spacing:0.04em}
/* "How to read this" button in the compare-scenarios header */
.scn-info-btn{background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;
  font-size:11px;font-weight:600;font-family:inherit;cursor:pointer;
  padding:5px 10px;border-radius:7px;margin-right:8px;white-space:nowrap}
.scn-info-btn:hover{background:#dbeafe}
/* Compare-scenarios explainer infographic */
#scn-info-modal{position:fixed;inset:0;z-index:2200;background:rgba(20,25,40,0.45);
  display:none;align-items:center;justify-content:center;
  backdrop-filter:blur(2px);font-family:inherit}
#scn-info-modal.open{display:flex}
.scn-info-panel{background:#fff;width:min(440px,94vw);max-height:90vh;
  border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,0.32);
  display:flex;flex-direction:column;overflow:hidden}
.scn-info-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:13px 18px;border-bottom:1px solid #e5e7eb;flex-shrink:0}
.scn-info-hdr h3{font-size:14px;font-weight:700;margin:0;color:#111}
.scn-info-close{background:none;border:none;font-size:22px;color:#666;cursor:pointer;
  padding:0 4px;line-height:1;font-family:inherit}
.scn-info-close:hover{color:#111}
.scn-info-body{padding:16px 18px;overflow-y:auto;font-size:13px;color:#374151;
  line-height:1.55}
.scn-info-row{display:flex;gap:11px;margin-bottom:13px}
.scn-info-row b{color:#111}
.scn-info-chip{flex-shrink:0;width:34px;height:34px;border-radius:7px;
  border:1px solid rgba(0,0,0,0.12);margin-top:1px}
.scn-info-chip.chip-left{background:linear-gradient(135deg,rgb(255,255,178),rgb(189,0,38))}
.scn-info-chip.chip-right{background:linear-gradient(135deg,rgb(33,102,172),rgb(247,247,247) 50%,rgb(178,24,43))}
.ink-blue{color:#2166ac}
.ink-red{color:#b2182b}
.scn-info-key{margin:2px 0 12px}
.scn-info-keybar{height:12px;border-radius:3px;border:1px solid #ddd;
  background:linear-gradient(to right,rgb(33,102,172),rgb(146,197,222),rgb(247,247,247),rgb(244,165,130),rgb(178,24,43))}
.scn-info-keyends{display:flex;justify-content:space-between;font-size:10px;
  color:#6b7280;margin-top:4px}
.scn-info-note{font-size:11.5px;color:#6b7280;line-height:1.5;margin:0;
  background:#f8fafc;border-left:3px solid #cbd5e1;padding:8px 11px;border-radius:6px}
.scn-info-foot{padding:11px 18px;border-top:1px solid #e5e7eb;flex-shrink:0;
  display:flex;justify-content:flex-end;background:#fafafa}
.scn-info-ok{padding:7px 16px;border-radius:8px;font-size:12px;font-weight:600;
  cursor:pointer;font-family:inherit;border:1px solid #2563eb;background:#2563eb;color:#fff}
.scn-info-ok:hover{background:#1d4ed8}

/* Permanent city-name labels on jurisdiction layers */
.juris-label{background:transparent;border:none;box-shadow:none;padding:0;
  font-size:12px;font-weight:700;color:#1a1a1a;white-space:nowrap;
  text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;pointer-events:none}
.juris-label::before{display:none}
.tribal-label{color:#6b4423}

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
.wlc-body ul,.wlc-body ol{margin:0 0 12px;padding-left:18px}
.wlc-body ul:last-child,.wlc-body ol:last-child{margin-bottom:0}
.wlc-body li{margin-bottom:7px}
.wlc-body li b{color:#111}
.wlc-body .wlc-quickstart{background:#f5f7fb;border-left:3px solid #2563eb;
  border-radius:6px;padding:10px 12px 4px 14px;margin:0 0 14px}
.wlc-body .wlc-quickstart p{margin:0 0 6px;font-weight:600;color:#1d4ed8;
  font-size:12px;text-transform:uppercase;letter-spacing:0.04em}
.wlc-body .wlc-quickstart ol{margin:0;padding-left:20px}
.wlc-footer{padding:12px 18px;border-top:1px solid #e5e7eb;
  display:flex;gap:8px;justify-content:flex-end;flex-shrink:0;background:#fafafa}
.wlc-btn{padding:8px 16px;border-radius:8px;font-size:12px;font-weight:600;
  cursor:pointer;font-family:inherit;border:1px solid #2563eb;
  background:#2563eb;color:#fff;text-decoration:none;display:inline-block}
.wlc-btn:hover{background:#1d4ed8;border-color:#1d4ed8}
.wlc-btn.secondary{background:#fff;color:#2563eb}
.wlc-btn.secondary:hover{background:#eff6ff;color:#1d4ed8}

/* --- Guided tour (opt-in from welcome popup) ---
   Dim the page with four fixed rects around the target's bounding rect so the
   target itself stays interactive (the user advances by actually clicking it).
   A separate spotlight ring sits over the target with pointer-events:none so
   clicks pass through. */
#tour-overlay{display:none;font-family:inherit}
#tour-overlay.open{display:block}
.tour-dim{position:fixed;background:rgba(15,23,42,0.55);z-index:2400;
  transition:all 0.18s ease;backdrop-filter:blur(1px)}
.tour-spotlight{position:fixed;z-index:2401;pointer-events:none;
  border-radius:8px;box-shadow:0 0 0 3px #2563eb,0 0 16px 4px rgba(37,99,235,0.45);
  transition:all 0.18s ease}
.tour-card{position:fixed;z-index:2402;background:#fff;width:320px;max-width:92vw;
  border-radius:12px;box-shadow:0 12px 36px rgba(0,0,0,0.30);
  padding:14px 16px 12px;font-family:inherit;color:#1f2937;font-size:13px;
  line-height:1.5;transition:opacity 0.12s ease}
.tour-card.centered{left:50%;top:50%;transform:translate(-50%,-50%);width:380px}
.tour-card .tc-progress{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.06em;color:#6b7280;margin-bottom:4px}
.tour-card .tc-title{font-size:15px;font-weight:700;color:#111;margin:0 0 6px;
  line-height:1.3}
.tour-card .tc-body{font-size:13px;color:#374151;margin:0 0 12px;line-height:1.55}
.tour-card .tc-body b{color:#111}
.tour-card .tc-prompt{display:flex;align-items:center;gap:6px;
  background:#eff6ff;border-left:3px solid #2563eb;padding:7px 9px;
  border-radius:6px;font-size:12px;color:#1d4ed8;font-weight:500;margin:0 0 12px}
.tour-card .tc-prompt::before{content:"\\1F446";font-size:14px}
.tour-card .tc-footer{display:flex;align-items:center;gap:6px}
.tour-card .tc-footer .spacer{flex:1}
.tour-card .tc-btn{padding:6px 12px;border-radius:7px;font-size:12px;
  font-weight:600;cursor:pointer;font-family:inherit;border:1px solid #d1d5db;
  background:#fff;color:#374151;transition:all 0.12s}
.tour-card .tc-btn:hover{background:#f3f4f6}
.tour-card .tc-btn.primary{border-color:#2563eb;background:#2563eb;color:#fff}
.tour-card .tc-btn.primary:hover{background:#1d4ed8;border-color:#1d4ed8}
.tour-card .tc-btn.primary:disabled{background:#9ca3af;border-color:#9ca3af;
  cursor:not-allowed}
.tour-card .tc-skip{background:none;border:none;color:#6b7280;font-size:11px;
  cursor:pointer;font-family:inherit;padding:4px 6px;text-decoration:underline}
.tour-card .tc-skip:hover{color:#374151}
.tour-card .tc-arrow{position:absolute;width:0;height:0;
  border:8px solid transparent}
.tour-card.arrow-left .tc-arrow{left:-16px;top:24px;
  border-right-color:#fff}
.tour-card.arrow-right .tc-arrow{right:-16px;top:24px;
  border-left-color:#fff}
.tour-card.arrow-top .tc-arrow{top:-16px;left:24px;
  border-bottom-color:#fff}
.tour-card.arrow-bottom .tc-arrow{bottom:-16px;left:24px;
  border-top-color:#fff}

/* --- Tract detail modal (opened from address-search popup) --- */
#detail-modal{display:none;position:fixed;inset:0;z-index:2000;
  background:rgba(15,23,42,0.55);justify-content:center;align-items:center;
  font-family:inherit;padding:20px}
#detail-modal.open{display:flex}
.dt-panel{background:#fff;border-radius:12px;width:640px;max-width:96vw;
  max-height:92vh;display:flex;flex-direction:column;
  box-shadow:0 20px 50px rgba(0,0,0,0.35);overflow:hidden}
.dt-hdr{display:flex;align-items:center;gap:10px;padding:14px 18px 12px;
  border-bottom:1px solid #e5e7eb;flex-shrink:0;background:#f9fafb}
.dt-hdr h2{font-size:16px;font-weight:700;margin:0;color:#111;line-height:1.2}
.dt-hdr .dt-sub{font-size:11px;color:#6b7280;margin-top:2px}
.dt-hdr-text{flex:1;min-width:0}
.dt-close{width:30px;height:30px;border:none;border-radius:6px;
  background:transparent;font-size:22px;color:#6b7280;cursor:pointer;
  font-family:inherit;line-height:1;padding:0;flex-shrink:0}
.dt-close:hover{background:#e5e7eb;color:#111}
.dt-ctx{font-size:10.5px;color:#6b7280;font-style:italic;
  padding:6px 18px;background:#fafafa;border-bottom:1px solid #f1f5f9;flex-shrink:0}
.dt-body{flex:1;overflow-y:auto;padding:0;scrollbar-width:thin}
.dt-body::-webkit-scrollbar{width:6px}
.dt-body::-webkit-scrollbar-thumb{background:#ccc;border-radius:3px}
.dt-section{padding:14px 18px;border-bottom:1px solid #f1f5f9}
.dt-section:last-child{border-bottom:none}
.dt-section-hdr{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;color:#666;margin-bottom:10px}
/* Summary cards (composite + pillar scores) */
.dt-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.dt-card{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
  padding:9px 10px;display:flex;flex-direction:column;gap:2px}
.dt-card-lbl{font-size:9.5px;font-weight:600;color:#6b7280;text-transform:uppercase;
  letter-spacing:0.4px;line-height:1.1}
.dt-card-val{font-size:18px;font-weight:700;color:#111;
  font-variant-numeric:tabular-nums;line-height:1.1}
.dt-card-pct{font-size:9.5px;color:#9ca3af;line-height:1.1}
.dt-driver{margin-top:10px;font-size:11.5px;color:#374151;line-height:1.45}
.dt-driver b{color:#111}
.dt-note{margin-top:8px;font-size:10px;color:#9ca3af;line-height:1.4}
/* Layer toggle buttons above the single tract map */
.dt-view-switch{display:inline-flex;background:#f3f4f6;border:1px solid #e5e7eb;
  border-radius:8px;padding:3px;gap:2px;margin-bottom:6px}
.dt-view-btn{background:transparent;border:none;font-family:inherit;font-size:11.5px;
  font-weight:600;padding:5px 11px;border-radius:6px;cursor:pointer;color:#4b5563;
  letter-spacing:0.2px}
.dt-view-btn:hover{background:#fff;color:#111}
.dt-view-btn.active{background:#fff;color:#111;box-shadow:0 1px 2px rgba(0,0,0,0.08)}
.dt-view-sub{font-size:10.5px;color:#6b7280;margin-bottom:6px}
.dt-map-frame{border-radius:8px;overflow:hidden;border:1px solid #e5e7eb}
.dt-map{height:280px;background:#eef}
.dt-lst-legend{display:flex;flex-direction:column;gap:2px;font-size:10px;
  color:#666;margin-top:6px}
.dt-lst-bar{height:8px;border-radius:2px}
.dt-lst-ticks{display:flex;justify-content:space-between;font-size:9px;color:#888;margin-top:1px}
.dt-lu-legend{display:flex;margin-top:6px;flex-wrap:wrap;gap:6px;font-size:10px}
.dt-lu-item{display:flex;align-items:center;gap:4px;color:#555}
.dt-lu-sw{width:11px;height:11px;border-radius:2px;border:1px solid rgba(0,0,0,0.18)}
.dt-pillar-hdr{font-size:11.5px;font-weight:700;margin-bottom:8px;
  padding-bottom:5px;border-bottom:2px solid currentColor;
  text-transform:uppercase;letter-spacing:0.3px}
.dt-var-row{padding:6px 0;border-bottom:1px solid #f5f5f5;font-size:11.5px}
.dt-var-row:last-child{border-bottom:none}
.dt-var-top{display:flex;justify-content:space-between;align-items:baseline;
  gap:8px;margin-bottom:4px}
.dt-var-label{color:#374151;flex:1;min-width:0}
.dt-var-val{font-weight:600;color:#111;font-variant-numeric:tabular-nums;
  white-space:nowrap}
.dt-var-dir{font-weight:400;font-size:10px;color:#9ca3af;margin-left:4px}
.dt-rows-note{margin:6px 0 10px}
.dt-risk-legend{font-size:10px;color:#666}
.dt-var-bar-wrap{position:relative;height:6px;background:#e5e7eb;
  border-radius:3px;overflow:hidden}
.dt-var-bar{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.dt-var-pct{font-size:9.5px;color:#9ca3af;margin-top:2px;text-align:right;
  font-variant-numeric:tabular-nums}
.dt-na{color:#bbb;font-style:italic;font-weight:400}
.dt-hint{background:#fffbeb;color:#92400e;padding:10px 12px;border-radius:6px;
  font-size:11.5px;line-height:1.45}
.popup-more-btn{display:inline-block;margin-top:6px;margin-left:6px;
  background:#1d4ed8;color:#fff;font-size:11px;font-weight:600;
  padding:5px 10px;border:none;border-radius:4px;cursor:pointer;
  font-family:inherit}
.popup-more-btn:hover{background:#1e40af}
/* Secondary action: the one-page tract handout. Outlined rather than filled so
   it reads as subordinate to the two blue primary actions above it. */
.popup-pdf-btn{display:inline-block;background:#fff;color:#334155;font-size:11px;
  font-weight:600;padding:5px 10px;border:1px solid #cbd5e1;border-radius:4px;
  cursor:pointer;font-family:inherit}
.popup-pdf-btn:hover{background:#f1f5f9;border-color:#94a3b8}
.popup-pdf-btn:disabled{opacity:0.55;cursor:default}
.popup-btn-row{display:flex;flex-direction:column;gap:6px;align-items:stretch;margin-top:8px}
.popup-btn-row .popup-more-btn,.popup-btn-row .popup-compare-btn,
.popup-btn-row .popup-pdf-btn{margin:0;width:100%;
  box-sizing:border-box;text-align:center;border-radius:6px}

/* --- bottom-left control stack -------------------------------------------
   One anchored flex column holding every on-map control. Cards are ordered
   narrowest-first (see the HTML), so width grows top-to-bottom: left edges
   align and right edges cascade outward into a waterfall, the widest card
   resting in the bottom-left corner. The Leaflet scale bar is moved to the
   bottom-right (see L.control.scale) so this corner stays clean. On mobile the
   whole stack reparents into the drawer (see the reparenting JS at body end). */
#map-controls-bl{position:absolute;left:10px;bottom:12px;z-index:500;
  display:flex;flex-direction:column;gap:10px;align-items:flex-start}
/* Bottom-right stack: basemap picker + colorblind toggle, anchored above the
   SIO logo and right-aligned so both cards hug the corner. */
#map-controls-br{position:absolute;right:14px;bottom:86px;z-index:999;
  display:flex;flex-direction:column;gap:10px;align-items:flex-end}
/* Colorblind-friendly toggle, a small floating card in the stack. */
#cbmode-control{background:rgba(255,255,255,0.92);border:1px solid rgba(0,0,0,0.12);
  border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);padding:7px 10px}
#cbmode-control label{display:flex;align-items:center;gap:7px;font-family:inherit;
  font-size:11px;font-weight:600;color:#374151;cursor:pointer;white-space:nowrap}
#cbmode-control input[type=checkbox]{width:14px;height:14px;accent-color:#2563eb;
  cursor:pointer;flex-shrink:0;margin:0}

/* --- on-map basemap picker (top of the bottom-right stack) --- */
#basemap-picker{display:flex;gap:8px;background:rgba(255,255,255,0.92);
  padding:6px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
  border:1px solid rgba(0,0,0,0.12)}
.bm-thumb{position:relative;width:54px;height:54px;border-radius:6px;
  overflow:hidden;cursor:pointer;border:2px solid #fff;outline:none;
  padding:0;background-color:#e5e7eb;background-size:cover;
  background-position:center;transition:transform 0.15s,box-shadow 0.15s;
  box-shadow:0 1px 3px rgba(0,0,0,0.15)}
.bm-thumb:hover{transform:translateY(-1px);box-shadow:0 2px 6px rgba(0,0,0,0.25)}
.bm-thumb.active{border-color:#2563eb;box-shadow:0 0 0 2px #2563eb,0 2px 6px rgba(0,0,0,0.25)}
.bm-thumb-lbl{position:absolute;bottom:0;left:0;right:0;
  font-size:10px;font-weight:600;color:#fff;text-align:center;
  padding:2px 0;background:linear-gradient(transparent,rgba(0,0,0,0.65));
  text-shadow:0 1px 2px rgba(0,0,0,0.5);font-family:inherit;
  letter-spacing:0.2px}

/* --- land-use overlay toggle (main map) --- */
#lu-toggle-btn{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,0.92);
  padding:6px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
  border:1px solid rgba(0,0,0,0.12);font-family:inherit;font-size:11px;font-weight:600;
  color:#374151;cursor:pointer;white-space:nowrap}
#lu-toggle-btn:hover{background:rgba(255,255,255,1)}
#lu-toggle-btn.active{background:rgba(139,69,19,0.1);border-color:#8b4513;color:#5a2d0c}
#lu-main-legend{background:rgba(255,255,255,0.92);border:1px solid rgba(0,0,0,0.12);
  border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);padding:8px 10px;
  display:none;font-size:10px;line-height:1.5}
#lu-main-legend.open{display:block}
.lu-main-item{display:flex;align-items:center;gap:5px;white-space:nowrap}
.lu-main-swatch{width:10px;height:10px;border-radius:2px;flex-shrink:0;border:1px solid rgba(0,0,0,0.2)}

/* --- Data Layers dropdown (informational reference overlays, main map) --- */
#data-layers-control{display:flex;flex-direction:column;gap:6px;background:rgba(255,255,255,0.92);
  padding:8px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
  border:1px solid rgba(0,0,0,0.12);width:200px;box-sizing:border-box}
#data-layers-control .dl-ctl-lbl{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;color:#374151}
#data-layer-select{width:100%;padding:6px 8px;border:1px solid #d1d5db;border-radius:6px;
  font-family:inherit;font-size:12px;color:#374151;background:#fff;cursor:pointer}
#data-layer-opacity-wrap{display:flex;align-items:center;gap:6px;color:#374151}
#data-layer-opacity-wrap .opacity-lbl{font-weight:600;white-space:nowrap;font-size:10px}
#data-layer-opacity-slider{flex:1;cursor:pointer;accent-color:#2563eb}
#data-layer-opacity-val{font-variant-numeric:tabular-nums;color:#6b7280;min-width:32px;
  text-align:right;font-size:10px}
#data-layer-legend{font-size:10px;line-height:1.5;color:#374151}

/* --- composite-risk opacity slider (sits above the basemap picker) --- */
#map-opacity-control{display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.92);
  padding:6px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
  border:1px solid rgba(0,0,0,0.12);font-family:inherit;font-size:11px;
  color:#374151}
#map-opacity-control .opacity-lbl{font-weight:600;white-space:nowrap}
#map-opacity-slider{width:110px;cursor:pointer;accent-color:#2563eb}
#map-opacity-val{font-variant-numeric:tabular-nums;color:#6b7280;
  min-width:34px;text-align:right}

/* --- address search (floating card, bottom-left of the map) --- */
#addr-search{background:rgba(255,255,255,0.95);border:1px solid rgba(0,0,0,0.12);
  border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);padding:8px 10px;
  width:236px;box-sizing:border-box}
#addr-search h3{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:0.5px;margin:0 0 6px;color:#374151}
.addr-help{font-size:11px;color:#777;line-height:1.4;margin-bottom:6px}
.addr-row{display:flex;gap:6px;position:relative}
.addr-input-wrap{position:relative;flex:1;min-width:0}
#addr-input{width:100%;padding:8px 10px;border:1px solid #d1d5db;
  border-radius:8px;font-size:12px;font-family:inherit;color:#374151;
  background:#fff;outline:none;transition:border-color 0.15s}
#addr-input:focus{border-color:#2563eb}
.addr-suggest{position:absolute;bottom:calc(100% + 4px);top:auto;left:0;right:0;
  background:#fff;border:1px solid #d1d5db;border-radius:8px;
  box-shadow:0 4px 12px rgba(0,0,0,0.08);max-height:220px;overflow-y:auto;
  z-index:1000;list-style:none;margin:0;padding:4px 0;display:none}
.addr-suggest.open{display:block}
.addr-suggest li{padding:6px 10px;font-size:11.5px;color:#374151;
  cursor:pointer;line-height:1.35;border-bottom:1px solid #f1f5f9}
.addr-suggest li:last-child{border-bottom:none}
.addr-suggest li:hover,.addr-suggest li.active{background:#eff6ff;color:#1d4ed8}
.addr-suggest li.empty{color:#999;font-style:italic;cursor:default;background:#fff}
.addr-suggest li.empty:hover{background:#fff;color:#999}
.addr-suggest .addr-sg-main{font-weight:500;color:#111;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.addr-suggest li.active .addr-sg-main,
.addr-suggest li:hover .addr-sg-main{color:#1d4ed8}
.addr-suggest .addr-sg-sub{font-size:10.5px;color:#888;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:1px}
#addr-go{padding:8px 12px;border:1px solid #2563eb;border-radius:8px;
  background:#2563eb;color:#fff;font-size:12px;font-weight:600;
  cursor:pointer;font-family:inherit;flex-shrink:0}
#addr-go:hover{background:#1d4ed8;border-color:#1d4ed8}
#addr-go:disabled{background:#9ca3af;border-color:#9ca3af;cursor:wait}
#addr-status{font-size:11px;color:#555;margin-top:6px;line-height:1.4;min-height:1em}
#addr-status.err{color:#b91c1c}
#addr-status .addr-found{color:#111;font-weight:500;display:block}
#addr-status .addr-tract{color:#666;display:block;margin-top:2px}
#addr-clear{display:none;margin-top:6px;width:100%;padding:6px;
  border:1px solid #d1d5db;border-radius:6px;background:#fff;
  font-size:11px;color:#6b7280;cursor:pointer;font-family:inherit}
#addr-clear:hover{background:#f3f4f6;color:#374151}
.addr-marker-icon{background:transparent;border:none}

/* ============================================================
   MOBILE / RESPONSIVE
   One responsive page — there is no separate mobile build. The
   mobile layout (slide-in drawer + reflowed floating controls)
   triggers when the viewport is <=768px wide OR is a short
   landscape viewport (<=500px tall). The height clause catches
   large phones held sideways (e.g. iPhone Pro Max landscape is
   ~932x430) without dragging tablets/desktop windows in — those
   are >=744px tall in landscape. Everything else is inherited, so
   future viewer changes appear on phones automatically.
   ============================================================ */
/* Hamburger that opens the controls drawer on small screens.
   display:none on desktop where the sidebar is always visible. */
#menu-toggle{display:none;position:fixed;top:10px;left:10px;z-index:1700;
  width:42px;height:42px;align-items:center;justify-content:center;
  flex-direction:column;gap:4px;background:rgba(255,255,255,0.95);
  border:1px solid rgba(0,0,0,0.12);border-radius:10px;cursor:pointer;
  box-shadow:0 2px 8px rgba(0,0,0,0.18);padding:0}
#menu-toggle span{display:block;width:20px;height:2px;border-radius:2px;
  background:#374151;transition:transform 0.2s,opacity 0.2s}
#sidebar-scrim{display:none;position:fixed;inset:0;z-index:1500;
  background:rgba(0,0,0,0.35)}
/* Tap-to-reveal for the info tooltips — touch devices have no hover.
   tapInfo() (JS, end of body) toggles .open. Harmless on desktop. */
.info-ic.open .info-pop{display:block}
/* The bottom-left map control stack relocates into the drawer on mobile
   (see reparenting JS at end of body); the slot is an empty, hidden
   placeholder on desktop. */
#drawer-tools-slot{display:none}

@media (max-width:768px),(max-height:500px) and (orientation:landscape){
  body{overflow:hidden}
  #menu-toggle{display:flex}
  /* Sidebar becomes a left slide-in drawer, lifted out of normal flow
     so the map fills the full width whether or not the drawer is open. */
  #sidebar{position:fixed;top:0;left:0;height:100vh;height:100dvh;
    width:min(86vw,340px);transform:translateX(-100%);
    transition:transform 0.25s ease;z-index:1600;
    box-shadow:2px 0 16px rgba(0,0,0,0.18);padding-top:14px}
  body.sidebar-open #sidebar{transform:none}
  body.sidebar-open #sidebar-scrim{display:block}
  /* Toggle morphs into a close (X) at the drawer's top-right when open. */
  body.sidebar-open #menu-toggle{left:calc(min(86vw,340px) - 52px)}
  body.sidebar-open #menu-toggle span:nth-child(1){transform:translateY(6px) rotate(45deg)}
  body.sidebar-open #menu-toggle span:nth-child(2){opacity:0}
  body.sidebar-open #menu-toggle span:nth-child(3){transform:translateY(-6px) rotate(-45deg)}
  #map-wrap{width:100%}
  /* Sidebar is an overlay drawer here; keep info bubbles in-drawer (revert the
     desktop fixed-right positioning, which the transformed drawer would trap). */
  #sidebar .info-pop{position:absolute;left:auto;right:0;top:calc(100% + 8px);transform:none}
  #sidebar .info-pop::after{top:auto;bottom:100%;left:auto;right:5px;transform:none;
    border:5px solid transparent;border-bottom-color:#1f2937}
  /* Branding: keep the two top-right logos but shrink them; drop the
     bottom-right SIO mark so it doesn't fight the scale bar and the
     bottom-left control stack on a narrow screen. */
  #logos{top:10px;right:10px;padding:5px 9px;gap:9px}
  #logos img{height:20px}
  #logo-sio{display:none}
  /* Legend drops below the top row so it clears the hamburger + logos. */
  #legend{top:60px;left:10px;right:auto}
  #legend .grad{width:min(150px,52vw)}
  /* Bottom-left control stack: narrower address card, slimmer slider,
     smaller basemap thumbs so it doesn't dominate the map. */
  #addr-search{width:min(74vw,236px)}
  #map-opacity-slider{width:90px}
  .bm-thumb{width:46px;height:46px}
  /* Top-center status bars wrap and stay on-screen. */
  #city-focus-bar,#compare-banner{max-width:calc(100vw - 20px);flex-wrap:wrap;
    justify-content:center;row-gap:6px}
  /* Stack the side-by-side comparison modals into one scrolling column. */
  .cmp-body,.scn-cmp-body{grid-template-columns:1fr;overflow-y:auto}
  /* Tooltip bubbles never wider than the screen. */
  .info-pop{width:min(220px,76vw)}
  /* Push Leaflet's top-left zoom control below the hamburger. */
  .leaflet-top.leaflet-left{margin-top:52px}
  /* Keep the centered "select variables" prompt off the screen edges. */
  #prompt{font-size:13px;line-height:1.4;padding:0 46px;text-align:center}
  /* Relocated bottom-left controls: show the slot and flatten the floating
     cards into plain sidebar sections (no translucent bg / shadow / absolute
     positioning) so they read as part of the drawer. */
  #drawer-tools-slot{display:block;border-top:1px solid #e0e0e0;
    padding-top:10px;margin-top:10px}
  #drawer-tools-slot>h3{font-size:12px;font-weight:700;text-transform:uppercase;
    letter-spacing:0.5px;margin-bottom:8px;color:#666}
  #drawer-tools-slot #map-controls-bl,
  #drawer-tools-slot #map-controls-br{position:static;width:auto;z-index:auto;
    gap:12px;align-items:flex-start}
  #drawer-tools-slot #data-layers-control,
  #drawer-tools-slot #cbmode-control,
  #drawer-tools-slot #addr-search,
  #drawer-tools-slot #map-opacity-control{background:none;border:none;
    box-shadow:none;padding:0;width:auto}
  #drawer-tools-slot #basemap-picker{background:none;border:none;
    box-shadow:none;padding:0}
  /* In the drawer, address suggestions open downward instead of upward. */
  #drawer-tools-slot .addr-suggest{top:calc(100% + 4px);bottom:auto}
}
</style>
</head>
<body>

<button id="menu-toggle" aria-label="Toggle the controls panel" aria-expanded="false" onclick="toggleSidebar()"><span></span><span></span><span></span></button>
<div id="sidebar-scrim" onclick="toggleSidebar(false)"></div>

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
  <!-- On mobile, the bottom-left map controls (land use, find address,
       opacity, basemap) are reparented into this slot; empty on desktop. -->
  <div id="drawer-tools-slot"><h3>Map Tools</h3></div>
  <div id="overlay-toggles" style="display:none">
    <h3>Jurisdictions</h3>
    <details class="subsec" id="juris-subsec" style="margin-top:2px"><summary>Cities</summary>
      <div class="ov-row" style="margin-left:6px"><label><input type="checkbox" id="cb-juris-all" onchange="toggleJurisAll(this.checked)">Show all cities</label></div>
      <div id="juris-items" style="margin-top:2px"></div>
      <div class="ov-row" style="margin-left:6px;margin-top:6px;border-top:1px solid #eee;padding-top:6px"><label><input type="checkbox" id="cb-unincorp" onchange="toggleUnincorp(this.checked)"><span style="color:#4f46e5;font-weight:600">Show unincorporated county</span></label></div>
      <div style="font-size:10px;color:#666;margin:2px 0 0 6px">Land the County governs directly &mdash; everything outside the 18 incorporated cities (~85% of the county&#39;s area).</div>
    </details>
    <details class="subsec" id="tribal-subsec" style="margin-top:6px"><summary>Tribal Lands</summary>
      <div class="ov-row" style="margin-left:6px"><label><input type="checkbox" id="cb-tribal-all" onchange="toggleTribalAll(this.checked)"><span style="color:#8b5a2b;font-weight:600">Show all tribal lands</span></label></div>
      <div id="tribal-items" style="margin-top:2px"></div>
    </details>
    <details class="subsec" id="coolzones-subsec" style="margin-top:6px"><summary>Cool Zones</summary>
      <div class="ov-row" style="margin-left:6px"><label><input type="checkbox" id="cb-coolzones" onchange="toggleCoolZones(this.checked)">Show cool-zone sites</label></div>
      <div style="font-size:10px;color:#666;margin:2px 0 0 6px">Sites designated by the County Cool Zones program &mdash; free, air-conditioned refuge during extreme heat (libraries, community &amp; senior centers). Click a marker for address &amp; phone.</div>
    </details>
  </div>
  <details id="scenario-bar">
    <summary>Climate Scenario</summary>
    <div class="scenario-body">
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
    <button id="scn-cmp-btn" onclick="openScnCompareModal()" title="Open a side-by-side composite-risk map (left) plus Heat Exposure pillar-rank change in percentile points (right) under two climate scenarios">Compare Scenarios Side-by-Side</button>
    </div>
  </details>
  <div id="export-section" style="border-top:1px solid #e0e0e0;padding-top:10px;margin-top:14px">
    <h3 style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;color:#666">Export</h3>
    <button id="dl-pdf-btn" onclick="downloadMapPdf()" style="width:100%;padding:9px;border:1px solid #d1d5db;border-radius:8px;background:#fff;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;color:#374151;transition:all 0.15s">Download Map as PDF</button>
    <button id="dl-data-btn" onclick="openExportModal()" style="width:100%;padding:9px;border:1px solid #d1d5db;border-radius:8px;background:#fff;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;color:#374151;transition:all 0.15s">Download Data</button>
  </div>
  <!-- Plain mailto, not a form: no third-party service, so the no-tracking
       promise in the README and user guide stays literally true. -->
  <div id="feedback-section" style="border-top:1px solid #e0e0e0;padding-top:10px;margin-top:14px">
    <h3 style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;color:#666">Feedback</h3>
    <a id="feedback-btn" href="__FEEDBACK_MAILTO_PLACEHOLDER__" style="display:block;box-sizing:border-box;width:100%;padding:9px;border:1px solid #d1d5db;border-radius:8px;background:#fff;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;color:#374151;text-align:center;text-decoration:none;transition:all 0.15s">Send Feedback or Request</a>
    <div style="font-size:10px;color:#666;margin-top:5px;line-height:1.4">Questions, corrections, a variable you wish were here, or a tract that looks wrong &mdash; all welcome. Opens your email client.</div>
  </div>
</div>

<div id="map-wrap">
  <div id="map"></div>
  <!-- Bottom-left control stack, ordered narrowest-first so panel width grows
       top-to-bottom and the right edges cascade into a waterfall (see CSS). -->
  <div id="map-controls-bl">
    <div id="data-layers-control">
      <label class="dl-ctl-lbl" for="data-layer-select">Exploratory Data Layers</label>
      <select id="data-layer-select" onchange="setDataLayer(this.value)" title="Show an informational reference layer over the map (not part of the risk analysis)">
        <option value="">None</option>
        <option value="population">Population density</option>
        <option value="impervious">Impervious surface</option>
        <option value="canopy">Tree canopy</option>
        <option value="landuse">Land use (SANDAG)</option>
        <option value="lst">Land surface temp (LST)</option>
        <option value="hosp_day">Heat hospitalizations</option>
        <option value="excess_er">Excess heat ER visits</option>
        <option value="threshold">Heat-risk threshold (°F)</option>
      </select>
      <div id="data-layer-opacity-wrap" style="display:none">
        <span class="opacity-lbl">Opacity</span>
        <input type="range" id="data-layer-opacity-slider" min="0" max="100" value="80" oninput="setDataLayerOpacity(this.value)">
        <span id="data-layer-opacity-val">80%</span>
      </div>
      <div id="chs-level-wrap" style="display:none;align-items:center;gap:6px">
        <span class="opacity-lbl">CalHeatScore</span>
        <select id="chs-level-select" onchange="setThresholdLevel(this.value)" style="flex:1;font-size:11px;padding:3px;border:1px solid #d1d5db;border-radius:5px">
          <option value="1">1 — Mild (onset)</option>
          <option value="2">2 — Moderate</option>
          <option value="3">3 — High</option>
          <option value="4">4 — Severe</option>
        </select>
      </div>
      <div id="data-layer-legend" style="display:none"></div>
    </div>
    <div id="lu-main-legend"></div>
    <div id="addr-search">
      <h3>Find address</h3>
      <div class="addr-row">
        <div class="addr-input-wrap">
          <input id="addr-input" type="text" aria-label="Search a county address" placeholder="Search a county address" autocomplete="off" spellcheck="false" />
          <ul id="addr-suggest" class="addr-suggest" role="listbox"></ul>
        </div>
        <button id="addr-go" onclick="searchAddress()">Go</button>
      </div>
      <div id="addr-status"></div>
      <button id="addr-clear" onclick="clearAddressMarker()">Clear marker</button>
    </div>
  </div>
  <div id="logos">
    <img src="../assets/usd-logo-horizontal.png" alt="University of San Diego">
    <img src="../assets/ccsd_logo.png" alt="San Diego Regional Climate Collaborative">
  </div>
  <!-- Bottom-right stack: basemap picker with the colorblind toggle beneath it,
       anchored above the SIO logo. Reparented into the mobile drawer. -->
  <div id="map-controls-br">
    <div id="basemap-picker">
      <button class="bm-thumb" data-bm="light" onclick="setBasemap('light')" title="Light basemap"
              style="background-image:url('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/11/825/358'),url('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/11/825/358')">
        <span class="bm-thumb-lbl">Light</span>
      </button>
      <button class="bm-thumb active" data-bm="satellite" onclick="setBasemap('satellite')" title="Satellite basemap"
              style="background-image:url('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/11/825/358')">
        <span class="bm-thumb-lbl">Satellite</span>
      </button>
    </div>
    <div id="cbmode-control">
      <label title="Swap red/green palettes for Okabe-Ito + viridis (deuteranopia/protanopia/tritanopia safe)"><input type="checkbox" id="cb-cbmode" onchange="toggleCbMode(this.checked)">Colorblind-friendly colors</label>
    </div>
    <div id="map-opacity-control" title="Fade the composite-risk choropleth to see the basemap underneath">
      <span class="opacity-lbl">Map opacity</span>
      <input type="range" id="map-opacity-slider" min="0" max="100" value="100"
             oninput="setTractOpacity(this.value)">
      <span id="map-opacity-val">100%</span>
    </div>
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
  <div id="city-focus-bar">
    <span class="cf-name" id="cf-name">—</span>
    <span class="cf-hint">click tracts to explore</span>
    <button class="cf-primary" id="cf-pdf-btn" onclick="downloadCityPdf()">Export risk PDF</button>
    <button aria-label="Exit city view" class="cf-exit" onclick="exitCityView()" title="Exit city view (ESC)">&times;</button>
  </div>
</div>

<div id="pdf-overlay">
  <div class="box"><div class="spinner"></div><span id="pdf-overlay-msg">Generating PDF\\u2026</span></div>
</div>

<div id="export-modal" role="dialog" aria-modal="true" aria-labelledby="export-modal-title" onclick="if(event.target===this)closeExportModal()">
  <div class="exp-panel">
    <div class="exp-hdr">
      <h2 id="export-modal-title">Download Data</h2>
      <button aria-label="Close" class="exp-close" onclick="closeExportModal()" title="Close (ESC)">&times;</button>
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
        <label class="exp-radio" style="cursor:pointer;margin-top:6px"><input type="checkbox" id="exp-incl-dict" checked>Include the data dictionary<span class="exp-radio-desc">second file: label, units, direction &amp; source for every column</span></label>
        <div class="exp-note"><b>Read the dictionary first.</b> Column headers in the data file are raw variable keys (<code>EP_NOINT</code>, <code>shututility_prev</code>); <code>&lt;stem&gt;_data_dictionary.csv</code> gives each one a plain-English label, its units, whether a higher value means more or less risk, and a link to the original source. It also records the heat mode, scenario and switched-on variables the index columns were built from. GeoJSON downloads carry the same information at <code>metadata</code>. <b>Watch the direction:</b> <code>adaptive_score</code> is stored capacity-side (100&nbsp;=&nbsp;most capacity, least risk), the opposite of <code>exposure_score</code> and <code>sensitivity_score</code>.</div>
        <div class="exp-note">Raw source values are included, except where a source's terms bar redistribution: <a href="https://www.healthyplacesindex.org/" target="_blank" rel="noopener">Healthy&nbsp;Places&nbsp;Index</a> variables (tree canopy, park access, income, education, life expectancy, pollution, &amp;c.) are non-commercial, and the CalHeatScore heat-hospitalization variables have no published license. Both stay on the map and are listed in the dictionary with <code>in_download&nbsp;=&nbsp;no</code> and a note on where to request them. Percentile ranks (0&ndash;100) are always computed against all San Diego County tracts, even when you export a single jurisdiction. Every row carries <code>citation</code>, <code>license</code> (CC&nbsp;BY&nbsp;4.0) and <code>sources_url</code> columns; the full source table is in the <a href="../userguide/index.html#data" target="_blank" rel="noopener">User&nbsp;Guide</a>.</div>
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

<div id="compare-modal" role="dialog" aria-modal="true" aria-labelledby="compare-modal-title" onclick="if(event.target===this)closeCompareModal()">
  <div class="cmp-panel">
    <div class="cmp-hdr">
      <h2 id="compare-modal-title">Census Tract Comparison<span class="cmp-sub" id="cmp-context"></span></h2>
      <button aria-label="Close" class="cmp-close" onclick="closeCompareModal()" title="Close (ESC)">&times;</button>
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

<div id="scn-compare-modal" role="dialog" aria-modal="true" aria-labelledby="scn-compare-title" onclick="if(event.target===this)closeScnCompareModal()">
  <div class="cmp-panel scn-cmp-panel">
    <div class="cmp-hdr">
      <h2 id="scn-compare-title">Compare Scenarios<span class="cmp-sub">Left: full heat risk &middot; Right: change in heat exposure</span></h2>
      <span style="flex:1"></span>
      <button class="scn-info-btn" onclick="openScnInfo()" title="How to read this panel">? How to read this</button>
      <button aria-label="Close" class="cmp-close" onclick="closeScnCompareModal()" title="Close (ESC)">&times;</button>
    </div>
    <div class="scn-cmp-body">
      <div class="scn-cmp-col">
        <div class="scn-cmp-picker">
          <label>Left</label>
          <select id="scn-cmp-sel-a" onchange="setScnCompare('a',this.value)"></select>
        </div>
        <div class="scn-cmp-map" id="scn-cmp-map-a"></div>
        <div class="scn-cmp-leg" id="scn-cmp-leg-a"></div>
        <div class="scn-cmp-meta" id="scn-cmp-meta-a"></div>
      </div>
      <div class="scn-cmp-col">
        <div class="scn-cmp-picker">
          <label>Right</label>
          <select id="scn-cmp-sel-b" onchange="setScnCompare('b',this.value)"></select>
        </div>
        <div class="scn-cmp-map" id="scn-cmp-map-b"></div>
        <div class="scn-cmp-leg" id="scn-cmp-leg-b"></div>
        <div class="scn-cmp-meta" id="scn-cmp-meta-b"></div>
      </div>
    </div>
  </div>
</div>

<div id="scn-info-modal" role="dialog" aria-modal="true" aria-labelledby="scn-info-title" onclick="if(event.target===this)closeScnInfo()">
  <div class="scn-info-panel">
    <div class="scn-info-hdr">
      <h3 id="scn-info-title">How to read this</h3>
      <button aria-label="Close" class="scn-info-close" onclick="closeScnInfo()" title="Close (ESC)">&times;</button>
    </div>
    <div class="scn-info-body">
      <div class="scn-info-row">
        <span class="scn-info-chip chip-left"></span>
        <div><b>Left map &mdash; the full heat risk.</b> How hot it gets, who is vulnerable, and how well they can cope, combined into one score. Darker red means higher risk.</div>
      </div>
      <div class="scn-info-row">
        <span class="scn-info-chip chip-right"></span>
        <div><b>Right map &mdash; the change.</b> How heat exposure shifts from the left scenario to the right one. <b class="ink-blue">Blue</b> means less exposed, <b class="ink-red">red</b> means more exposed, white means little change.</div>
      </div>
      <div class="scn-info-key">
        <div class="scn-info-keybar"></div>
        <div class="scn-info-keyends"><span>cooler &middot; less exposed</span><span>hotter &middot; more exposed</span></div>
      </div>
      <p class="scn-info-note">Only heat exposure responds to the climate scenario. Who is vulnerable and their ability to cope stay at today's levels, so the right map isolates the warming signal. The scale is percentile points &mdash; a tract's countywide exposure rank, 0 to 100.</p>
    </div>
    <div class="scn-info-foot">
      <button class="scn-info-ok" onclick="closeScnInfo()">Got it</button>
    </div>
  </div>
</div>

<div id="welcome-modal" class="open" role="dialog" aria-modal="true" aria-labelledby="welcome-title" onclick="if(event.target===this)closeWelcome()">
  <div class="wlc-panel">
    <div class="wlc-hdr">
      <h2 id="welcome-title">Welcome to the San Diego Heat Risk Explorer</h2>
      <button aria-label="Close" class="wlc-close" id="wlc-close-btn" title="Close (ESC)">&times;</button>
    </div>
    <div class="wlc-body">
      <div class="wlc-quickstart">
        <p>Quick start</p>
        <ol>
          <li>Defaults are already selected &mdash; click <b>Build Composite Risk Map</b>.</li>
          <li>Click any tract to see its risk score and dominant driver.</li>
          <li>Under <b>Map Layer</b>, switch to <b>Dominant Driver</b> to see what drives risk countywide.</li>
        </ol>
      </div>
      <p>Or use the left panel to customize your map. Here&rsquo;s what each section does:</p>
      <ul>
        <li><b>Heat Exposure</b> &mdash; pick one heat hazard to map (daytime, nighttime, humid heat, or surface temperature). They affect different places, so you compare them one at a time.</li>
        <li><b>Population Sensitivity / Adaptive Capacity</b> &mdash; two groups of factors that shape heat risk. Turn any on or off to include them in your map.</li>
        <li><b>Build Composite Risk Map</b> &mdash; combines the factors you picked into a single risk score for every census tract.</li>
        <li><b>Map Layer</b> &mdash; switch between the combined risk score, each group on its own, or any single factor.</li>
        <li><b>Jurisdictions</b> &mdash; show city, unincorporated-county, or tribal-land boundaries on the map.</li>
        <li><b>Climate Scenario</b> &mdash; see how risk shifts under future warming, for the year 2050 or 2080.</li>
        <li><b>Export</b> &mdash; save the current map as a PDF or download the data.</li>
        <li><b>Feedback</b> &mdash; <a href="__FEEDBACK_MAILTO_PLACEHOLDER__" style="color:#2563eb">email us</a> about anything that looks wrong or missing. Also at the bottom of the sidebar.</li>
      </ul>
      <p style="margin-top:14px;font-size:11.5px;color:#6b7280;line-height:1.5">
        &copy;&nbsp;2026 Southern California Extreme Heat Research Hub &amp; San Diego Regional Climate Collaborative.
        Code <a href="../LICENSE" target="_blank" rel="noopener" style="color:#6b7280;text-decoration:underline">MIT</a>;
        data &amp; content <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener" style="color:#6b7280;text-decoration:underline">CC&nbsp;BY&nbsp;4.0</a>,
        each source under its own terms.
      </p>
    </div>
    <div class="wlc-footer">
      <a class="wlc-btn secondary" href="../userguide/index.html" target="_blank" rel="noopener noreferrer">User Guide</a>
      <button class="wlc-btn secondary" id="wlc-take-tour">Take a tour</button>
      <button class="wlc-btn" id="wlc-got-it">Got it</button>
    </div>
  </div>
</div>

<div id="tour-overlay" role="dialog" aria-label="Guided tour">
  <div class="tour-dim" id="tour-dim-top"></div>
  <div class="tour-dim" id="tour-dim-right"></div>
  <div class="tour-dim" id="tour-dim-bottom"></div>
  <div class="tour-dim" id="tour-dim-left"></div>
  <div class="tour-spotlight" id="tour-spotlight"></div>
  <div class="tour-card" id="tour-card">
    <div class="tc-arrow"></div>
    <div class="tc-progress" id="tc-progress">Step 1 of 7</div>
    <h3 class="tc-title" id="tc-title"></h3>
    <div class="tc-body" id="tc-body"></div>
    <div class="tc-prompt" id="tc-prompt" style="display:none"></div>
    <div class="tc-footer">
      <button class="tc-skip" id="tc-skip">Skip tour</button>
      <span class="spacer"></span>
      <button class="tc-btn" id="tc-example" style="display:none">See worked example</button>
      <button class="tc-btn" id="tc-back">Back</button>
      <button class="tc-btn primary" id="tc-next">Next</button>
    </div>
  </div>
</div>

<div id="detail-modal" role="dialog" aria-label="Tract Detail" onclick="if(event.target===this)closeDetailDrawer()">
  <div class="dt-panel">
    <div class="dt-hdr">
      <div class="dt-hdr-text">
        <h2 id="dt-title">Tract</h2>
        <div class="dt-sub" id="dt-sub"></div>
      </div>
      <button aria-label="Close" class="dt-close" onclick="closeDetailDrawer()" title="Close (ESC)">&times;</button>
    </div>
    <div class="dt-ctx" id="dt-ctx"></div>
    <div class="dt-body" id="dt-body"></div>
  </div>
</div>

<script>
/* ================================================================
   DATA  (injected by build script)
   ================================================================ */
var GJ  = __GEOJSON_PLACEHOLDER__;
var VM  = __VARMETA_PLACEHOLDER__;
var VSRC = __VARSOURCES_PLACEHOLDER__;
var SRCNAME = __SOURCENAMES_PLACEHOLDER__;    // {sourceUrl: plain-English name} — for the export data dictionary
var VUNITS = __VARUNITS_PLACEHOLDER__;        // {varKey: units string} — for the export data dictionary
var NO_EXPORT = __NOEXPORT_PLACEHOLDER__;     // var keys omitted from Download Data (restricted source terms; still shown on map)
var NO_EXPORT_WHY = __NOEXPORTWHY_PLACEHOLDER__; // {varKey: why it is withheld} — shown in the data dictionary
var EXPORT_META = __EXPORTMETA_PLACEHOLDER__; // {citation,license,sources_url} stamped on every exported row
var DELTAS = __DELTAS_PLACEHOLDER__;         // {period:{ssp:{tasmax:[...],tasmin:[...]}}} or null
var DELTA_MAP = __DELTAMAP_PLACEHOLDER__;    // {varKey:"tasmax"|"tasmin"}
var JURIS = __JURIS_PLACEHOLDER__;           // city jurisdiction GeoJSON or null
var TRIBAL = __TRIBAL_PLACEHOLDER__;         // tribal lands GeoJSON or null
var UNINCORP = __UNINCORP_PLACEHOLDER__;     // unincorporated county boundary GeoJSON or null
var COOLZONES = __COOLZONES_PLACEHOLDER__;   // [{name,address,city,zip,phone,lat,lon,src}, ...] or null
var LST   = __LST_PLACEHOLDER__;             // Landsat LST overlay meta or null
var POP        = __POP_PLACEHOLDER__;        // population density overlay meta or null
var DS_ENC     = __DS_ENC_PLACEHOLDER__;     // base64 PNG: R=exposure pct, G/B=tract idx, A=pop mask
var DS_GEOIDS  = __DS_GEOIDS_PLACEHOLDER__;  // [GEOID,...] in encoded-tract-index order
var DS_DENS    = __DS_DENS_PLACEHOLDER__;    // base64 PNG: R=log-scaled population density (0-255), A=255
var IMPERVIOUS = __IMPERVIOUS_PLACEHOLDER__; // impervious surface overlay meta or null
var CANOPY     = __CANOPY_PLACEHOLDER__;     // tree canopy overlay meta or null
var POLARITY = __POLARITY_PLACEHOLDER__;     // {varKey: true if higher=worse(red)}
var HEAT_MODES = __HEATMODES_PLACEHOLDER__;  // [{key,label,layers:[...],desc}] — exposure is single-select, not PCA

var PMETA = {
  exposure:    {label:"Heat Exposure",     color:"#c0392b"},
  sensitivity: {label:"Population Sensitivity", color:"#6a51a3"},
  adaptive:    {label:"Adaptive Capacity", color:"#006d2c"}
};
// Accent for the Composite summary card. It isn't a pillar, so it has no PMETA
// hue — this is the brand blue used by the PDF title rule and the popup buttons,
// which keeps it from reading as a fourth pillar next to the three PMETA cards.
var _DT_CARD_ACCENT = "#2563eb";
var SREF = {exposure:"mean_tmax", sensitivity:"bphigh_prev", adaptive:"income_pctile"};

/* ================================================================
   STATE
   ================================================================ */
var enabled = {};   // varKey -> bool (sensitivity/adaptive only; exposure uses heatMode)
var heatMode = (HEAT_MODES[0]?HEAT_MODES[0].key:"daytime"); // selected exposure hazard
var curLayer = "composite";
var rawVar   = null; // key of raw variable being previewed (before PCA)
var results  = null;
// The composite is a COMMITTED artifact: only compute() ever writes _comp and the
// pillar scores. Changing an input (heat mode, enabled variables) marks it stale
// instead of silently recomputing — the old behaviour redrew the composite the
// instant you picked a new heat mode, so it looked like the map was
// live-calculating and there was no way to tell a rebuild had happened.
var compositeStale = false;
var curScenario = "present"; // "present" | "ssp245" | "ssp370" | "ssp585"
var curPeriod   = "";        // "" (present) | "2050" | "2080"
var cbMode = (function(){try{return localStorage.getItem('hrv-cb-mode')==='1';}catch(e){return false;}})();
// Exposure is a single-select heat mode, so its raw variables are excluded from
// the `enabled` toggle set — only Sensitivity and Adaptive Capacity use toggles.
// Open with a curated, high-contrast default rather than every variable: turning
// on all 73 makes each pillar's PCA collapse toward a generic "everything
// correlates" axis and washes out spatial contrast. Defaults below are one
// composite signal + a couple of focused drivers per pillar (exposure stays on
// the daytime heat mode). Every other variable stays in the panel, unchecked.
var DEFAULT_ON = {
  // Sensitivity
  RPL_THEMES:1,        // Overall SVI
  EP_AGE65:1,          // Age 65+
  heat_hosp_day:1,     // Heat-Day Hospitalization Rate Ratio
  // Adaptive Capacity
  ac_prob:1,           // AC Prevalence (LACE)
  treecanopy_pctile:1, // Tree Canopy
  income_pctile:1      // Income
};
VM.forEach(function(v){ enabled[v.key] = (v.pillar!=="exposure") && !!DEFAULT_ON[v.key]; });

// Resolve a heat-mode key to its definition object.
function heatModeDef(key){
  for(var i=0;i<HEAT_MODES.length;i++)if(HEAT_MODES[i].key===key)return HEAT_MODES[i];
  return HEAT_MODES[0]||null;
}

// The inputs a composite is scored from. Build reads them off the panel; after
// that, everything that describes or re-scores the build reads the copy saved on
// `results`, because the panel may have moved on (the composite is then stale).
function _panelSettings(){
  var on={sensitivity:[],adaptive:[]};
  VM.forEach(function(v){if(v.pillar!=="exposure"&&enabled[v.key])on[v.pillar].push(v.key);});
  return {heatMode:heatMode,on:on};
}
// The build's settings, or the panel's before the first build.
function _builtSettings(){ return results?results.settings:_panelSettings(); }

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
  // Re-score the committed build under the new scenario. A bare compute() is a
  // Build: it would apply pending panel changes and clear the stale flag.
  if(results){compute(results.settings);}
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
// compare-tracts modal. Knows about temperature (°C + °F), 0–1 fractions
// (probabilities, AC-rank), pre-computed percentile vars (HPI _pctile, SVI
// RPL_*) and CDC PLACES / SVI EP_* / pct_* / humidity values that are stored
// as 0–100 percentages.
var _CMP_TEMP_KEYS={mean_tmax:1,p85_tmax:1,p99_tmax:1,max_tmax:1,
  mean_lst:1,p85_wbgt:1,p99_wbgt:1,
  mean_tmin_overnight:1,p90_tmin_overnight:1,p95_dewpoint:1};
var _CMP_FRAC_KEYS={no_relief_fraction:1,ac_prob:1,ac_prob_romitti:1};
// Absolute Celsius -> Fahrenheit. Temperatures are shown in both units so
// non-scientists can read them. NOTE: this is for absolute values; temperature
// *differences* (deltas) convert with the scale only (x9/5, no +32 offset).
function cToF(c){return c*9/5+32;}
function fmtCmpRaw(key,val){
  if(val==null||isNaN(val))return '\\u2014';
  var v=+val;
  if(_CMP_TEMP_KEYS[key])return v.toFixed(1)+'\\u00b0C / '+cToF(v).toFixed(1)+'\\u00b0F';
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
// Esri Light Gray Canvas — free, no API key, attribution required. It
// replaces CARTO Positron, which started stamping "API KEY REQUIRED" across
// every keyless tile. Two differences from CARTO drive the helpers below:
// Esri splits the style into an unlabeled base plus a transparent label
// raster, so the light basemap is always TWO tile layers; and it has no tiles
// past z16 (it serves a "Map data not yet available" placeholder there), so
// maxNativeZoom upscales beyond that instead of requesting them.
var ESRI_CANVAS = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
var LIGHT_BASE_URL = ESRI_CANVAS + "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}";
var LIGHT_REF_URL  = ESRI_CANVAS + "World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}";
var LIGHT_MAX_NATIVE_ZOOM = 16;
var LIGHT_ATTRIB = "Esri, HERE, Garmin, \\u00a9 OpenStreetMap contributors";
var BASEMAPS = {
  light: {
    url: LIGHT_BASE_URL,
    labels: LIGHT_REF_URL,
    opts: {attribution:LIGHT_ATTRIB, maxNativeZoom:LIGHT_MAX_NATIVE_ZOOM, maxZoom:19}
  },
  satellite: {
    // Esri World Imagery — free, no API key, attribution required.
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    opts: {attribution:"Imagery \\u00a9 Esri, Maxar, Earthstar Geographics, and the GIS User Community", maxZoom:19}
  }
};

// The two rasters of the light basemap, in draw order, ready to drop into a
// map or a LayerGroup. Every mini-map in this file builds its base through
// here so the label layer is never forgotten.
function lightTiles(maxZoom){
  var o={maxNativeZoom:LIGHT_MAX_NATIVE_ZOOM, maxZoom:maxZoom||19};
  return [L.tileLayer(LIGHT_BASE_URL,o), L.tileLayer(LIGHT_REF_URL,o)];
}
// One layer handle for a BASEMAPS entry. A multi-raster basemap comes back as
// a LayerGroup, which has no bringToBack() of its own — forward it to the
// children so setBasemap() can treat both entries identically.
function makeBasemap(name){
  var b=BASEMAPS[name];
  if(!b.labels) return L.tileLayer(b.url,b.opts);
  var g=L.layerGroup([L.tileLayer(b.url,b.opts), L.tileLayer(b.labels,b.opts)]);
  g.bringToBack=function(){g.eachLayer(function(l){l.bringToBack();});return g;};
  return g;
}
var currentBasemap = 'satellite';
var basemapLayer = makeBasemap('satellite').addTo(map);
L.control.scale({position:'bottomright'}).addTo(map);
var tLayer = null;
// User-adjustable opacity for the composite-risk choropleth (see #map-opacity-slider).
// Read by the tract style function in drawTracts() so changes survive resetStyle().
var tractOpacity = 1;

/* ===== "Downscaled" mode: the slide hero's per-pixel analysis, live ==========
   Composite/Driver/Driver-Mix painted at ~100 m using the DOWNSCALED LST exposure
   field plus each tract's CURRENT sensitivity/adaptive score, masked to the
   populated footprint -- i.e. the analysis itself runs sub-tract, not a clipped
   choropleth. Encoded raster (base64, no canvas taint): R = exposure percentile,
   G/B = tract index (-> GEOID via DS_GEOIDS), A = populated mask. */
var dsRenderOn = false;
var _dsReady = false, _dsData = null, _dsW = 0, _dsH = 0;
var _dsIdx2Feat = null, _dsCanvas = null, _dsOverlay = null;
var DS_LAYERS = {composite:1, driver:1, drivermix:1};

// Shared tract style: in downscaled mode the vector fills/borders hide so only the
// ~100 m overlay shows (tracts stay clickable for popups). Survives resetStyle().
function tractStyle(f){
  var idx=window._gIdx[f.properties.GEOID]||0;
  return {fillColor:fcolor(f.properties,idx),color:"#fff",
          weight:dsRenderOn?0:0.5, fillOpacity:dsRenderOn?0:tractOpacity};
}

// Decode the embedded encoded raster once (base64 => same-origin => no taint).
(function(){
  if(typeof DS_ENC==='undefined'||!DS_ENC) return;
  var img=new Image();
  img.onload=function(){
    var c=document.createElement('canvas'); c.width=img.naturalWidth; c.height=img.naturalHeight;
    c.getContext('2d').drawImage(img,0,0);
    _dsData=c.getContext('2d').getImageData(0,0,c.width,c.height).data; _dsW=c.width; _dsH=c.height;
    _dsReady=true;
    if(dsRenderOn) renderDownscaled();
  };
  img.src=DS_ENC;
})();

// Companion density raster (R = log-scaled population density, same grid) for the
// bivariate risk x population downscaled composite. GrPink corners from the report
// slide: C00 low-low, CP low-risk/high-pop, CR high-risk/low-pop, CRP high-high.
var _dsDensData=null, _dsDensReady=false;
var DS_BIV=[[255,255,178],[171,155,136],[227,74,51],[74,0,0]];
(function(){
  if(typeof DS_DENS==='undefined'||!DS_DENS) return;
  var img=new Image();
  img.onload=function(){
    var c=document.createElement('canvas'); c.width=img.naturalWidth; c.height=img.naturalHeight;
    c.getContext('2d').drawImage(img,0,0);
    _dsDensData=c.getContext('2d').getImageData(0,0,c.width,c.height).data; _dsDensReady=true;
    if(dsRenderOn) renderDownscaled();
  };
  img.src=DS_DENS;
})();

// Repaint the downscaled overlay: per ~100 m pixel, composite = (downscaled exposure
// percentile + the tract's live sensitivity + its adaptive deficit)/3, colored the
// same way fcolor() colors composite/driver/drivermix. Recolors on score/layer change.
function renderDownscaled(){
  if(!_dsReady||!window._gIdx||!POP||!POP.bounds) return;
  var N=DS_GEOIDS.length;
  if(!_dsIdx2Feat){_dsIdx2Feat=new Int32Array(N);
    for(var t=0;t<N;t++){var fi=window._gIdx[DS_GEOIDS[t]]; _dsIdx2Feat[t]=(fi==null?-1:fi);}}
  // live per-tract sensitivity / adaptive-deficit, keyed by encoded tract index
  var sA=new Float32Array(N), aA=new Float32Array(N), ok=new Uint8Array(N);
  for(var u=0;u<N;u++){var fi2=_dsIdx2Feat[u]; if(fi2<0) continue;
    var pr=GJ.features[fi2].properties, sv=pr._sensitivity, av=pr._adaptive;
    if(sv==null||av==null) continue; sA[u]=sv; aA[u]=100-av; ok[u]=1;}
  var lyr=DS_LAYERS[curLayer]?curLayer:'composite', cr=pal('CR'), drgb=pal('DRGB');
  // ramp LUTs (101 steps) so the hot per-pixel loop never allocates
  var compLUT=new Uint8Array(303);
  for(var c=0;c<=100;c++){var rc=interpC(cr,(c-2)/96); compLUT[c*3]=rc[0]; compLUT[c*3+1]=rc[1]; compLUT[c*3+2]=rc[2];}
  var drvLUT=null;
  if(lyr==='driver'){drvLUT={};['exposure','sensitivity','adaptive'].forEach(function(d){
    var arr=new Uint8Array(303), b=drgb[d];
    for(var cc=0;cc<=100;cc++){var inn=0.15+0.85*(cc/100);
      arr[cc*3]=Math.round(255-inn*(255-b[0])); arr[cc*3+1]=Math.round(255-inn*(255-b[1])); arr[cc*3+2]=Math.round(255-inn*(255-b[2]));}
    drvLUT[d]=arr;});}
  // bivariate (risk x population) LUT for the downscaled composite (slide hero scheme)
  var biv=(lyr==='composite' && _dsDensReady && _dsDensData), NB=64, bivLUT=null;
  if(biv){ bivLUT=new Uint8Array(NB*NB*3);
    for(var pi=0;pi<NB;pi++){var pp=pi/(NB-1);
      for(var ri=0;ri<NB;ri++){var rr=ri/(NB-1), bo=(pi*NB+ri)*3, q=1-pp, qq=1-rr;
        bivLUT[bo]  =Math.round(q*qq*DS_BIV[0][0]+pp*qq*DS_BIV[1][0]+q*rr*DS_BIV[2][0]+pp*rr*DS_BIV[3][0]);
        bivLUT[bo+1]=Math.round(q*qq*DS_BIV[0][1]+pp*qq*DS_BIV[1][1]+q*rr*DS_BIV[2][1]+pp*rr*DS_BIV[3][1]);
        bivLUT[bo+2]=Math.round(q*qq*DS_BIV[0][2]+pp*qq*DS_BIV[1][2]+q*rr*DS_BIV[2][2]+pp*rr*DS_BIV[3][2]);
      }}}
  var W=_dsW,H=_dsH,src=_dsData,n=W*H;
  if(!_dsCanvas){_dsCanvas=document.createElement('canvas');_dsCanvas.width=W;_dsCanvas.height=H;}
  var octx=_dsCanvas.getContext('2d'), out=octx.createImageData(W,H), o=out.data;
  for(var p=0;p<n;p++){var k=p*4;
    if(src[k+3]!==255){o[k+3]=0; continue;}
    var ti=(src[k+1]<<8)|src[k+2];
    if(ti>=N||!ok[ti]){o[k+3]=0; continue;}
    var e=src[k], s=sA[ti], a=aA[ti], comp=(e+s+a)/3, ci=(comp+0.5)|0;
    if(ci<0)ci=0; else if(ci>100)ci=100;
    if(lyr==='drivermix'){var rgb=_triMix(e,s,a,drgb); o[k]=rgb[0]; o[k+1]=rgb[1]; o[k+2]=rgb[2];}
    else if(lyr==='driver'){var d=(e>=s&&e>=a)?'exposure':(s>=a)?'sensitivity':'adaptive', ar=drvLUT[d], f=ci*3;
      o[k]=ar[f]; o[k+1]=ar[f+1]; o[k+2]=ar[f+2];}
    else if(biv){var rb=(comp-2)/96; rb=rb<0?0:(rb>1?1:rb);
      var pb=_dsDensData[k]/255, rri=(rb*(NB-1)+0.5)|0, ppi=(pb*(NB-1)+0.5)|0, bb=(ppi*NB+rri)*3;
      o[k]=bivLUT[bb]; o[k+1]=bivLUT[bb+1]; o[k+2]=bivLUT[bb+2];}
    else {var g=ci*3; o[k]=compLUT[g]; o[k+1]=compLUT[g+1]; o[k+2]=compLUT[g+2];}
    o[k+3]=255;
  }
  octx.putImageData(out,0,0);
  var url=_dsCanvas.toDataURL(), b2=POP.bounds, bnds=[[b2.south,b2.west],[b2.north,b2.east]];
  if(!_dsOverlay){
    if(!map.getPane('dsPane')){var pn=map.createPane('dsPane');pn.style.zIndex=410;pn.style.pointerEvents='none';}
    _dsOverlay=L.imageOverlay(url,bnds,{opacity:tractOpacity,interactive:false,pane:'dsPane'});
  } else { _dsOverlay.setUrl(url); }
}

function setDsRender(on){
  dsRenderOn=!!on;
  if(on){
    renderDownscaled();
    if(_dsOverlay && !map.hasLayer(_dsOverlay)) _dsOverlay.addTo(map);
  } else if(_dsOverlay && map.hasLayer(_dsOverlay)){ map.removeLayer(_dsOverlay); }
  if(tLayer) tLayer.setStyle(tractStyle);
  _updateBivLegend();
}

// Bivariate (risk x population) legend, shown bottom-left only for the downscaled
// COMPOSITE view (the one painted with the GrPink bilinear scheme).
function _drawBivSwatch(cv){
  // discrete 3x3 GrPink matrix, matching the report slide legend: population -> right,
  // risk -> up, with thin gaps between the nine cells.
  var x=cv.getContext('2d'), W=cv.width, H=cv.height, N=3, gap=3;
  x.clearRect(0,0,W,H);
  var cw=(W-gap*(N-1))/N, ch=(H-gap*(N-1))/N;
  for(var crow=0;crow<N;crow++)for(var col=0;col<N;col++){
    var pp=(col+0.5)/N, rr=(N-1-crow+0.5)/N, q=1-pp, qq=1-rr;
    var R=Math.round(q*qq*DS_BIV[0][0]+pp*qq*DS_BIV[1][0]+q*rr*DS_BIV[2][0]+pp*rr*DS_BIV[3][0]);
    var G=Math.round(q*qq*DS_BIV[0][1]+pp*qq*DS_BIV[1][1]+q*rr*DS_BIV[2][1]+pp*rr*DS_BIV[3][1]);
    var B=Math.round(q*qq*DS_BIV[0][2]+pp*qq*DS_BIV[1][2]+q*rr*DS_BIV[2][2]+pp*rr*DS_BIV[3][2]);
    x.fillStyle='rgb('+R+','+G+','+B+')';
    x.fillRect(Math.round(col*(cw+gap)), Math.round(crow*(ch+gap)), Math.ceil(cw), Math.ceil(ch));
  }
}
// Bivariate legend lives at the TOP of the bottom-left control stack, so it sits
// ABOVE the Data Layers / address cards instead of overlapping that corner.
var _bivEl=null;
function _updateBivLegend(){
  var show=dsRenderOn && curLayer==='composite' && typeof DS_DENS!=='undefined' && DS_DENS;
  var host=document.getElementById('map-controls-bl');
  if(show && host){
    if(!_bivEl){
      _bivEl=document.createElement('div'); _bivEl.className='biv-legend';
      _bivEl.innerHTML='<div class="biv-title">Risk \\u00d7 pop. density</div>'+
        '<canvas class="biv-sw" width="74" height="74"></canvas>'+
        '<div class="biv-axes"><span>\\u2191 risk</span><span>people \\u2192</span></div>';
    }
    if(_bivEl.parentNode!==host) host.insertBefore(_bivEl, host.firstChild);
    _drawBivSwatch(_bivEl.querySelector('.biv-sw'));
  } else if(_bivEl && _bivEl.parentNode){ _bivEl.parentNode.removeChild(_bivEl); }
}

function setBasemap(name){
  if(!BASEMAPS[name] || name===currentBasemap) return;
  if(basemapLayer){map.removeLayer(basemapLayer);}
  basemapLayer = makeBasemap(name).addTo(map);
  basemapLayer.bringToBack();
  currentBasemap = name;
  var btns = document.querySelectorAll('#basemap-picker .bm-thumb');
  for(var i=0;i<btns.length;i++){
    if(btns[i].getAttribute('data-bm')===name) btns[i].classList.add('active');
    else btns[i].classList.remove('active');
  }
}

function setTractOpacity(pct){
  tractOpacity = (+pct)/100;
  var lbl = document.getElementById('map-opacity-val');
  if(lbl) lbl.textContent = pct+'%';
  if(dsRenderOn){ if(_dsOverlay) _dsOverlay.setOpacity(tractOpacity); }
  else if(tLayer){ tLayer.setStyle({fillOpacity: tractOpacity}); }
}

/* ===== Data Layers (informational reference overlays) =====================
   Single-select dropdown of context layers drawn over the basemap, separate
   from the PCA risk analysis. Rasters (population/impervious/canopy/LST) are
   PNG imageOverlays; land use reuses the SANDAG PMTiles machinery; heat
   hospitalizations is a tract choropleth. All sit in landusePane (z 450). */
var RASTER_META = {population:POP, impervious:IMPERVIOUS, canopy:CANOPY, lst:LST};
var _dataLayer = "";           // current selection id ("" = none)
var _dataLayerObj = null;      // current non-landuse Leaflet layer (imageOverlay/geojson)
var _dataLayerOpacity = 0.80;
var HOSP_RAMP = [[255,247,236],[254,196,79],[217,95,14],[153,52,4]];  // sequential YlOrBr-ish
var _hospDomain = null;

function _ensureDataPane(){
  if(!map.getPane('landusePane')){ map.createPane('landusePane'); map.getPane('landusePane').style.zIndex=450; }
}
function rasterVariant(id){
  var m = RASTER_META[id];
  if(!m) return null;
  if(m.variants && m.variants[cbMode?'cb':'default']) return m.variants[cbMode?'cb':'default'];
  return {url:m.url, colormap:m.colormap, legend:m.legend};
}

function setDataLayer(id){
  _ensureDataPane();
  if(_dataLayerObj){ try{map.removeLayer(_dataLayerObj);}catch(e){} _dataLayerObj=null; }
  if(_mainLanduseShown && id!=='landuse'){ toggleMainLanduse(); }   // turn land use off
  _dataLayer = id || "";
  var opw = document.getElementById('data-layer-opacity-wrap');
  var leg = document.getElementById('data-layer-legend');
  var chw = document.getElementById('chs-level-wrap');
  if(leg){ leg.style.display='none'; leg.innerHTML=''; }
  if(chw){ chw.style.display='none'; }
  if(!_dataLayer){ if(opw) opw.style.display='none'; return; }

  if(id==='landuse'){                       // reuse existing SANDAG land-use machinery
    if(!_mainLanduseShown) toggleMainLanduse();
    if(opw) opw.style.display='none';        // protomaps fill opacity is fixed
    return;
  }
  if(opw) opw.style.display='';

  if(id==='hosp_day'){
    _dataLayerObj = buildHospChoropleth();
    if(_dataLayerObj) _dataLayerObj.addTo(map);
    renderDataLayerLegend('hosp_day');
    return;
  }
  if(id==='excess_er'){
    _dataLayerObj = buildExcessErChoropleth();
    if(_dataLayerObj) _dataLayerObj.addTo(map);
    renderDataLayerLegend('excess_er');
    return;
  }
  if(id==='threshold'){
    if(chw) chw.style.display='flex';
    _dataLayerObj = buildThresholdChoropleth();
    if(_dataLayerObj) _dataLayerObj.addTo(map);
    renderDataLayerLegend('threshold');
    return;
  }
  var meta = RASTER_META[id], v = rasterVariant(id);
  if(!meta || !v){ if(leg){ leg.style.display='block'; leg.innerHTML='<i style="color:#999">layer unavailable</i>'; } return; }
  var b = meta.bounds;
  _dataLayerObj = L.imageOverlay(v.url, [[b.south,b.west],[b.north,b.east]],
                                 {opacity:_dataLayerOpacity, pane:'landusePane'}).addTo(map);
  renderDataLayerLegend(id);
}

function setDataLayerOpacity(pct){
  _dataLayerOpacity = (+pct)/100;
  var lbl = document.getElementById('data-layer-opacity-val'); if(lbl) lbl.textContent = pct+'%';
  if(_dataLayerObj){
    if(typeof _dataLayerObj.setOpacity==='function') _dataLayerObj.setOpacity(_dataLayerOpacity);
    else if(typeof _dataLayerObj.setStyle==='function') _dataLayerObj.setStyle({fillOpacity:_dataLayerOpacity});
  }
}

function _hospColor(v){
  if(v==null||isNaN(v)) return '#cccccc';
  var d=_hospDomain||[0,1], span=(d[1]-d[0])||1, t=Math.max(0,Math.min(1,(v-d[0])/span));
  return toHex(interpC(HOSP_RAMP, t));
}
function buildHospChoropleth(){
  if(typeof GJ==='undefined' || !GJ.features) return null;
  var vals = GJ.features.map(function(f){return +f.properties.heat_hosp_day;}).filter(function(v){return !isNaN(v);});
  if(!vals.length) return null;
  vals.sort(function(a,b){return a-b;});
  _hospDomain = [vals[0], vals[vals.length-1]];
  return L.geoJSON(GJ,{pane:'landusePane',
    style:function(f){return{fillColor:_hospColor(+f.properties.heat_hosp_day),color:'#fff',weight:0.3,fillOpacity:_dataLayerOpacity};},
    onEachFeature:function(f,lyr){var v=+f.properties.heat_hosp_day;
      lyr.bindPopup('<div style="font-size:11px"><b>Heat hospitalizations</b><br>'+
        'Standardized heat-day score: <b>'+(isNaN(v)?'n/a':v.toFixed(2))+'</b><br>'+
        '<span style="color:#888">0 = county-area average; higher = more heat-day hospitalizations than typical</span><br>'+
        '<span style="color:#888">ZIP-code resolution (CalHeatScore), shown per tract</span></div>');}
  });
}

/* UCLA Heat Maps — excess-ER + CalHeatScore thresholds, ZIP→tract centroid xwalk */
var EXR_RAMP = [[255,255,178],[254,204,92],[253,141,60],[240,59,32],[189,0,38]];   // YlOrRd
var TH_RAMP  = [[44,123,182],[171,217,233],[255,255,191],[253,174,97],[215,25,28]]; // cool->hot
var _exrDomain = null, _thDomain = null, _thLevel = 1, _thSharedDomain = null;
var CHS_NAME = {1:'1 — Mild (onset)', 2:'2 — Moderate', 3:'3 — High', 4:'4 — Severe'};

function _rampColor(ramp, v, d){
  if(v==null||isNaN(v)) return '#cccccc';
  d=d||[0,1]; var span=(d[1]-d[0])||1, t=Math.max(0,Math.min(1,(v-d[0])/span));
  return toHex(interpC(ramp, t));
}
function _uclaPopup(p){
  var er=+p.excess_er_rate;
  function f(x){return (x==null||isNaN(+x))?'n/a':(+x).toFixed(0)+'°F';}
  return '<div style="font-size:11px"><b>UCLA Heat Maps</b><br>'+
    'Excess ER visits: <b>'+(isNaN(er)?'n/a':er.toFixed(1))+'</b> per 10,000<br>'+
    '<span style="color:#888">(CalHeatScore 4, 2008–2018)</span>'+
    '<div style="margin-top:3px">Heat-risk onset temperature:</div>'+
    'CHS1 '+f(p.chs1_th)+' · CHS2 '+f(p.chs2_th)+'<br>CHS3 '+f(p.chs3_th)+' · CHS4 '+f(p.chs4_th)+'<br>'+
    '<span style="color:#888">ZIP-resolution (UCLA/CalHeatScore), per tract</span></div>';
}
function buildExcessErChoropleth(){
  if(typeof GJ==='undefined' || !GJ.features) return null;
  var vals = GJ.features.map(function(f){return +f.properties.excess_er_rate;}).filter(function(v){return !isNaN(v);});
  if(!vals.length) return null;
  vals.sort(function(a,b){return a-b;});
  _exrDomain = [vals[0], vals[vals.length-1]];
  return L.geoJSON(GJ,{pane:'landusePane',
    style:function(f){return{fillColor:_rampColor(EXR_RAMP,+f.properties.excess_er_rate,_exrDomain),color:'#fff',weight:0.3,fillOpacity:_dataLayerOpacity};},
    onEachFeature:function(f,lyr){lyr.bindPopup(_uclaPopup(f.properties));}});
}
function _thSharedDom(){   // min(CHS1)..max(CHS4) — fixed scale so switching levels visibly warms the map
  if(_thSharedDomain) return _thSharedDomain;
  var lo=Infinity, hi=-Infinity;
  if(typeof GJ!=='undefined' && GJ.features){
    GJ.features.forEach(function(f){
      for(var k=1;k<=4;k++){ var v=+f.properties['chs'+k+'_th'];
        if(!isNaN(v)){ if(v<lo)lo=v; if(v>hi)hi=v; } }
    });
  }
  _thSharedDomain = (lo<hi)?[lo,hi]:[70,110];
  return _thSharedDomain;
}
function buildThresholdChoropleth(){
  if(typeof GJ==='undefined' || !GJ.features) return null;
  var key='chs'+_thLevel+'_th';
  var vals = GJ.features.map(function(f){return +f.properties[key];}).filter(function(v){return !isNaN(v);});
  if(!vals.length) return null;
  vals.sort(function(a,b){return a-b;});
  _thDomain = [vals[0], vals[vals.length-1]];   // this level's own span (for the legend note)
  var dom = _thSharedDom();                      // fixed color scale across CHS 1–4
  return L.geoJSON(GJ,{pane:'landusePane',
    style:function(f){return{fillColor:_rampColor(TH_RAMP,+f.properties[key],dom),color:'#fff',weight:0.3,fillOpacity:_dataLayerOpacity};},
    onEachFeature:function(f,lyr){lyr.bindPopup(_uclaPopup(f.properties));}});
}
function setThresholdLevel(lv){
  _thLevel=+lv;
  if(_dataLayer==='threshold'){
    if(_dataLayerObj){ try{map.removeLayer(_dataLayerObj);}catch(e){} }
    _dataLayerObj=buildThresholdChoropleth();
    if(_dataLayerObj) _dataLayerObj.addTo(map);
    renderDataLayerLegend('threshold');
  }
}

function renderDataLayerLegend(id){
  var leg=document.getElementById('data-layer-legend'); if(!leg) return;
  leg.style.display='block';
  if(id==='hosp_day'){
    var d=_hospDomain||[0,1];
    var hstops=HOSP_RAMP.map(function(c){return 'rgb('+c.join(',')+')';}).join(',');
    leg.innerHTML='<div style="font-weight:600;margin-bottom:2px">Heat hospitalizations</div>'+
      '<div class="dt-lst-bar" style="background:linear-gradient(to right,'+hstops+')"></div>'+
      '<div class="dt-lst-ticks"><span>'+d[0].toFixed(1)+'</span><span>standardized score</span><span>'+d[1].toFixed(1)+'</span></div>'+
      '<div style="font-size:9.5px;color:#888;margin-top:3px;line-height:1.3">0 = county-area average. ZIP-code resolution (CalHeatScore), shown per tract.</div>';
    return;
  }
  if(id==='excess_er'){
    var de=_exrDomain||[0,1];
    var es=EXR_RAMP.map(function(c){return 'rgb('+c.join(',')+')';}).join(',');
    leg.innerHTML='<div style="font-weight:600;margin-bottom:2px">Excess heat-related ER visits</div>'+
      '<div class="dt-lst-bar" style="background:linear-gradient(to right,'+es+')"></div>'+
      '<div class="dt-lst-ticks"><span>'+de[0].toFixed(1)+'</span><span>per 10,000 · CHS 4 · 2008–2018</span><span>'+de[1].toFixed(1)+'</span></div>'+
      '<div style="font-size:9.5px;color:#888;margin-top:3px;line-height:1.3">UCLA Heat Maps. ZIP-resolution (CalHeatScore), shown per tract.</div>';
    return;
  }
  if(id==='threshold'){
    var sd=_thSharedDom(), lv=_thDomain||sd;
    var ts=TH_RAMP.map(function(c){return 'rgb('+c.join(',')+')';}).join(',');
    leg.innerHTML='<div style="font-weight:600;margin-bottom:2px">Heat-risk threshold — CalHeatScore '+CHS_NAME[_thLevel]+'</div>'+
      '<div class="dt-lst-bar" style="background:linear-gradient(to right,'+ts+')"></div>'+
      '<div class="dt-lst-ticks"><span>'+sd[0].toFixed(0)+'°F</span><span>daily high temp (fixed scale)</span><span>'+sd[1].toFixed(0)+'°F</span></div>'+
      '<div style="font-size:9.5px;color:#888;margin-top:3px;line-height:1.3">This level: SD tracts '+lv[0].toFixed(0)+'–'+lv[1].toFixed(0)+'°F. Scale fixed across CHS 1–4, so higher levels shade warmer. UCLA Heat Maps, per tract.</div>';
    return;
  }
  var meta=RASTER_META[id], v=rasterVariant(id);
  if(!meta || !v){ leg.innerHTML=''; return; }
  var label = meta.label || (id==='lst'?'Land surface temperature':id);
  var units = meta.units || (id==='lst'?'annual mean':'');
  var stops = v.legend || [];
  var grad = stops.map(function(s){return 'rgb('+s.rgb.join(',')+')';}).join(',');
  var lo = stops.length?stops[0].value:'', hi = stops.length?stops[stops.length-1].value:'';
  leg.innerHTML='<div style="font-weight:600;margin-bottom:2px">'+label+
    (units?' <span style="font-weight:400;color:#888">('+units+')</span>':'')+'</div>'+
    '<div class="dt-lst-bar" style="background:linear-gradient(to right,'+grad+')"></div>'+
    '<div class="dt-lst-ticks"><span>'+lo+'</span><span>'+hi+'</span></div>';
}

/* ---- Main-map land-use overlay ---- */
if(!map.getPane('landusePane')){
  map.createPane('landusePane');
  map.getPane('landusePane').style.zIndex = 450;
}
var _mainLanduseLayer = null;
var _mainLanduseShown = false;

map.on('click', function(e){
  if(!_mainLanduseShown || !_mainLanduseLayer) return;
  if(typeof _mainLanduseLayer.queryTileFeaturesDebug !== 'function') return;
  var hits;
  try{ hits = _mainLanduseLayer.queryTileFeaturesDebug(e.latlng.lng, e.latlng.lat, 16); }catch(_){ return; }
  if(!hits || !hits.size) return;
  var feats = null;
  hits.forEach(function(arr){ if(!feats && arr && arr.length) feats = arr; });
  if(!feats || !feats.length) return;
  var p = (feats[0].feature && feats[0].feature.props) || {};
  var cat = _luCategoryName(p);
  var label = p.Landuse || cat;
  var lu = p.lu || '';
  L.popup({offset:[0,-4]}).setLatLng(e.latlng)
    .setContent('<div style="font-size:11px;line-height:1.35"><b>'+label+'</b><br>'+
      '<span style="color:#888">'+cat+(lu?' &middot; LU '+lu:'')+'</span></div>')
    .openOn(map);
});

function toggleMainLanduse(){
  _mainLanduseShown = !_mainLanduseShown;
  var btn = document.getElementById('lu-toggle-btn');
  if(_mainLanduseShown){
    if(!_mainLanduseLayer){
      _mainLanduseLayer = _buildLanduseLayer();
      if(_mainLanduseLayer) _mainLanduseLayer.addTo(map);
    } else {
      _mainLanduseLayer.addTo(map);
    }
    if(btn) btn.classList.add('active');
  } else {
    if(_mainLanduseLayer){ try{ map.removeLayer(_mainLanduseLayer); }catch(_){} }
    if(btn) btn.classList.remove('active');
  }
  _renderMainLuLegend();
}

function _renderMainLuLegend(){
  var el = document.getElementById('lu-main-legend');
  if(!el) return;
  if(!_mainLanduseShown){ el.classList.remove('open'); el.innerHTML=''; return; }
  var cats = LU_CATS.concat([LU_WATER_CAT]);
  el.innerHTML = '<div style="font-weight:700;margin-bottom:4px;color:#374151">Land Use (SANDAG)</div>' +
    cats.map(function(c){
      return '<div class="lu-main-item"><span class="lu-main-swatch" style="background:'+c.color+'"></span>'+c.name+'</div>';
    }).join('');
  el.classList.add('open');
}

/* ================================================================
   ADDRESS SEARCH (sidebar bottom)
   Geocodes via OpenStreetMap Nominatim (no API key, biased to SD County),
   drops a marker, finds the containing census tract via point-in-polygon,
   and opens the tract's existing popup so variable info appears.
   ================================================================ */
var addrMarker = null;
var ADDR_BBOX = {minLat:32.45, maxLat:33.55, minLon:-117.65, maxLon:-115.95};

function _addrSetStatus(msg, isErr){
  var el = document.getElementById('addr-status'); if(!el) return;
  el.className = isErr ? 'err' : '';
  el.innerHTML = msg || '';
}

function clearAddressMarker(){
  if(addrMarker){map.removeLayer(addrMarker); addrMarker = null;}
  _addrSetStatus('');
  var cb = document.getElementById('addr-clear'); if(cb) cb.style.display = 'none';
  var inp = document.getElementById('addr-input'); if(inp) inp.value = '';
  if(typeof _addrHideSuggest === 'function') _addrHideSuggest();
  _addrLastQuery = '';
}

// Ray-cast point-in-polygon (single ring; ring is [[lon,lat],...]).
function _addrPointInRing(lon, lat, ring){
  var inside = false;
  for(var i=0, j=ring.length-1; i<ring.length; j=i++){
    var xi=ring[i][0], yi=ring[i][1], xj=ring[j][0], yj=ring[j][1];
    if(((yi>lat)!==(yj>lat)) && (lon < (xj-xi)*(lat-yi)/((yj-yi)||1e-12) + xi))
      inside = !inside;
  }
  return inside;
}
function _addrPointInPoly(lon, lat, rings){
  if(!rings || !rings.length) return false;
  if(!_addrPointInRing(lon, lat, rings[0])) return false;
  for(var k=1;k<rings.length;k++)
    if(_addrPointInRing(lon, lat, rings[k])) return false; // in a hole
  return true;
}
function _addrFindTract(lon, lat){
  if(!GJ || !GJ.features) return null;
  for(var i=0;i<GJ.features.length;i++){
    var f = GJ.features[i], g = f.geometry; if(!g) continue;
    if(g.type === 'Polygon'){
      if(_addrPointInPoly(lon, lat, g.coordinates)) return f;
    }else if(g.type === 'MultiPolygon'){
      for(var p=0;p<g.coordinates.length;p++)
        if(_addrPointInPoly(lon, lat, g.coordinates[p])) return f;
    }
  }
  return null;
}

// Given a Nominatim row, drop the marker, fly the map, find the containing
// tract, and open the tract popup. Used by both the Go button and the
// autocomplete dropdown so we don't double-geocode.
function _addrApplyHit(hit, fallbackLabel){
  var lat = parseFloat(hit.lat), lon = parseFloat(hit.lon);
  if(!isFinite(lat) || !isFinite(lon)){_addrSetStatus('Could not parse location.', true); return;}
  var inBbox = (lat>=ADDR_BBOX.minLat && lat<=ADDR_BBOX.maxLat &&
                lon>=ADDR_BBOX.minLon && lon<=ADDR_BBOX.maxLon);
  if(addrMarker) map.removeLayer(addrMarker);
  addrMarker = L.marker([lat, lon], {title: hit.display_name || fallbackLabel || ''}).addTo(map);
  map.flyTo([lat, lon], inBbox ? 14 : 11, {duration:0.6});
  var feat = _addrFindTract(lon, lat);
  var dn = hit.display_name || fallbackLabel || '';
  var dnShort = dn.length>80 ? dn.slice(0,77)+'\\u2026' : dn;
  // Nominatim is backed by OpenStreetMap, which anyone can edit: display_name is
  // untrusted and this string reaches innerHTML via _addrSetStatus.
  var html = '<span class="addr-found">'+_addrEsc(dnShort)+'</span>';
  if(feat){
    var gid = feat.properties.GEOID;
    html += '<span class="addr-tract">Census tract '+gid.slice(-6)+'</span>';
    var moreBtn = '<button class="popup-more-btn" onclick="map.closePopup();openDetailDrawer(\\''+gid+'\\')">Show more info \\u25b8</button>';
    if(tLayer){
      _addrSetStatus(html, false);
      tLayer.eachLayer(function(lyr){
        if(lyr.feature && lyr.feature.properties.GEOID === gid){
          var content = lyr.getPopup() ? lyr.getPopup().getContent() : null;
          if(content){
            // Insert the more-info button just before the popup's closing </div>
            var enhanced = content.replace(/<\\/div>\\s*$/, '') + moreBtn + '</div>';
            addrMarker.bindPopup(enhanced).openPopup();
          }
          try{
            lyr.setStyle({weight:2.5, color:'#1d4ed8'});
            setTimeout(function(){if(tLayer) tLayer.resetStyle(lyr);}, 2500);
          }catch(_){}
        }
      });
    }else{
      html += '<span class="addr-tract" style="color:#b45309">Click <b>Build Composite Risk Map</b> to see this tract\\'s risk info.</span>';
      _addrSetStatus(html, false);
      // Even without the composite map, let the user open the detail drawer
      // (it will show "build the composite map" hint at the top, plus the
      // mini-map and per-variable percentiles which are computed at load).
      var basic = '<div style="font-family:sans-serif;font-size:13px;line-height:1.6">'+
        '<b>Tract '+gid.slice(-6)+'</b><br>'+moreBtn+'</div>';
      addrMarker.bindPopup(basic).openPopup();
    }
  }else{
    html += '<span class="addr-tract">No San Diego County census tract at this location.</span>';
    _addrSetStatus(html, !inBbox);
  }
  var cb = document.getElementById('addr-clear'); if(cb) cb.style.display = 'block';
}

function searchAddress(){
  var input = document.getElementById('addr-input');
  var btn = document.getElementById('addr-go');
  if(!input || !btn) return;
  // If a suggestion is highlighted, prefer that.
  if(_addrSuggestActive >= 0 && _addrSuggestRows[_addrSuggestActive]){
    _addrHideSuggest();
    _addrApplyHit(_addrSuggestRows[_addrSuggestActive], input.value);
    return;
  }
  var q = (input.value||'').trim();
  if(!q){_addrSetStatus('Enter an address.', true); return;}
  _addrHideSuggest();
  btn.disabled = true; var prev = btn.textContent; btn.textContent = '...';
  _addrSetStatus('Searching\\u2026');
  var vb = [ADDR_BBOX.minLon, ADDR_BBOX.maxLat, ADDR_BBOX.maxLon, ADDR_BBOX.minLat].join(',');
  var url1 = 'https://nominatim.openstreetmap.org/search?format=json&limit=1&countrycodes=us&viewbox='+vb+'&bounded=1&q='+encodeURIComponent(q);
  fetch(url1, {headers:{'Accept':'application/json'}})
    .then(function(r){if(!r.ok) throw new Error('HTTP '+r.status); return r.json();})
    .then(function(rows){
      if(rows && rows.length) return rows;
      return fetch('https://nominatim.openstreetmap.org/search?format=json&limit=1&countrycodes=us&q='+encodeURIComponent(q),
        {headers:{'Accept':'application/json'}}).then(function(r){if(!r.ok) throw new Error('HTTP '+r.status); return r.json();});
    })
    .then(function(rows){
      btn.disabled = false; btn.textContent = prev;
      if(!rows || !rows.length){
        _addrSetStatus('No match found. Try adding a city, e.g. "San Diego".', true); return;
      }
      _addrApplyHit(rows[0], q);
    })
    .catch(function(err){
      btn.disabled = false; btn.textContent = prev;
      _addrSetStatus('Search failed: '+_addrEsc(err && err.message ? err.message : 'network error'), true);
    });
}

/* ---- Autocomplete dropdown (debounced live suggestions) ---- */
var _addrDebounceTimer = null;
var _addrAbort = null;
var _addrSuggestRows = [];
var _addrSuggestActive = -1;
var _addrLastQuery = '';

function _addrEsc(s){
  return String(s||'').replace(/[&<>"']/g, function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

// Split Nominatim's "display_name" into a primary line (street/place) and
// secondary line (city, county, state) for a cleaner two-line suggestion.
function _addrFormatRow(row){
  var dn = String(row.display_name||'');
  var parts = dn.split(',').map(function(s){return s.trim();}).filter(Boolean);
  var primary = parts[0] || dn;
  var secondary = parts.slice(1, 4).join(', ');
  return {primary: primary, secondary: secondary};
}

function _addrHideSuggest(){
  var ul = document.getElementById('addr-suggest'); if(!ul) return;
  ul.classList.remove('open');
  ul.innerHTML = '';
  _addrSuggestRows = [];
  _addrSuggestActive = -1;
}

function _addrRenderSuggest(rows, msg){
  var ul = document.getElementById('addr-suggest'); if(!ul) return;
  ul.innerHTML = '';
  _addrSuggestRows = rows || [];
  _addrSuggestActive = -1;
  if(msg){
    var li = document.createElement('li');
    li.className = 'empty'; li.textContent = msg;
    ul.appendChild(li);
    ul.classList.add('open');
    return;
  }
  if(!rows || !rows.length){_addrHideSuggest(); return;}
  rows.forEach(function(row, i){
    var fmt = _addrFormatRow(row);
    var li = document.createElement('li');
    li.setAttribute('role', 'option');
    li.dataset.idx = String(i);
    var badge = (row._inSD === false) ? ' <span style="font-size:9.5px;color:#b45309;font-weight:600">outside SD County</span>' : '';
    li.innerHTML = '<div class="addr-sg-main">'+_addrEsc(fmt.primary)+badge+'</div>'+
                   (fmt.secondary ? '<div class="addr-sg-sub">'+_addrEsc(fmt.secondary)+'</div>' : '');
    li.addEventListener('mousedown', function(e){
      // mousedown (not click) so we fire before the input's blur hides the list
      e.preventDefault();
      var idx = parseInt(li.dataset.idx, 10);
      var hit = _addrSuggestRows[idx];
      if(!hit) return;
      var inp = document.getElementById('addr-input');
      if(inp) inp.value = fmt.primary + (fmt.secondary ? ', ' + fmt.secondary : '');
      _addrHideSuggest();
      _addrApplyHit(hit, inp ? inp.value : '');
    });
    li.addEventListener('mouseenter', function(){
      _addrSuggestActive = parseInt(li.dataset.idx, 10);
      _addrUpdateActiveStyle();
    });
    ul.appendChild(li);
  });
  ul.classList.add('open');
}

function _addrUpdateActiveStyle(){
  var ul = document.getElementById('addr-suggest'); if(!ul) return;
  var items = ul.querySelectorAll('li');
  for(var i=0;i<items.length;i++){
    if(i === _addrSuggestActive) items[i].classList.add('active');
    else items[i].classList.remove('active');
  }
  if(_addrSuggestActive >= 0 && items[_addrSuggestActive]){
    var el = items[_addrSuggestActive];
    if(el.scrollIntoView) el.scrollIntoView({block:'nearest'});
  }
}

function _addrInBbox(lat, lon){
  return (lat>=ADDR_BBOX.minLat && lat<=ADDR_BBOX.maxLat &&
          lon>=ADDR_BBOX.minLon && lon<=ADDR_BBOX.maxLon);
}

function _addrFetchSuggest(q){
  if(_addrAbort){try{_addrAbort.abort();}catch(_){}}
  var ctl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
  _addrAbort = ctl;
  // viewbox biases ranking toward SD County but does NOT filter (no bounded=1),
  // so partial prefixes like "5998 Alc" still return matches. We then filter
  // client-side to addresses inside the county and fall back to nearby out-of-
  // county hits if none are local.
  var vb = [ADDR_BBOX.minLon, ADDR_BBOX.maxLat, ADDR_BBOX.maxLon, ADDR_BBOX.minLat].join(',');
  var url = 'https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&limit=12&countrycodes=us&viewbox='+vb+'&q='+encodeURIComponent(q);
  var opts = {headers:{'Accept':'application/json'}};
  if(ctl) opts.signal = ctl.signal;
  fetch(url, opts)
    .then(function(r){if(!r.ok) throw new Error('HTTP '+r.status); return r.json();})
    .then(function(rows){
      if(q !== _addrLastQuery) return; // a newer query has fired
      if(!rows || !rows.length){
        _addrRenderSuggest([], 'No matches \\u2014 try a more specific address.');
        return;
      }
      // Partition by SD County bbox.
      var inSD = [], outSD = [];
      for(var i=0;i<rows.length;i++){
        var lat=parseFloat(rows[i].lat), lon=parseFloat(rows[i].lon);
        if(!isFinite(lat)||!isFinite(lon)) continue;
        rows[i]._inSD = _addrInBbox(lat, lon);
        if(rows[i]._inSD) inSD.push(rows[i]); else outSD.push(rows[i]);
      }
      var combined = inSD.concat(outSD).slice(0, 6);
      if(!combined.length){
        _addrRenderSuggest([], 'No matches \\u2014 try a more specific address.');
        return;
      }
      _addrRenderSuggest(combined, null);
    })
    .catch(function(err){
      if(err && err.name === 'AbortError') return;
    });
}

function _addrOnInput(){
  var inp = document.getElementById('addr-input'); if(!inp) return;
  var q = (inp.value||'').trim();
  _addrLastQuery = q;
  if(_addrDebounceTimer){clearTimeout(_addrDebounceTimer); _addrDebounceTimer = null;}
  if(q.length < 4){_addrHideSuggest(); return;}
  _addrDebounceTimer = setTimeout(function(){_addrFetchSuggest(q);}, 350);
}

function _addrOnKey(e){
  var ul = document.getElementById('addr-suggest');
  var open = ul && ul.classList.contains('open') && _addrSuggestRows.length > 0;
  if(e.key === 'ArrowDown'){
    if(!open) return;
    e.preventDefault();
    _addrSuggestActive = (_addrSuggestActive + 1) % _addrSuggestRows.length;
    _addrUpdateActiveStyle();
  }else if(e.key === 'ArrowUp'){
    if(!open) return;
    e.preventDefault();
    _addrSuggestActive = (_addrSuggestActive - 1 + _addrSuggestRows.length) % _addrSuggestRows.length;
    _addrUpdateActiveStyle();
  }else if(e.key === 'Escape'){
    if(open){e.preventDefault(); _addrHideSuggest();}
  }else if(e.key === 'Enter'){
    e.preventDefault();
    searchAddress();
  }
}

(function(){
  var inp = document.getElementById('addr-input');
  if(!inp) return;
  inp.addEventListener('input', _addrOnInput);
  inp.addEventListener('keydown', _addrOnKey);
  inp.addEventListener('focus', function(){
    if(_addrSuggestRows.length > 0){
      var ul = document.getElementById('addr-suggest');
      if(ul) ul.classList.add('open');
    }
  });
  // Hide on outside click
  document.addEventListener('mousedown', function(e){
    var wrap = document.querySelector('#addr-search .addr-input-wrap');
    if(wrap && !wrap.contains(e.target)) _addrHideSuggest();
  });
})();

/* ================================================================
   TRACT DETAIL MODAL
   Single-tract centered modal opened from the address-search popup.
   Shows: popup-style summary cards (composite + pillar scores +
   dominant driver), LST + land-use mini-map, and a curated set of
   key variables per pillar with polarity-aware percentile bars.
   ================================================================ */
// One map in the detail modal with three switchable primary layers:
// satellite / SANDAG land-use / Landsat LST. _dtLayers holds the pre-built
// layer for each view; _dtView is the currently active key.
var _dtMap=null, _dtLayers={sat:null,lu:null,lst:null};
var _dtView='sat', _dtGeoid=null;

// Fallback variables per pillar, shown only when nothing is selected/built yet.
var _DT_CURATED = {
  exposure: ['mean_tmax', 'mean_tmin_overnight', 'p99_wbgt'],
  sensitivity: ['RPL_THEMES', 'EP_AGE65', 'diabetes_prev'],
  adaptive: ['ac_prob', 'pct_green', 'pct_pre1980']
};
// Variables to list for a pillar in the detail modal, driven by what's selected
// (mirrors the radar). Exposure: the selected heat mode's member layers — the
// actual ingredients of the Heat Exposure score. Sensitivity/Adaptive: the
// enabled variables, ordered by PCA loading (contribution to the composite);
// pre-build, enabled vars in panel order; nothing enabled -> curated fallback.
function dtPillarKeys(pid){
  if(pid==='exposure'){
    var mode=heatModeDef(_builtSettings().heatMode);
    return (mode&&mode.layers)?mode.layers.slice():(_DT_CURATED.exposure||[]);
  }
  var R=results&&results[pid];
  if(R&&R.keys&&R.keys.length)return radarImportance(R);
  // A pillar left out of the build has no score, so listing variables under it
  // would suggest they're behind one.
  if(results)return [];
  var ks=_panelSettings().on[pid];
  return ks.length?ks:(_DT_CURATED[pid]||[]);
}

function openDetailDrawer(geoid){
  if(!geoid) return;
  var idx=null;
  if(window._gIdx && window._gIdx[geoid]!=null) idx=window._gIdx[geoid];
  else{
    for(var i=0;i<GJ.features.length;i++){
      if(GJ.features[i].properties.GEOID===geoid){idx=i;break;}
    }
  }
  if(idx==null||!GJ.features[idx]) return;
  var feat=GJ.features[idx], pr=feat.properties;
  _dtGeoid=geoid;

  document.getElementById('dt-title').textContent='Tract '+geoid.slice(-6);
  document.getElementById('dt-sub').textContent='GEOID '+geoid;
  var ctx='Present Day';
  if(curScenario!=='present'){
    var sl={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'};
    ctx=(sl[curScenario]||curScenario)+' \\u2022 '+(curPeriod==='2050'?'Mid-Century 2050':'End-of-Century 2080');
  }
  document.getElementById('dt-ctx').textContent=ctx;

  var html='';
  // ---- Summary section (popup-style: composite + pillar score cards + driver)
  html+='<div class="dt-section">';
  if(pr._comp!=null){
    // Every card is risk-oriented (higher = higher risk), matching the popup
    // radar. _adaptive is stored capacity-side (higher = MORE capacity), so it
    // is flipped to a capacity deficit here exactly as the radar's adaptive
    // wedge and the composite (100-s in compute()) already do.
    // Accent bars carry the PILLAR colour, so each card matches its pillar
    // header below and its rose-plot wedge. (Keep in step with the tract PDF.)
    var cards=[
      {lbl:'Composite', val:pr._comp, col:_DT_CARD_ACCENT},
      {lbl:'Heat Exp.', val:pr._exposure, col:PMETA.exposure.color},
      {lbl:'Sensitivity', val:pr._sensitivity, col:PMETA.sensitivity.color},
      {lbl:'Adaptive Cap.', val:(pr._adaptive==null?null:100-pr._adaptive), col:PMETA.adaptive.color}
    ];
    html+='<div class="dt-section-hdr">Risk Summary</div>';
    html+='<div class="dt-summary">';
    cards.forEach(function(c){
      if(c.val==null){
        html+='<div class="dt-card"><div class="dt-card-lbl">'+c.lbl+'</div><div class="dt-card-val dt-na">\\u2014</div></div>';
      }else{
        var col=c.col;
        html+='<div class="dt-card" style="border-left:3px solid '+col+'">'+
          '<div class="dt-card-lbl">'+c.lbl+'</div>'+
          '<div class="dt-card-val">'+c.val.toFixed(0)+'</div>'+
          '<div class="dt-card-pct">/ 100</div></div>';
      }
    });
    html+='</div>';
    html+='<div class="dt-note">County percentile \\u2014 higher score = higher risk (Adaptive Capacity is scored as a capacity gap, so higher = less capacity).</div>';
    if(pr._drv){
      // PMETA label, not the raw pillar key — the Dominant Driver legend prints
      // "Heat Exposure"/"Population Sensitivity"/"Adaptive Capacity", and this
      // card used to contradict it with the internal key ("exposure").
      var drvLbl=(PMETA[pr._drv]&&PMETA[pr._drv].label)||pr._drv;
      html+='<div class="dt-driver">Dominant driver: <b>'+_addrEsc(drvLbl)+'</b></div>';
    }
  }else{
    html+='<div class="dt-hint">Click <b>Build Composite Risk Map</b> in the left panel to populate composite and pillar scores. The map and variable rows below work without it.</div>';
  }
  html+='</div>';

  // ---- Single map with switchable layer (satellite / land use / LST)
  html+='<div class="dt-section">'+
    '<div class="dt-section-hdr">Imagery, Land Use &amp; Heat</div>'+
    '<div class="dt-view-switch" role="tablist">'+
      '<button class="dt-view-btn active" data-view="sat" onclick="_dtSetView(\\'sat\\')">Satellite</button>'+
      '<button class="dt-view-btn" data-view="lu" onclick="_dtSetView(\\'lu\\')">Land Use</button>'+
      '<button class="dt-view-btn" data-view="lst" onclick="_dtSetView(\\'lst\\')">Landsat LST</button>'+
    '</div>'+
    '<div class="dt-view-sub" id="dt-view-sub">Esri World Imagery</div>'+
    '<div class="dt-map-frame"><div class="dt-map" id="dt-map"></div></div>'+
    '<div id="dt-lu-legend" class="dt-lu-legend" style="display:none"></div>'+
    '<div id="dt-lst-legend" class="dt-lst-legend" style="display:none"></div>'+
  '</div>';

  // ---- Selected-variable rows per pillar (driven by the current selection)
  var rowsNoted=false;
  ['exposure','sensitivity','adaptive'].forEach(function(pid){
    var pm=PMETA[pid];
    var keys=dtPillarKeys(pid);
    var hdr=pm.label;
    if(pid==='exposure'){var md=heatModeDef(_builtSettings().heatMode);if(md)hdr+=' \\u2014 '+md.label;}
    var rows='';
    keys.forEach(function(key){
      var meta=null;
      for(var i=0;i<VM.length;i++){if(VM[i].key===key){meta=VM[i];break;}}
      if(!meta) return;
      var rawV=pr[key];
      if(rawV==null||(typeof rawV==='number'&&isNaN(rawV))){
        rows+=_dtRowNA(meta.label);return;
      }
      var dArr=getDeltaArr(key);
      var disp=dArr?getAdj(key,idx):+rawV;
      var dispStr=fmtCmpRaw(key,disp);
      var p=(rawPct[key]&&rawPct[key][idx]!=null)?rawPct[key][idx]:null;
      var higherIsWorse=(POLARITY&&POLARITY[key]!==false);
      rows+=_dtRow(meta.label, dispStr, p, higherIsWorse);
    });
    if(rows){
      html+='<div class="dt-section">'+
        (rowsNoted?'':_dtRiskLegend()+'<div class="dt-note dt-rows-note">Bars show each tract\\'s risk percentile among San Diego County tracts, not the value itself. Where higher values = less risk, a low <em>value</em> translates to high percentile <em>risk</em>.</div>')+
        '<div class="dt-pillar-hdr" style="color:'+pm.color+'">'+hdr+'</div>'+
        rows+'</div>';
      rowsNoted=true;
    }
  });

  document.getElementById('dt-body').innerHTML=html;
  document.getElementById('detail-modal').classList.add('open');
  // Two-stage init so each map container has its final size before Leaflet
  // measures it, and so the protomaps tile layer attaches after the size
  // stabilizes (otherwise parcels can silently fail to render).
  setTimeout(function(){
    _dtBuildMaps(feat);
    _dtRenderLstLegend();
    _dtRenderLuLegend();
  }, 60);
}

function closeDetailDrawer(){
  var dr=document.getElementById('detail-modal');
  if(!dr) return;
  dr.classList.remove('open');
  if(_dtMap){try{_dtMap.remove();}catch(_){}_dtMap=null;}
  _dtLayers={sat:null,lu:null,lst:null};
  _dtView='sat';_dtGeoid=null;
}

// One variable row. The bar length, its color and the printed percentile are
// all the RISK percentile, so this reconciles exactly with the popup radar,
// which plots protective variables (POLARITY===false) as deficits. The value
// on the right stays the real-world reading, so a high-canopy tract reads
// "85.6 / 0th percentile risk".
// The polarity note sits beside the VALUE it describes, not the percentile:
// "86th percentile risk · higher value = less risk" was read as "86 is high,
// high is good" (City of SD feedback, 2026-10). No "lower than X% of tracts"
// gloss either — pctRank midranks ties, so it would be false for zero-heavy
// variables (ag workers: 522 of 735 tracts at 0%).
function _dtRow(label, dispVal, pct, higherIsWorse){
  if(pct==null){
    return '<div class="dt-var-row">'+
      '<div class="dt-var-top"><span class="dt-var-label">'+_addrEsc(label)+'</span>'+
      '<span class="dt-var-val">'+_addrEsc(dispVal)+'</span></div></div>';
  }
  var p=Math.max(0,Math.min(100,pct));
  var riskPct = higherIsWorse ? p : (100 - p);
  var color = _dtRiskColor(riskPct);
  var width = Math.max(2, riskPct).toFixed(1);   // toFixed: 100-p can land on 10.299999999999997
  var dirNote = higherIsWorse ? 'higher = more risk' : 'higher = less risk';
  return '<div class="dt-var-row">'+
    '<div class="dt-var-top"><span class="dt-var-label">'+_addrEsc(label)+'</span>'+
    '<span class="dt-var-val">'+_addrEsc(dispVal)+'<span class="dt-var-dir">\\u00b7 '+dirNote+'</span></span></div>'+
    '<div class="dt-var-bar-wrap"><div class="dt-var-bar" style="width:'+width+'%;background:'+color+'"></div></div>'+
    '<div class="dt-var-pct">'+_ord(riskPct)+' percentile risk</div>'+
  '</div>';
}
function _dtRowNA(label){
  return '<div class="dt-var-row">'+
    '<div class="dt-var-top"><span class="dt-var-label">'+_addrEsc(label)+'</span>'+
    '<span class="dt-var-val dt-na">no data</span></div></div>';
}
// Risk-aligned percentile (0=safe, 100=high risk) -> color via green->yellow->red.
function _dtRiskColor(rp){
  rp = Math.max(0, Math.min(100, rp));
  var stops=[[0,[34,160,80]],[50,[240,180,20]],[100,[220,60,40]]];
  function lerp(a,b,t){return a+(b-a)*t;}
  for(var i=1;i<stops.length;i++){
    if(rp<=stops[i][0]){
      var t=(rp-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
      var c0=stops[i-1][1], c1=stops[i][1];
      return 'rgb('+Math.round(lerp(c0[0],c1[0],t))+','+Math.round(lerp(c0[1],c1[1],t))+','+Math.round(lerp(c0[2],c1[2],t))+')';
    }
  }
  return 'rgb(220,60,40)';
}
// Colour key for the variable-row bars. Sampled from _dtRiskColor itself so
// the legend can't drift from the bars it explains.
function _dtRiskLegend(){
  var stops=[0,25,50,75,100].map(_dtRiskColor).join(',');
  return '<div class="dt-risk-legend">'+
    '<div style="font-weight:600;color:#444;margin-bottom:2px">Percentile risk</div>'+
    '<div class="dt-lst-bar" style="background:linear-gradient(to right,'+stops+')"></div>'+
    '<div class="dt-lst-ticks"><span>0 \\u00b7 lower risk</span><span>50</span><span>100 \\u00b7 higher risk</span></div>'+
  '</div>';
}

// Build the single detail-modal map. We always start in 'sat' view; the
// other view layers (land-use protomaps, LST imageOverlay) are created lazily
// when the user clicks their toggle button. The tract outline and the
// 'landusePane' are set up up-front so they're ready for any view.
function _dtBuildMaps(feat){
  if(_dtMap){try{_dtMap.remove();}catch(_){}}
  _dtMap=null;_dtLayers={sat:null,lu:null,lst:null};_dtView='sat';
  var m=L.map('dt-map',{zoomControl:true,attributionControl:false,
    scrollWheelZoom:true,dragging:true});
  // Pre-create landusePane so the protomaps layer can attach cleanly later.
  if(!m.getPane('landusePane')){
    m.createPane('landusePane');
    m.getPane('landusePane').style.zIndex=450;
  }
  // Initial view is satellite — add the satellite tile layer now.
  _dtLayers.sat=L.tileLayer(BASEMAPS.satellite.url,BASEMAPS.satellite.opts).addTo(m);
  // Tract outline sits on top of every view.
  var poly=L.geoJSON(feat,{style:{color:'#fff',weight:2.5,fill:false,opacity:0.95}}).addTo(m);
  m.fitBounds(poly.getBounds(),{padding:[18,18]});
  m.on('click', _dtLanduseClickHandler);
  _dtMap=m;
  _dtUpdateViewUi();
}

// Switch the visible primary layer. Each layer is built on first use and
// cached on _dtLayers; the previous layer is removed (not destroyed) so
// switching back is instant.
function _dtSetView(view){
  if(!_dtMap||view===_dtView) return;
  var m=_dtMap;
  // Remove the current primary layer.
  var cur=_dtLayers[_dtView];
  if(cur){try{m.removeLayer(cur);}catch(_){}}
  // Build the target layer if it doesn't exist yet.
  if(!_dtLayers[view]){
    if(view==='sat'){
      _dtLayers.sat=L.tileLayer(BASEMAPS.satellite.url,BASEMAPS.satellite.opts);
    }else if(view==='lu'){
      // Light basemap + SANDAG parcels via protomaps-leaflet. We bundle them
      // into a LayerGroup so we can add/remove with a single handle.
      var group=L.layerGroup(lightTiles());
      if(window.protomapsL){
        var luLayer=_buildLanduseLayer();
        if(luLayer){group.addLayer(luLayer); group._luLayer=luLayer;}
        else{console.warn('detail-modal: land-use layer construction failed');}
      }else{
        console.warn('detail-modal: protomaps-leaflet not loaded');
      }
      // Switch tract-outline color to dark for visibility over the light base.
      group._outlineColor='#111';
      _dtLayers.lu=group;
    }else if(view==='lst'){
      var grp=L.layerGroup(lightTiles());
      var lv=lstVariant();
      if(lv){
        var b=LST.bounds;
        grp.addLayer(L.imageOverlay(lv.url,[[b.south,b.west],[b.north,b.east]],{opacity:0.85}));
      }
      grp._outlineColor='#111';
      _dtLayers.lst=grp;
    }
  }
  var next=_dtLayers[view];
  if(next){
    next.addTo(m);
    // bringToBack so the tract outline stays visible above tiles/parcels.
    if(next.eachLayer){
      next.eachLayer(function(l){if(l.bringToBack)l.bringToBack();});
    }else if(next.bringToBack){
      next.bringToBack();
    }
    // Adjust tract-outline color for legibility against the new base.
    var outlineColor=(view==='sat')?'#fff':'#111';
    m.eachLayer(function(l){
      if(l instanceof L.GeoJSON){
        try{l.setStyle({color:outlineColor});}catch(_){}
      }
    });
  }
  _dtView=view;
  _dtUpdateViewUi();
}

// Update the active toggle button, subtitle, and per-view legend visibility.
function _dtUpdateViewUi(){
  var subText={sat:'Esri World Imagery',
               lu:'SANDAG parcels — click a parcel for its land-use category',
               lst:'Landsat 100 m annual mean surface temperature'};
  var sub=document.getElementById('dt-view-sub');
  if(sub) sub.textContent=subText[_dtView]||'';
  var btns=document.querySelectorAll('.dt-view-btn');
  for(var i=0;i<btns.length;i++){
    if(btns[i].getAttribute('data-view')===_dtView) btns[i].classList.add('active');
    else btns[i].classList.remove('active');
  }
  var lu=document.getElementById('dt-lu-legend');
  if(lu) lu.style.display=(_dtView==='lu')?'flex':'none';
  var lst=document.getElementById('dt-lst-legend');
  if(lst) lst.style.display=(_dtView==='lst')?'flex':'none';
}

function _dtLanduseClickHandler(e){
  // Only fire when the land-use view is active.
  if(_dtView!=='lu') return;
  var grp=_dtLayers.lu;
  var luLayer=grp&&grp._luLayer;
  if(!luLayer||typeof luLayer.queryTileFeaturesDebug!=='function') return;
  var hits=null;
  try{hits=luLayer.queryTileFeaturesDebug(e.latlng.lng,e.latlng.lat,16);}catch(err){return;}
  if(!hits||!hits.size) return;
  var feats=null;
  hits.forEach(function(arr){if(!feats&&arr&&arr.length)feats=arr;});
  if(!feats||!feats.length) return;
  var p=(feats[0].feature&&feats[0].feature.props)||{};
  var cat=_luCategoryName(p);
  var label=p.Landuse||cat;
  var lu=p.lu||'';
  L.popup({offset:[0,-4]}).setLatLng(e.latlng)
    .setContent('<div style="font-size:11px;line-height:1.35"><b>'+label+'</b><br><span style="color:#888">'+cat+(lu?' &middot; LU '+lu:'')+'</span></div>')
    .openOn(_dtMap);
}
function _dtRenderLstLegend(){
  var el=document.getElementById('dt-lst-legend');
  var lv=lstVariant();
  if(!el||!lv){if(el)el.innerHTML='';return;}
  var stops=lv.legend.map(function(s){return"rgb("+s.rgb.join(",")+")"}).join(",");
  var ticks='';
  lv.legend.forEach(function(s){ticks+='<span>'+s.value+'&deg;</span>';});
  el.innerHTML='<div style="font-weight:600;color:#444;margin-bottom:2px">Landsat LST 100m (annual mean &deg;C)</div>'+
    '<div class="dt-lst-bar" style="background:linear-gradient(to right,'+stops+')"></div>'+
    '<div class="dt-lst-ticks">'+ticks+'</div>';
}
function _dtRenderLuLegend(){
  var el=document.getElementById('dt-lu-legend');
  if(!el) return;
  var cats=LU_CATS.concat([LU_WATER_CAT]);
  el.innerHTML=cats.map(function(c){
    return '<div class="dt-lu-item"><div class="dt-lu-sw" style="background:'+c.color+'"></div>'+_addrEsc(c.name)+'</div>';
  }).join('');
}

// Close drawer on Esc
document.addEventListener('keydown', function(e){
  if(e.key==='Escape'){
    var dr=document.getElementById('detail-modal');
    if(dr&&dr.classList.contains('open')) closeDetailDrawer();
  }
});

// City jurisdiction boundary overlays — one layer per city, all rendered in black
var jurisLayers={};
var jurisLabels={};   // name -> permanent center label tooltip, shown with the layer
var JURIS_COLOR="#000000";
function jurisCheckboxId(name){return "cb-juris-"+name.replace(/\\s+/g,"_");}
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
        focusCity(name);
      });
      lyr.bindTooltip(f.properties.NAME+" \\u2014 click to zoom in",{sticky:true,className:"juris-tip"});
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
    // Permanent center label so the city name is visible without hovering.
    jurisLabels[name]=L.tooltip({permanent:true,direction:"center",className:"juris-label",interactive:false})
      .setContent(name)
      .setLatLng(jurisLayers[name].getBounds().getCenter());
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
  if(cb&&cb.checked){lyr.addTo(map);lyr.bringToFront();if(jurisLabels[name])jurisLabels[name].addTo(map);}
  else{map.removeLayer(lyr);if(jurisLabels[name])map.removeLayer(jurisLabels[name]);if(focusedCity===name)exitCityView();}
  syncJurisAllCheckbox();
}
function toggleJurisAll(on){
  if(!on&&focusedCity)exitCityView();
  Object.keys(jurisLayers).forEach(function(name){
    var cb=document.getElementById(jurisCheckboxId(name));
    if(cb)cb.checked=!!on;
    var lyr=jurisLayers[name];
    if(!lyr)return;
    if(on){lyr.addTo(map);lyr.bringToFront();if(jurisLabels[name])jurisLabels[name].addTo(map);}
    else{map.removeLayer(lyr);if(jurisLabels[name])map.removeLayer(jurisLabels[name]);}
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

// ── City focus: click a city to zoom in and explore its tracts ──────────
// While focused, the focused city's polygon fill is made non-interactive so
// clicks/hovers fall through to the census tracts underneath; other city
// polygons stay clickable so you can hop straight to another city.
var focusedCity=null;
function _setCityFillClickable(name,on){
  var lyr=jurisLayers[name];
  if(!lyr)return;
  lyr.eachLayer(function(l){
    if(l&&l._path)l._path.style.pointerEvents=on?"":"none";
  });
}
function focusCity(name){
  var lyr=jurisLayers[name];
  if(!lyr)return;
  // Make sure the city's boundary + label are actually on the map.
  var cb=document.getElementById(jurisCheckboxId(name));
  if(cb&&!cb.checked){cb.checked=true;toggleJurisLayer(name);}
  // Restore the previously focused city's interior, mute the new one's.
  if(focusedCity&&focusedCity!==name)_setCityFillClickable(focusedCity,true);
  focusedCity=name;
  _setCityFillClickable(name,false);
  try{map.fitBounds(lyr.getBounds(),{padding:[24,24]});}catch(e){}
  var bar=document.getElementById("city-focus-bar");
  document.getElementById("cf-name").textContent=name;
  if(bar)bar.classList.add("show");
}
function exitCityView(){
  if(focusedCity){_setCityFillClickable(focusedCity,true);focusedCity=null;}
  var bar=document.getElementById("city-focus-bar");
  if(bar)bar.classList.remove("show");
  if(tLayer&&tLayer.getBounds){
    try{map.fitBounds(tLayer.getBounds(),{padding:[10,10]});}catch(e){}
  }
}

// Tribal lands overlays — Census TIGER AIANNH 2024 (San Diego County)
// Reservations and off-reservation trust lands are paired under one tribe entry;
// the map distinguishes them with a solid (reservation) vs dashed (trust) outline.
var tribalLayers={};      // keyed by full feature name (each polygon)
var tribalLabels={};      // name -> permanent center label, shown with the layer
var tribalGroups={};      // tribeKey -> [feature names]
var TRIBAL_COLORS={reservation:"#8b5a2b",trust:"#b78050"};
function tribalGroupKey(name){
  return name.replace(/ Reservation$/,"").replace(/ Off-Reservation Trust Land$/,"");
}
function tribalCheckboxId(groupKey){return "cb-tribal-"+groupKey.replace(/\\s+/g,"_");}
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
    // Permanent center label so the tribe name is visible without hovering.
    var gk0=tribalGroupKey(name);
    var labelTxt=gk0+((f.properties.TYPE==='trust')?" (Trust Land)":"");
    tribalLabels[name]=L.tooltip({permanent:true,direction:"center",className:"juris-label tribal-label",interactive:false})
      .setContent(labelTxt)
      .setLatLng(tribalLayers[name].getBounds().getCenter());
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
    if(on){lyr.addTo(map);lyr.bringToFront();if(tribalLabels[name])tribalLabels[name].addTo(map);}
    else{map.removeLayer(lyr);if(tribalLabels[name])map.removeLayer(tribalLabels[name]);}
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
      if(on){lyr.addTo(map);lyr.bringToFront();if(tribalLabels[name])tribalLabels[name].addTo(map);}
      else{map.removeLayer(lyr);if(tribalLabels[name])map.removeLayer(tribalLabels[name]);}
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

// Unincorporated county boundary — SANGIS JUR_MUNICIPAL "S.D. COUNTY" (code CN),
// dissolved to one region (~85% of the county). Drawn non-interactive so clicks
// fall through to the tract underneath; it only provides visual context.
var unincorpLayer=null;
var UNINCORP_COLOR="#4f46e5";
if(typeof UNINCORP!=='undefined'&&UNINCORP&&UNINCORP.features&&UNINCORP.features.length){
  unincorpLayer=L.geoJSON(UNINCORP,{
    interactive:false,
    style:function(){return{color:UNINCORP_COLOR,weight:2,fillColor:UNINCORP_COLOR,fillOpacity:0.08};}
  });
  document.getElementById("overlay-toggles").style.display="block";
}
function toggleUnincorp(on){
  if(!unincorpLayer)return;
  if(on){
    unincorpLayer.addTo(map);unincorpLayer.bringToFront();
    // Keep city and tribal outlines above the unincorporated region.
    Object.keys(jurisLayers).forEach(function(k){if(map.hasLayer(jurisLayers[k]))jurisLayers[k].bringToFront();});
    if(typeof tribalLayers!=='undefined')Object.keys(tribalLayers).forEach(function(k){if(map.hasLayer(tribalLayers[k]))tribalLayers[k].bringToFront();});
  } else {
    map.removeLayer(unincorpLayer);
  }
}

/* ================================================================
   Cool Zones — point overlay of sites designated by the County Cool
   Zones program. s.src carries provenance so a marker is attributed
   to the list it came from rather than to a hardcoded label.
   ================================================================ */
var COOLZONE_SRC={county:"SD County Cool Zones"};
var coolZonesLayer=null;
if(COOLZONES&&COOLZONES.length){
  coolZonesLayer=L.layerGroup();
  COOLZONES.forEach(function(s){
    var m=L.circleMarker([s.lat,s.lon],{
      radius:5, color:"#ffffff", weight:1.5,
      fillColor:"#111111", fillOpacity:0.95
    });
    var addr=s.address+(s.city?", "+s.city:"")+(s.zip?" "+s.zip:"");
    var phone=s.phone?'<div style="color:#444;margin-top:2px">'+_addrEsc(s.phone)+'</div>':'';
    var src=COOLZONE_SRC[s.src]||"SD County Cool Zones";
    m.bindPopup(
      '<div style="font-size:12px;line-height:1.35;max-width:240px">'+
      '<div style="font-weight:600;margin-bottom:3px">'+_addrEsc(s.name)+'</div>'+
      '<div>'+_addrEsc(addr)+'</div>'+phone+
      '<div style="margin-top:4px;font-size:10px;color:#888">Source: '+src+'</div>'+
      '</div>',
      {closeButton:true, autoPan:true, maxWidth:260}
    );
    m.on("click",function(e){if(L.DomEvent)L.DomEvent.stopPropagation(e);});
    coolZonesLayer.addLayer(m);
  });
  document.getElementById("overlay-toggles").style.display="block";
}
function toggleCoolZones(on){
  if(!coolZonesLayer)return;
  if(on){
    coolZonesLayer.addTo(map);
    coolZonesLayer.eachLayer(function(l){if(l.bringToFront)l.bringToFront();});
  } else {
    map.removeLayer(coolZonesLayer);
  }
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
  // a P=1 selection of e.g. pct_renter would skip alignment and end up with the
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
  // Orient EVERY component to the pillar's risk reference before it is used.
  // Eigenvector signs are arbitrary, so an unanchored PC used to enter the
  // weighted composite below with inverted polarity and cancel risk instead of
  // adding it — which also made the pillar score depend on which eigen solver
  // ran (numpy and this Jacobi routine disagreed at r=0.33 on the default
  // sensitivity selection, moving 32% of tracts to a different dominant
  // driver). Loadings flip with their scores so the "Driven by" tooltip and the
  // Primary Spatial Patterns layers stay in sync.
  for(k=0;k<P;k++){
    var pcCol=[];for(i=0;i<N;i++)pcCol.push(sc[i][k]);
    if(corrXY(rc,pcCol)<0){
      for(i=0;i<N;i++)sc[i][k]=-sc[i][k];
      for(j=0;j<P;j++)eig.V[j][ord[k]]=-eig.V[j][ord[k]];
    }
  }
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

// Exposure pillar score for the currently-selected heat mode. NOT PCA:
// the mean of the mode's z-scored member layers (climate deltas applied),
// percentile-ranked. Returns a result shaped like pillarPCA's (scores + empty
// pcs) so the rest of the pipeline is unchanged. Returns null if no data.
function exposureScore(modeKey){
  var mode=heatModeDef(modeKey);if(!mode)return null;
  var feats=GJ.features,N=feats.length,i,j;
  var keys=mode.layers.filter(function(k){
    for(var t=0;t<N;t++){if(feats[t].properties[k]!=null)return true;}return false;});
  if(!keys.length)return null;
  var acc=new Array(N);for(i=0;i<N;i++)acc[i]=0;
  for(j=0;j<keys.length;j++){
    var d=getDeltaArr(keys[j]),x=new Array(N),mu=0;
    for(i=0;i<N;i++){var v=feats[i].properties[keys[j]];if(v==null)v=0;if(d)v+=d[i];x[i]=v;mu+=v;}
    mu/=N;var sd=0;for(i=0;i<N;i++)sd+=(x[i]-mu)*(x[i]-mu);
    sd=Math.sqrt(sd/(N-1));if(sd<1e-10)sd=1;
    for(i=0;i<N;i++)acc[i]+=(x[i]-mu)/sd;
  }
  for(i=0;i<N;i++)acc[i]/=keys.length;
  return{scores:pctRank(acc),pcs:[],nret:0,totvar:null,keys:keys,
         mode:mode.key,label:mode.label};
}

/* ================================================================
   COMPUTE
   ================================================================ */
// Per-tract fields a build writes for each pillar: the score and its PCs.
var _PILLAR_FIELD=/^_(exposure|sensitivity|adaptive)(_pc\\d+)?$/;
// Build: score the panel's selection and commit it as the composite. Given
// `settings` (a scenario switch), re-score the committed build under the new
// scenario instead: pending panel changes stay pending and the view is kept.
function compute(settings){
  var rescore=!!settings;
  try{
    document.getElementById("prompt").style.display="none";
    document.getElementById("error-bar").style.display="none";
    settings=settings||_panelSettings();
    if(!rescore){
      rawVar=null; // switch from raw preview to results mode
      // Building commits the heat-mode preview, so surface the thing they asked
      // for: the composite. Any other built layer (driver, a PC) is left alone —
      // it gets rebuilt in place rather than yanking the user back to composite.
      if(curLayer==="heatmode")curLayer="composite";
    }
    // Exposure = selected heat mode (no PCA); Sensitivity/Adaptive = PCA.
    var res={exposure:exposureScore(settings.heatMode)};
    ["sensitivity","adaptive"].forEach(function(pid){res[pid]=pillarPCA(pid,settings.on[pid]);});
    // write to features
    var N=GJ.features.length;
    for(var i=0;i<N;i++){
      var pr=GJ.features[i].properties,sum=0,cnt=0;
      // Drop the previous build's pillar fields first. A pillar left out of this
      // build would otherwise keep its old score, and the radar, detail drawer,
      // compare views and tract PDF would go on showing it.
      for(var fld in pr)if(_PILLAR_FIELD.test(fld))delete pr[fld];
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
      // Per-pillar risk contributions (exposure, sensitivity, adaptive deficit)
      // kept for the trivariate Driver Mix layer's barycentric blend.
      pr._drvE=eS;pr._drvS=sS;pr._drvA=aD;
    }
    res.settings=settings;
    results=res;
    if(!rescore)compositeStale=false;   // the panel and the map now agree again
    syncBuildBtn();
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
//    pillar set uses red/purple/green together for the dominant-driver layer,
//    matching each pillar's choropleth so the swatch reads as the same hue.)
var CR=[[255,255,178],[254,204,92],[253,141,60],[240,59,32],[189,0,38]];
var CE=[[254,229,217],[252,174,145],[251,106,74],[203,24,29]];
var CS=[[242,240,247],[203,201,226],[158,154,200],[106,81,163]];
var CA=[[229,245,224],[161,217,155],[65,171,93],[0,109,44]];
var CD=[[255,255,204],[255,237,160],[254,178,76],[253,141,60],[252,78,42],[227,26,28],[177,0,38]];
var DRGB={exposure:[192,57,43],sensitivity:[106,81,163],adaptive:[22,160,82]};

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
  // Live preview of the selected heat mode. Reads expModeScores(), NOT pr._exposure
  // — that still belongs to the last build and may be a different mode entirely.
  if(curLayer==="heatmode"){
    var msf=expModeScores(), harr=msf?msf[heatMode]:null;
    var hv=harr?harr[fi]:null;
    if(hv==null)return"#ccc";
    return toHex(interpC(pcc.exposure||cr,(hv-2)/96));}
  if(curLayer==="composite"){if(pr._comp==null)return"#ccc";return toHex(interpC(cr,(pr._comp-2)/96));}
  if(curLayer==="exposure"){var ev=pr._exposure;if(ev==null)return"#ccc";
    return toHex(interpC(pcc.exposure||cr,(ev-2)/96));}
  if(curLayer==="driver"){var b=drgb[pr._drv]||[128,128,128];
    var intn=0.15+0.85*(pr._drvI||0.5);
    return toHex(b.map(function(c){return Math.round(255-intn*(255-c))}));}
  if(curLayer==="drivermix"){
    // Trivariate blend: white where the three drivers are balanced, each pillar
    // hue toward its corner (see _triMix), then whitened by composite intensity
    // (lighter = lower overall risk, matching the Dominant Driver layer).
    var base=_triMix(pr._drvE||0,pr._drvS||0,pr._drvA||0,drgb);
    var intnM=0.15+0.85*(pr._drvI||0.5);
    return toHex(base.map(function(c){return Math.round(255-intnM*(255-c))}));}
  var m=curLayer.match(/^(exposure|sensitivity|adaptive)_pc(\\d+)$/);
  if(m){var v=pr["_"+m[1]+"_pc"+m[2]];if(v==null)return"#ccc";
    return toHex(interpC(pcc[m[1]]||cr,(v-2)/96));}
  return"#ccc";
}

/* ================================================================
   TRACT POPUP RADAR
   ================================================================ */
// Sectored heat-vulnerability radar for the tract click popup. The circle is
// split into three 120-degree wedges, one per pillar (tinted by PMETA color).
// Each wedge holds one big "main" dot — the pillar score — plus four smaller
// variable dots. Everything is a county percentile (0-100) from rawPct, and
// adaptive-capacity values are plotted as DEFICITS (100 - percentile) so the
// shape grows outward with risk in every wedge. Each variable label shows its
// name plus a value the popup's Percentile/Value toggle switches between the
// county percentile rank and the real-world reading; dots also carry a <title>.
var radarValMode='pct';   // 'pct' = percentile rank, 'abs' = real-world value
// Short spoke labels for the sensitivity/adaptive radar (full names live in
// varLabel[]). Kept concise so dynamically-chosen spokes stay legible.
var RADAR_SHORT={
  bphigh_prev:'High BP', casthma_prev:'Asthma', chd_prev:'Heart Dis.',
  cognition_prev:'Cognitive', copd_prev:'COPD', depression_prev:'Depression',
  diabetes_prev:'Diabetes', stroke_prev:'Stroke', obesity_prev:'Obesity',
  mhlth_prev:'Mental Dist.', leb_pctile:'Life Exp.', loneliness_prev:'Loneliness',
  emotionspt_prev:'No Support', heat_hosp_day:'Heat Hosp.', heat_hosp_night:'Night Hosp.',
  EP_AGE65:'Seniors', pct_senior_alone:'Sr. Alone', EP_AGE17:'Children',
  EP_DISABL:'Disability', EP_LIMENG:'Ltd English', EP_SNGPNT:'Single Par.',
  EP_POV150:'Poverty', EP_UNEMP:'Unemployed', EP_HBURD:'Housing Cost',
  EP_NOHSDP:'No HS Dip.', EP_UNINSUR:'Uninsured', EP_MINRTY:'Minority',
  EP_MUNIT:'Multi-Unit', EP_MOBILE:'Mobile Home', EP_CROWD:'Crowding',
  EP_NOVEH:'No Vehicle', EP_GROUPQ:'Group Qtrs', EP_NOINT:'No Internet',
  pct_outdoor_workers:'Outdoor Wk', pct_ag_workers:'Ag Workers',
  foodinsecu_prev:'Food Insec.', foodstamp_prev:'SNAP', housinsecu_prev:'Housing Ins.',
  lacktrpt_prev:'No Transit', shututility_prev:'Util. Shutoff', energy_burden:'Energy Burd.',
  RPL_THEMES:'SVI', RPL_THEME1:'SVI: SES', RPL_THEME2:'SVI: HH',
  RPL_THEME3:'SVI: Race', RPL_THEME4:'SVI: Hous.',
  pollution_pctile:'Pollution', ozone_pctile:'Ozone', pm25_pctile:'PM2.5', dieselpm_pctile:'Diesel PM',
  ac_prob:'AC', ac_prob_romitti:'AC (Rom.)', treecanopy_pctile:'Canopy',
  parkaccess_pctile:'Parks', pct_pre1980:'Old Housing', pct_pre1960:'Pre-1960',
  pct_renter:'Renters', pct_green:'Greenspace', pct_impervious:'Impervious',
  income_pctile:'Income', abovepoverty_pctile:'Above Pov.', economic_pctile:'Economic',
  education_pctile:'Education', houseing_pctile:'Housing Q.', transportation_pctile:'Transport',
  social_pctile:'Social', neighborhood_pctile:'Neighb.', insured_pctile:'Insured',
  automobile_pctile:'Auto', homeownership_pctile:'Homeown.', commute_pctile:'Commute',
  voting_pctile:'Voting', retail_pctile:'Food Access'
};
var _EXP_MODE_SHORT={daytime:'Daytime', nighttime:'Nighttime', wbgt:'Humid', surface:'Surface'};
// Radar wedges (the "thirds"). Exposure spokes are the four heat modes (the
// selected one active, the rest greyed context). Sensitivity/Adaptive spokes are
// chosen at render time from the enabled variables (top by PCA loading), then
// backfilled from `defaults` (greyed) when fewer than four are enabled.
var _RADAR_SECTORS=[
  {pillar:'exposure',    label:'Heat Exposure', c:-90, scoreKey:'_exposure',    invScore:false, defaults:[]},
  {pillar:'sensitivity', label:'Sensitivity',   c:30,  scoreKey:'_sensitivity', invScore:false,
    defaults:['EP_AGE65','EP_AGE17','RPL_THEMES','EP_POV150']},
  {pillar:'adaptive',    label:'Adaptive Capacity', c:150, scoreKey:'_adaptive', invScore:true,
    defaults:['ac_prob','treecanopy_pctile','income_pctile','pct_green']}
];
// County percentile for one variable, flipped to a deficit when inv (capacity).
function _radarPct(key,inv,fi){
  var arr=rawPct[key]||[]; var b=(arr[fi]!=null)?arr[fi]:null;
  return (b==null)?null:(inv?100-b:b);
}
// Plot a spoke as a deficit (100-pct) when the variable is protective (higher
// value = lower risk, POLARITY===false), so "outward" always means more risk.
function _radarInv(key){ return POLARITY[key]===false; }

// Per-mode exposure composite percentiles — exposureScore() run for EVERY heat
// mode, not just the selected one, so each mode-spoke shows its true composite
// and the selected mode's spoke reconciles exactly with the pillar score.
// Cached per scenario/period token (deltas only change with those).
var _expModeCache={tok:null,val:null};
function expModeScores(){
  var tok=curScenario+'|'+curPeriod;
  if(_expModeCache.tok===tok&&_expModeCache.val)return _expModeCache.val;
  var feats=GJ.features,N=feats.length,out={},i,j;
  HEAT_MODES.forEach(function(mode){
    var keys=mode.layers.filter(function(k){
      for(var t=0;t<N;t++){if(feats[t].properties[k]!=null)return true;}return false;});
    if(!keys.length){out[mode.key]=null;return;}
    var acc=new Array(N);for(i=0;i<N;i++)acc[i]=0;
    for(j=0;j<keys.length;j++){
      var d=getDeltaArr(keys[j]),x=new Array(N),mu=0;
      for(i=0;i<N;i++){var v=feats[i].properties[keys[j]];if(v==null)v=0;if(d)v+=d[i];x[i]=v;mu+=v;}
      mu/=N;var sd=0;for(i=0;i<N;i++)sd+=(x[i]-mu)*(x[i]-mu);
      sd=Math.sqrt(sd/(N-1));if(sd<1e-10)sd=1;
      for(i=0;i<N;i++)acc[i]+=(x[i]-mu)/sd;
    }
    for(i=0;i<N;i++)acc[i]/=keys.length;
    out[mode.key]=pctRank(acc);
  });
  _expModeCache={tok:tok,val:out};
  return out;
}
// Rank a pillar PCA result's variables by contribution to the composite
// (variance-weighted squared loading over the retained PCs). Returns keys, desc.
function radarImportance(R){
  var keys=R.keys||[],pcs=R.pcs||[],nret=Math.max(1,R.nret||1);
  return keys.map(function(k,j){
    var s=0;
    for(var t=0;t<nret&&t<pcs.length;t++){
      var L=(pcs[t].loadings&&pcs[t].loadings[j]!=null)?pcs[t].loadings[j]:0;
      var w=(pcs[t].pct!=null)?pcs[t].pct:0;
      s+=L*L*w;
    }
    return {key:k,imp:s};
  }).sort(function(a,b){return b.imp-a.imp;}).map(function(o){return o.key;});
}
// Build up to four spokes for a wedge. Exposure: the four heat modes (selected
// one active). Sensitivity/Adaptive: top enabled variables by PCA loading
// (active), backfilled with greyed default variables to fill four.
function radarSpokes(sec,fi){
  if(sec.pillar==='exposure'){
    var ms=expModeScores(), built=_builtSettings().heatMode;
    return HEAT_MODES.map(function(m){
      var arr=ms[m.key], lead=(m.layers&&m.layers.length)?m.layers[0]:null;
      return {key:m.key, short:(_EXP_MODE_SHORT[m.key]||m.label), full:m.label,
              pct:(arr&&arr[fi]!=null)?arr[fi]:null, active:(m.key===built),
              rawKey:lead, isMode:true, inv:false};
    });
  }
  var out=[],have={},R=results&&results[sec.pillar];
  if(R&&R.keys&&R.keys.length){
    var ranked=radarImportance(R);
    for(var i=0;i<ranked.length&&out.length<4;i++){
      var k=ranked[i],iv=_radarInv(k);
      out.push({key:k, short:(RADAR_SHORT[k]||varLabel[k]||k), full:(varLabel[k]||k),
                pct:_radarPct(k,iv,fi), active:true, rawKey:k, isMode:false, inv:iv});
      have[k]=1;
    }
  }
  // Pre-build: show the defaults normally. After a build, a pillar left out of
  // it shows them greyed, like any other filler.
  var fillerActive=!results;
  for(var d=0;d<sec.defaults.length&&out.length<4;d++){
    var dk=sec.defaults[d];if(have[dk])continue;
    var div=_radarInv(dk);
    out.push({key:dk, short:(RADAR_SHORT[dk]||varLabel[dk]||dk), full:(varLabel[dk]||dk),
              pct:_radarPct(dk,div,fi), active:fillerActive, rawKey:dk, isMode:false, inv:div});
    have[dk]=1;
  }
  return out;
}
function radarSVG(fi,pr){
  var cx=200, cy=158, R=84, W=400, H=320, OFF=[-42,-18,18,42];
  function P(deg,r){var a=deg*Math.PI/180;return[cx+r*Math.cos(a),cy+r*Math.sin(a)];}
  function PX(deg,r){return P(deg,r)[0].toFixed(1);}
  function PY(deg,r){return P(deg,r)[1].toFixed(1);}
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="336" height="269" xmlns="http://www.w3.org/2000/svg">';
  s+='<style>text{font-family:-apple-system,BlinkMacSystemFont,sans-serif}</style>';
  // tinted wedges (the "thirds")
  _RADAR_SECTORS.forEach(function(sec){
    var col=PMETA[sec.pillar]?PMETA[sec.pillar].color:'#888';
    s+='<path d="M'+cx+','+cy+' L'+PX(sec.c-60,R)+','+PY(sec.c-60,R)+' A'+R+','+R+' 0 0 1 '+
      PX(sec.c+60,R)+','+PY(sec.c+60,R)+' Z" fill="'+col+'" fill-opacity="0.07"/>';
  });
  // concentric ring guides + wedge boundary spokes
  [0.25,0.5,0.75,1].forEach(function(f){
    s+='<circle cx="'+cx+'" cy="'+cy+'" r="'+(R*f).toFixed(1)+'" fill="none" stroke="#d9dce1" stroke-width="'+(f===1?1.2:1)+'"/>';
  });
  [-30,90,210].forEach(function(b){
    s+='<line x1="'+cx+'" y1="'+cy+'" x2="'+PX(b,R)+'" y2="'+PY(b,R)+'" stroke="#cfd3d9" stroke-width="1"/>';
  });
  // per-sector fans, dots, labels
  _RADAR_SECTORS.forEach(function(sec){
    var col=PMETA[sec.pillar]?PMETA[sec.pillar].color:'#888', GRAY='#9aa1ab';
    var isExp=(sec.pillar==='exposure');
    var spokes=radarSpokes(sec,fi);
    // Pillar score. Exposure reads the built mode's composite so the centre
    // dot and header reconcile exactly with that mode's spoke.
    var score;
    if(isExp){var ms=expModeScores(),sa=ms[_builtSettings().heatMode];score=(sa&&sa[fi]!=null)?sa[fi]:0;}
    else{var sv=pr[sec.scoreKey];
      if(sv!=null){score=sec.invScore?100-sv:sv;}
      else if(results){score=null;}   // pillar left out of the build: no score
      else{var sm=0,cn=0;spokes.forEach(function(sp){if(sp.active&&sp.pct!=null){sm+=sp.pct;cn++;}});score=cn?sm/cn:0;}}
    var pts=[
      {deg:sec.c+OFF[0],sp:spokes[0]||null},
      {deg:sec.c+OFF[1],sp:spokes[1]||null},
      {deg:sec.c,       main:true,pct:score},
      {deg:sec.c+OFF[2],sp:spokes[2]||null},
      {deg:sec.c+OFF[3],sp:spokes[3]||null}
    ];
    pts.forEach(function(pt){
      var muted=(!pt.main&&(!pt.sp||!pt.sp.active));
      s+='<line x1="'+cx+'" y1="'+cy+'" x2="'+PX(pt.deg,R)+'" y2="'+PY(pt.deg,R)+'" stroke="'+(muted?GRAY:col)+
        '" stroke-opacity="'+(pt.main?0.35:(muted?0.12:0.16))+'" stroke-width="'+(pt.main?1.4:1)+'"/>';
    });
    // Filled profile: exposure spans all four modes (heat fingerprint);
    // sensitivity/adaptive span only the active (selected) spokes + score, so
    // the coloured shape reflects what actually drives the pillar score.
    var poly=[cx.toFixed(1)+','+cy.toFixed(1)];
    pts.forEach(function(pt){
      if(!(pt.main||(pt.sp&&(isExp||pt.sp.active))))return;
      var pc=pt.main?(pt.pct==null?0:pt.pct):(pt.sp.pct==null?0:pt.sp.pct);
      poly.push(PX(pt.deg,R*pc/100)+','+PY(pt.deg,R*pc/100));
    });
    s+='<polygon points="'+poly.join(' ')+'" fill="'+col+'" fill-opacity="0.22" stroke="'+col+
      '" stroke-width="1.4" stroke-linejoin="round"/>';
    pts.forEach(function(pt){
      if(pt.main){
        if(pt.pct==null)return;
        var pcm=pt.pct;
        s+='<circle cx="'+PX(pt.deg,R*pcm/100)+'" cy="'+PY(pt.deg,R*pcm/100)+'" r="5.5" fill="'+col+
          '" stroke="#fff" stroke-width="1.4"><title>'+sec.label+' score: '+Math.round(pcm)+'</title></circle>';
        return;
      }
      var sp=pt.sp; if(!sp)return;
      var muted=!sp.active, dotCol=muted?GRAY:col, pc=(sp.pct==null?0:sp.pct);
      var titleTxt;
      if(sp.isMode){
        titleTxt=sp.full+' composite ('+(sp.pct==null?'n/a':Math.round(sp.pct)+' pctile')+(sp.active?', selected mode':', not selected')+')';
      }else{
        // getAdj, not pr[] — under a future scenario the raw property is still
        // present-day, so an unadjusted label would contradict the detail
        // modal's rows (which already apply the delta).
        var rv=pr[sp.rawKey];
        var rs=(sp.rawKey!=null&&rv!=null&&!isNaN(rv))?(': '+fmtCmpRaw(sp.rawKey,getAdj(sp.rawKey,fi))):'';
        var pctTxt=(sp.pct==null?'n/a':Math.round(sp.pct)+' pctile');
        titleTxt=sp.full+rs+' ('+pctTxt+(muted?', not selected':'')+')'+
          (sp.inv?' \\u2014 capacity gap, higher = lower capacity':'');
      }
      var rr=sp.active?(sp.isMode?4.2:3.4):2.5;
      s+='<circle cx="'+PX(pt.deg,R*pc/100)+'" cy="'+PY(pt.deg,R*pc/100)+'" r="'+rr+'" fill="'+dotCol+
        '" stroke="#fff" stroke-width="'+(sp.active?1.4:1)+'"><title>'+_addrEsc(titleTxt)+'</title></circle>';
      var ca=Math.cos(pt.deg*Math.PI/180), anc=Math.abs(ca)<0.30?'middle':(ca>0?'start':'end');
      var lx=PX(pt.deg,R+12), ly=P(pt.deg,R+12)[1];
      var valStr;
      if(radarValMode==='abs'){var rv2=(sp.rawKey!=null)?pr[sp.rawKey]:null;valStr=(rv2!=null&&!isNaN(rv2))?fmtCmpRaw(sp.rawKey,getAdj(sp.rawKey,fi)):(sp.pct==null?'-':String(Math.round(sp.pct)));}
      else{valStr=(sp.pct==null?'-':String(Math.round(sp.pct)));}
      s+='<text x="'+lx+'" y="'+(ly-1).toFixed(1)+'" text-anchor="'+anc+'" font-size="9.5" font-weight="'+(sp.active?'700':'600')+'" fill="'+(muted?GRAY:col)+'">'+_addrEsc(sp.short)+'</text>';
      s+='<text x="'+lx+'" y="'+(ly+9).toFixed(1)+'" text-anchor="'+anc+'" font-size="9" font-weight="500" fill="'+(muted?GRAY:'#6b7280')+'">'+_addrEsc(valStr)+'</text>';
    });
    var cam=Math.cos(sec.c*Math.PI/180), anc2=Math.abs(cam)<0.30?'middle':(cam>0?'start':'end'), mp=P(sec.c,R+26);
    // Adaptive's label is the longest; nudge it right + constrain width so
    // "Adaptive Capacity NN" clears the left edge without overlapping the disk.
    var isAdapt=(sec.pillar==='adaptive'), hx=mp[0]+(isAdapt?18:0);
    var tl=isAdapt?' textLength="116" lengthAdjust="spacingAndGlyphs"':'';
    s+='<text x="'+hx.toFixed(1)+'" y="'+mp[1].toFixed(1)+'" text-anchor="'+anc2+'"'+tl+' font-size="11.5" font-weight="700">'+
      '<tspan fill="'+col+'">'+sec.label+' </tspan><tspan fill="#374151">'+(score==null?'\\u2014':Math.round(score))+'</tspan></text>';
  });
  // Whole-compass reminder that every wedge is risk-oriented (higher = worse).
  // Easy to miss for Adaptive Capacity, whose map layer runs the opposite way.
  // Sits in the clear bottom margin so it never crowds the wedges or spokes.
  s+='<text x="'+cx+'" y="'+(cy+130)+'" text-anchor="middle" font-size="10.5" fill="#9aa1ab">(higher score = higher risk)</text>';
  return s+'</svg>';
}
// Segmented-button styles for the in-popup Percentile/Value toggle.
function _hvtgStyle(active,side){
  return 'border:1px solid #cbd5e1;font-family:inherit;font-size:10.5px;font-weight:600;'+
    'padding:3px 12px;cursor:pointer;'+
    (side==='l'?'border-radius:6px 0 0 6px;border-right:none;':'border-radius:0 6px 6px 0;')+
    (active?'background:#2563eb;color:#fff;border-color:#2563eb;':'background:#fff;color:#475569;');
}
// Flip the radar between percentile and real-value labels, re-rendering in place.
function setRadarMode(m){
  radarValMode=m;
  var holder=document.getElementById('hv-radar-holder');
  if(holder){var fi=+holder.getAttribute('data-fi'); var f=GJ.features[fi]; if(f) holder.innerHTML=radarSVG(fi,f.properties);}
  var a=document.getElementById('hv-tg-pct'), b=document.getElementById('hv-tg-abs');
  if(a) a.style.cssText=_hvtgStyle(m==='pct','l');
  if(b) b.style.cssText=_hvtgStyle(m==='abs','r');
}

/* ================================================================
   MAP DRAW
   ================================================================ */
function drawTracts(){
  if(tLayer){map.removeLayer(tLayer);tLayer=null;}
  // Build GEOID->index lookup for raw percentile access
  if(!window._gIdx){window._gIdx={};GJ.features.forEach(function(f,i){window._gIdx[f.properties.GEOID]=i;});}
  tLayer=L.geoJSON(GJ,{
    style:tractStyle,
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
      _ensureTractJuris();
      var jn=_tractJurisName[fi];
      var jurisLabel=jn?('City of '+jn):'Unincorporated San Diego County';
      var h='<div style="font-family:sans-serif;font-size:13px;line-height:1.6">'+
        '<b>Tract '+p.GEOID.slice(-6)+'</b> <span style="color:#555;font-size:11.5px">('+_addrEsc(jurisLabel)+')</span><br>';
      if(rawVar){
        var rv=p[rawVar];
        var dArrTip=getDeltaArr(rawVar);
        if(dArrTip){
          var delta=dArrTip[fi];
          var deltaF=delta*9/5;   // temperature DIFFERENCE: scale only, no +32 offset
          h+='\\u0394 '+(varLabel[rawVar]||rawVar)+': <b style="color:#c0392b">'+
            (delta>=0?'+':'')+delta.toFixed(2)+'\\u00b0C ('+(deltaF>=0?'+':'')+deltaF.toFixed(1)+'\\u00b0F)</b><br>';
          h+='<span style="color:#888;font-size:11px">Present: '+(rv!=null?fmtCmpRaw(rawVar,rv):'—')+
            ' \\u2192 Projected: '+fmtCmpRaw(rawVar,getAdj(rawVar,fi))+'</span><br>';
        }else{
          h+=(varLabel[rawVar]||rawVar)+': '+(rv!=null?rv:'—')+'<br>';
        }
      }else{
        if(p._comp!=null){
          var _pc=_dtRiskColor(p._comp);
          h+='<div style="display:inline-block;margin:2px 0 4px;font-size:11px;font-weight:600;'+
            'padding:3px 9px;border-radius:999px;color:#fff;background:'+_pc+'">Composite risk '+
            p._comp.toFixed(0)+'/100</div>';
        }
        h+='<div style="display:flex;justify-content:center;margin:8px 0 2px">'+
          '<button id="hv-tg-pct" onclick="setRadarMode(\\'pct\\')" style="'+_hvtgStyle(radarValMode==='pct','l')+'">Percentile</button>'+
          '<button id="hv-tg-abs" onclick="setRadarMode(\\'abs\\')" style="'+_hvtgStyle(radarValMode==='abs','r')+'">Value</button>'+
        '</div>';
        h+='<div id="hv-radar-holder" data-fi="'+fi+'" style="text-align:center">'+radarSVG(fi,p)+'</div>';
      }
      h+='<div class="popup-btn-row">'+
        '<button class="popup-more-btn" onclick="map.closePopup();openDetailDrawer(\\''+p.GEOID+'\\')">Show more info \\u25b8</button>'+
        '<button class="popup-compare-btn" onclick="startCompare(\\''+p.GEOID+'\\')">Compare with another tract</button>'+
        '<button class="popup-pdf-btn" onclick="downloadTractPdf(\\''+p.GEOID+'\\')">Export tract PDF</button>'+
      '</div>';
      h+='</div>';
      lyr.bindPopup(h,{maxWidth:360,minWidth:344});
    }
  }).addTo(map);
  // Keep overlays on top of tracts: unincorporated region first (lowest), then
  // city and tribal outlines above it.
  if(typeof unincorpLayer!=='undefined'&&unincorpLayer&&map.hasLayer(unincorpLayer))unincorpLayer.bringToFront();
  Object.keys(jurisLayers).forEach(function(k){if(map.hasLayer(jurisLayers[k]))jurisLayers[k].bringToFront();});
  if(typeof tribalLayers!=='undefined'){
    Object.keys(tribalLayers).forEach(function(k){if(map.hasLayer(tribalLayers[k]))tribalLayers[k].bringToFront();});
  }
  if(dsRenderOn){ if(DS_LAYERS[curLayer]) renderDownscaled(); else setDsRender(false); }   // recolor / auto-off on unsupported layer
  _updateBivLegend();   // bivariate legend tracks the downscaled-composite view
  showLegend();
}

// Hover-tooltip helper. Used by showLegend AND showLayerBtns, so it must live
// at module scope — when nested inside showLegend, calls from showLayerBtns
// throw ReferenceError and silently nuke the PC dropdown render.
function ic(t){return ' <span class="info-ic" tabindex="0">i<span class="info-pop">'+t+'</span></span>';}
// Position sidebar info-bubble popups to the RIGHT of their icon, out over the
// map. They're promoted to position:fixed in CSS (to escape #sidebar's overflow
// clip); here we set the fixed coords from the icon's rect on hover/focus.
// Delegated on document so dynamically-rendered buttons are covered. On mobile
// the sidebar is an overlay drawer, so the CSS media query keeps an in-drawer
// popup and we clear any stale inline coords instead of repositioning.
(function(){
  var _popMQ=window.matchMedia('(max-width:768px),(max-height:500px) and (orientation:landscape)');
  function positionSidebarPop(e){
    var t=e.target, ico=(t&&t.closest)?t.closest('#sidebar .info-ic'):null;
    if(!ico)return;
    var pop=ico.querySelector('.info-pop'); if(!pop)return;
    if(_popMQ.matches){pop.style.left='';pop.style.top='';return;}
    var r=ico.getBoundingClientRect();
    pop.style.left=(r.right+9)+'px';
    pop.style.top=(r.top+r.height/2)+'px';
  }
  document.addEventListener('mouseover',positionSidebarPop);
  document.addEventListener('focusin',positionSidebarPop);
})();

// Trivariate (ternary) color for the Driver Mix layer. White at the centroid
// (the three drivers contribute equally, so no single one dominates) and each
// pillar's pure hue at its corner. We subtract the most-balanced share (min)
// from each, so the "excess" measures how far one driver pulls ahead; the rest
// is white. e,s,a are the raw per-pillar risk contributions; dg is pal('DRGB').
// Saturation boost for the Driver Mix ternary blend (_triMix). Percentile
// pillar scores cluster in magnitude, so the raw excess over the most-balanced
// share is usually tiny and the map reads near-white. gamma<1 expands that
// dynamic range (lower = more saturated); endpoints are fixed (balanced tract
// stays white, single-driver tract stays its pure pillar hue).
var _TRI_SAT_GAMMA=0.5;
function _triMix(e,s,a,dg){
  var tot=e+s+a;
  if(tot<1e-6)return[230,230,230];                 // no enabled drivers -> neutral
  var we=e/tot,ws=s/tot,wa=a/tot,m=Math.min(we,ws,wa);
  // Boost saturation (sum of excesses) via gamma; hue direction unchanged.
  var S=1-3*m,k=(S>1e-6)?Math.pow(S,_TRI_SAT_GAMMA)/S:0;
  var xe=(we-m)*k,xs=(ws-m)*k,xa=(wa-m)*k,wt=(1-(xe+xs+xa))*255;          // wt = white share = (1-Σexcess)*255
  return[wt+xe*dg.exposure[0]+xs*dg.sensitivity[0]+xa*dg.adaptive[0],
         wt+xe*dg.exposure[1]+xs*dg.sensitivity[1]+xa*dg.adaptive[1],
         wt+xe*dg.exposure[2]+xs*dg.sensitivity[2]+xa*dg.adaptive[2]];
}

// Render the trivariate colorbar for the Driver Mix layer: a triangle whose
// corners are the three pillar hues and whose center is white, filled via
// _triMix. Returns a fresh <canvas> at the requested backing size, so the same
// routine serves both the live legend and the hi-res PNG/PDF export. Honors
// cbMode via pal().
function _triCanvas(w,h){
  var dg=pal('DRGB');
  var cv=document.createElement('canvas');cv.width=w;cv.height=h;
  var c=cv.getContext('2d');
  var fs=Math.max(9,Math.round(h*0.115));
  var padT=fs+3,padB=fs+3,padX=Math.round(w*0.07);
  // E top, A bl, S br — matches the popup radar's wedge order (_RADAR_SECTORS
  // puts sensitivity at c:30 = bottom-right, adaptive at c:150 = bottom-left),
  // so the two graphics don't mirror each other. The barycentric solve below is
  // generic over the vertices, and _triMix takes weights by argument position,
  // so moving corners here restyles the legend without recoloring any tract.
  var Ex=w/2,Ey=padT,Ax=padX,Ay=h-padB,Sx=w-padX,Sy=h-padB;
  var det=(Sy-Ay)*(Ex-Ax)+(Ax-Sx)*(Ey-Ay);
  var img=c.createImageData(w,h),dt=img.data;
  for(var py=0;py<h;py++){for(var px=0;px<w;px++){
    var l1=((Sy-Ay)*(px-Ax)+(Ax-Sx)*(py-Ay))/det;                 // weight: exposure
    var l2=((Ay-Ey)*(px-Ax)+(Ex-Ax)*(py-Ay))/det;                 // weight: sensitivity
    var l3=1-l1-l2,o=(py*w+px)*4;                                 // weight: adaptive
    if(l1>=-0.005&&l2>=-0.005&&l3>=-0.005){
      var rgb=_triMix(l1,l2,l3,dg);                               // shares sum to 1 here
      dt[o]=rgb[0];dt[o+1]=rgb[1];dt[o+2]=rgb[2];dt[o+3]=255;
    }
  }}
  c.putImageData(img,0,0);
  // Abbreviated corner labels, each in its pillar hue with a white halo.
  c.font='700 '+fs+'px -apple-system,Segoe UI,Roboto,sans-serif';
  c.lineWidth=Math.max(2,fs*0.28);c.strokeStyle='rgba(255,255,255,0.95)';c.lineJoin='round';
  function lab(txt,x,y,align,base,rgb){c.textAlign=align;c.textBaseline=base;
    c.strokeText(txt,x,y);c.fillStyle='rgb('+rgb.join(',')+')';c.fillText(txt,x,y);}
  lab('Heat',Ex,1,'center','top',dg.exposure);
  lab('Adapt.',1,h-1,'left','bottom',dg.adaptive);
  lab('Sens.',w-1,h-1,'right','bottom',dg.sensitivity);
  return cv;
}

function showLegend(){ _showLegendBody(); _appendStaleNote(); }

// Amber footnote whenever the map is showing a composite-derived layer that
// predates the user's latest panel changes. The preview views aren't stale —
// they're live — so they're exempt.
function _appendStaleNote(){
  if(!(compositeStale&&results))return;
  if(rawVar||curLayer==="heatmode")return;
  var el=document.getElementById("legend");
  if(!el||el.style.display==="none")return;
  el.insertAdjacentHTML('beforeend',
    '<div style="font-size:10px;color:#b45309;margin-top:5px;line-height:1.35">'+
    'Showing the last build \\u2014 your changes aren\\u2019t applied yet.</div>');
}

function _showLegendBody(){
  var el=document.getElementById("legend");
  // Selected heat mode, previewed live — not yet folded into a build.
  if(curLayer==="heatmode"&&!rawVar){
    var md=heatModeDef(heatMode);
    var hstops=(pal('PCC').exposure||pal('CR')).map(function(c){return"rgb("+c.join(",")+")"}).join(",");
    el.innerHTML='<div style="font-weight:600;margin-bottom:2px">Heat Exposure \\u2014 '+_addrEsc((md&&md.label)||'')+
      ic('The selected heat mode, scored as the mean of its z-scored layers and percentile-ranked across San Diego County tracts. Higher = hotter relative to other tracts. This is a preview of the mode on its own; it is not a composite.')+'</div>'+
      '<div style="font-size:10px;color:#b45309;margin-bottom:4px">Preview \\u2014 click Build to fold this into the composite.</div>'+
      '<div class="grad" style="background:linear-gradient(to right,'+hstops+')"></div>'+
      '<div class="labels"><span>Low</span><span>High</span></div>';
    el.style.display="block";return;
  }
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
        '<div class="labels"><span>+'+dr.min.toFixed(1)+'\\u00b0C<br>+'+(dr.min*9/5).toFixed(1)+'\\u00b0F</span>'+
        '<span style="text-align:right">+'+dr.max.toFixed(1)+'\\u00b0C<br>+'+(dr.max*9/5).toFixed(1)+'\\u00b0F</span></div>';
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
    el.innerHTML='<div style="font-weight:600;margin-bottom:2px">Composite Risk'+ic('Mean of the three pillar scores: Heat Exposure, Population Sensitivity, and low Adaptive Capacity. Heat Exposure is the heat mode you selected; Sensitivity and Adaptive Capacity are PCA-weighted composites of the variables you turned on. Each is percentile-ranked across San Diego County tracts. Higher = greater overall heat risk.')+'</div>'+
      '<div class="grad" style="background:linear-gradient(to right,'+cstops+')"></div>'+
      '<div class="labels"><span>Low</span><span>High</span></div>';
    el.style.display="block";
  }else if(curLayer==="exposure"){
    var estops=(pal('PCC').exposure||pal('CR')).map(function(c){return"rgb("+c.join(",")+")"}).join(",");
    var emL=(results&&results.exposure&&results.exposure.label)||"Heat Exposure";
    el.innerHTML='<div style="font-weight:600;margin-bottom:2px">Heat Exposure &mdash; '+emL+ic('The selected heat mode, scored as the mean of its z-scored layers and percentile-ranked across San Diego County tracts. Higher = hotter relative to other tracts. This is not a PCA blend.')+'</div>'+
      '<div class="grad" style="background:linear-gradient(to right,'+estops+')"></div>'+
      '<div class="labels"><span>Low</span><span>High</span></div>';
    el.style.display="block";
  }else if(curLayer==="driver"){
    var dr=pal('DRGB');
    var rgb=function(arr){return"rgb("+arr.join(",")+")"};
    el.innerHTML='<div style="font-weight:600;margin-bottom:4px">Dominant Driver'+ic('Each tract is colored by whichever driver it ranks highest on within San Diego County: Heat Exposure, Population Sensitivity, or low Adaptive Capacity. Each driver is a county percentile rank (where the tract sits relative to all other county tracts), so this flags the driver the tract stands out most on, not an absolute share of its risk.')+'</div>'+
      '<div class="dl-row"><div class="dl-sw" style="background:'+rgb(dr.exposure)+'"></div>Heat Exposure</div>'+
      '<div class="dl-row"><div class="dl-sw" style="background:'+rgb(dr.sensitivity)+'"></div>Population Sensitivity</div>'+
      '<div class="dl-row"><div class="dl-sw" style="background:'+rgb(dr.adaptive)+'"></div>Adaptive Capacity</div>'+
      '<div style="font-size:10px;color:#999;margin-top:3px;font-style:italic">Lighter = lower risk</div>';
    el.style.display="block";
  }else if(curLayer==="drivermix"){
    el.innerHTML='<div style="font-weight:600;margin-bottom:4px">Driver Mix'+ic('A trivariate blend of how this tract ranks within San Diego County on the three drivers &mdash; Heat Exposure, Population Sensitivity, and low Adaptive Capacity. Each driver is a county percentile rank (0&ndash;100): where the tract sits relative to all other county tracts, not an absolute share of its risk. The blend compares those three ranks, so a tract leans toward the corner of whichever driver it ranks highest on relative to its own other two. A white or pale tract ranks about evenly across all three, with no single dominant driver. Lightness also tracks overall composite risk (lighter = lower).')+'</div>'+
      '<div class="tri-host"></div>'+
      '<div style="font-size:10px;color:#999;margin-top:1px;font-style:italic">Corner = one dominant driver &middot; white = balanced mix &middot; lighter = lower risk</div>';
    el.style.display="block";
    var th=el.querySelector('.tri-host');
    if(th){var tcv=_triCanvas(300,234);tcv.style.width='150px';tcv.style.height='117px';th.appendChild(tcv);}
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
  // ── Heat Exposure: single-select heat mode (radio), not a PCA blend ──
  (function(){
    var pm=PMETA.exposure;
    html+='<div class="pillar-section" data-tour="pillar-exposure">';
    html+='<div class="pillar-hdr" style="border-color:'+pm.color+';color:'+pm.color+'">'+pm.label+'</div>';
    html+='<div class="mode-help">Pick one heat hazard to map. Each is scored on its own &mdash; daytime and nighttime heat affect different places.</div>';
    HEAT_MODES.forEach(function(m){
      var on=(heatMode===m.key);
      var n=m.layers.length;
      var meta=n>1?('('+n+' layers)'):'(1 layer)';
      // Hover the layer-count to see which layers are averaged into the mode.
      var layerTip=(n>1?'Averaged layers:\\n':'Layer:\\n')+
        m.layers.map(function(k){return '\\u2022 '+(varLabel[k]||k);}).join('\\n');
      html+='<label class="mode-row'+(on?' on':'')+'" id="mr_'+m.key+'" style="color:'+pm.color+'" title="'+m.desc.replace(/"/g,'&quot;')+'">'+
        '<input type="radio" name="heatmode"'+(on?' checked':'')+' onchange="selectHeatMode(\\''+m.key+'\\')">'+
        '<span class="mode-txt"><span class="mode-name">'+m.label+'</span>'+
        '<span class="mode-meta" title="'+layerTip.replace(/"/g,'&quot;')+'">'+meta+'</span></span></label>';
    });
    html+='</div>';
  })();
  // ── Sensitivity & Adaptive Capacity: multi-toggle variables (PCA) ──
  ["sensitivity","adaptive"].forEach(function(pid){
    var pm=PMETA[pid];
    html+='<div class="pillar-section" data-tour="pillar-'+pid+'">';
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
        var src=VSRC[v.key]?'<a class="src-link" href="'+VSRC[v.key]+'" target="_blank" rel="noopener">(Source)</a>':"";
        html+='<div class="'+cls+'" id="vr_'+v.key+'"><label style="color:'+pm.color+'">'+
          '<input type="checkbox" onchange="tog(\\''+v.key+'\\')" '+chk+'>'+
          '<span style="color:#444">'+v.label+'</span></label>'+src+'</div>';
      });
      html+='</details>';
    });
    html+='</div>';
  });
  el.innerHTML=html;
  ["sensitivity","adaptive"].forEach(syncPillarCb);
}

// Flag the built composite as out of date with the panel. No-op before the first
// build, when the button is already the obvious call to action.
function markCompositeStale(){
  if(!results||compositeStale)return;
  compositeStale=true;
  syncBuildBtn();
  showLegend();   // surfaces the "showing the last build" note
}
// Reflect stale state on the Build button: amber + a rewritten label, so the
// pending action is legible even if the user never looks at the legend.
function syncBuildBtn(){
  var b=document.getElementById('compute-btn');
  if(!b)return;
  var stale=(compositeStale&&!!results);
  b.classList.toggle('needs-build',stale);
  b.textContent=stale?'Rebuild Composite Risk Map':'Build Composite Risk Map';
}

// Select a heat mode (exposure is single-select). Always previews the mode on the
// map. The composite is deliberately NOT recomputed here — it goes stale until
// the user clicks Build.
function selectHeatMode(key){
  if(!heatModeDef(key))return;
  heatMode=key;
  var rows=document.querySelectorAll('.mode-row');
  for(var i=0;i<rows.length;i++)rows[i].classList.remove('on');
  var row=document.getElementById('mr_'+key);if(row)row.classList.add('on');
  previewHeatMode();
  markCompositeStale();
}

// Show the selected heat mode on the map: its own composite percentile — the
// very score that becomes Heat Exposure on Build — read live from
// expModeScores(). Nothing is written to the features, so the committed
// composite underneath survives untouched. Falls back to the mode's lead layer
// if the mode has no usable data.
function previewHeatMode(){
  var mode=heatModeDef(heatMode);if(!mode)return;
  var ms=expModeScores();
  if(ms&&ms[heatMode]){
    rawVar=null;curLayer="heatmode";
  }else{
    var lead=null;
    for(var i=0;i<mode.layers.length;i++){if(rawPct[mode.layers[i]]){lead=mode.layers[i];break;}}
    if(!lead)return;
    rawVar=lead;curLayer="composite";
  }
  updateDeltaRange();
  document.getElementById("prompt").style.display="none";
  drawTracts();
  if(results)showLayerBtns();   // clear the active-layer highlight: this is a preview, not a built layer
}

function tog(key){
  enabled[key]=!enabled[key];
  markCompositeStale();   // the built composite no longer reflects the panel
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
  markCompositeStale();   // the built composite no longer reflects the panel
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
    if(r&&pid==="exposure")
      html+='<div class="res-row"><b style="color:'+pm.color+'">'+pm.label+'</b>: '+
        (r.label||"heat mode")+'</div>';
    else if(r)
      html+='<div class="res-row"><b style="color:'+pm.color+'">'+pm.label+'</b>: '+
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
  [{id:"composite",label:"Composite Risk",dot:"#bd0026",
    info:"The overall score: the average of the three pillars (Heat Exposure, Population Sensitivity, and low Adaptive Capacity), each percentile-ranked across San Diego County tracts. Higher = greater heat risk."},
   {id:"driver",label:"Dominant Driver",dot:"#2c3e50",
    info:"Colors each tract by its largest risk driver: Heat Exposure, Population Sensitivity, or low Adaptive Capacity. Lighter shades mean lower composite risk."},
   {id:"drivermix",label:"Driver Mix",dot:"conic-gradient(#c0392b 0deg 120deg,#6a51a3 120deg 240deg,#16a052 240deg 360deg)"}].forEach(function(b){
    html+='<button class="lbtn'+(curLayer===b.id?" active":"")+'" onclick="setLyr(\\''+b.id+'\\')">'+
      '<span class="ldot" style="background:'+b.dot+'"></span>'+
      '<span class="lbtn-txt">'+b.label+'</span>'+(b.info?ic(b.info):'')+'</button>';});
  // Downscaled toggle (composite/driver/mix only) -- part of the map-layer group,
  // directly under the three main layer buttons. Runs the per-pixel analysis at
  // ~100 m (downscaled LST exposure + live tract sensitivity/adaptive, on the
  // populated footprint), like the report hero maps; the info icon explains it.
  if(DS_ENC && DS_LAYERS[curLayer]){
    var dsInfo=ic('Renders the map at ~100 m, masked to where people live (EPA dasymetric population). Only Heat Exposure varies within a tract, from a downscaled land-surface-temperature model; Population Sensitivity and Adaptive Capacity keep their tract-level scores.');
    html+='<div class="popwt"><label><input type="checkbox" '+(dsRenderOn?"checked":"")+
      ' onchange="setDsRender(this.checked)"><span>Populated areas</span></label>'+dsInfo+'</div>';
  }
  // Primary Spatial Patterns dropdown: the PCA pattern layers per pillar, PLUS the
  // Heat Exposure mode score nested inside it.
  if(results){
    var anyPC=false;
    ["exposure","sensitivity","adaptive"].forEach(function(pid){
      if(results[pid]&&results[pid].pcs&&results[pid].pcs.length)anyPC=true;});
    if(anyPC||results.exposure){
      // Open when the active layer is a PC pattern or the Heat Exposure layer.
      var pcOpen=/^(exposure|sensitivity|adaptive)_pc\\d+$/.test(curLayer)||curLayer==="exposure";
      html+='<details class="pc-group"'+(pcOpen?' open':'')+'>'+
        '<summary>Primary Spatial Patterns</summary>';
      if(anyPC){
        html+='<div class="pc-help">Underlying patterns from PCA on the variables in each pillar. Hover the <span class="info-ic-mini">i</span> for top contributors.</div>';
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
      }
      // Heat Exposure mode score, nested inside the dropdown (below the patterns).
      if(results.exposure){
        var emLab=results.exposure.label||"Heat Exposure";
        html+='<button class="lbtn'+(curLayer==="exposure"?" active":"")+'" onclick="setLyr(\\'exposure\\')">'+
          '<span class="ldot" style="background:'+PMETA.exposure.color+'"></span>'+
          '<span class="lbtn-txt">Heat Exposure &mdash; '+emLab+'</span></button>';
      }
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
    var scnInfo=document.getElementById('scn-info-modal');
    if(scnInfo&&scnInfo.classList.contains('open')){closeScnInfo();}
    else if(document.getElementById('scn-compare-modal').classList.contains('open')){closeScnCompareModal();}
    else if(document.getElementById('compare-modal').classList.contains('open')){closeCompareModal();}
    else if(cityModal&&cityModal.classList.contains('open')){closeCityCompareModal();}
    else if(compareMode==='awaiting'){cancelCompare();}
    else if(focusedCity){exitCityView();}
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
        var dispPctA=(pA!=null)?(Math.round(pA)+'th percentile'):'';
        var dispPctB=(pB!=null)?(Math.round(pB)+'th percentile'):'';
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
    var pcaA='<div class="cmp-pillar-hdr" style="border-color:#444;color:#444">Pillar Scores</div>';
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
  lightTiles().forEach(function(t){t.addTo(m);});
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

// UNREACHABLE — the city-compare modal markup was never written. This function
// and everything below it through closeCityCompareModal() target #city-compare-modal
// and its children (city-cmp-sel-a/b, city-cmp-vars-a/b, city-cmp-context,
// city-cmp-name/meta/map-*), none of which exist in the template; the CSS is there,
// the HTML never was. Calling this throws on the first getElementById deref, so the
// two entry points that used to call it (the city-focus bar's "Compare cities" and
// showTribalNoData's "Compare <city>") have been removed rather than left as dead
// buttons. To finish the feature, add the modal markup next to #compare-modal and
// restore those two buttons.
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
  lightTiles().forEach(function(t){t.addTo(m);});
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
    var pcaA='<div class="cmp-pillar-hdr" style="border-color:#444;color:#444">Pillar Scores</div>';
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
  var typeLabel=(type==='trust')?'off-reservation trust land':'reservation';
  var html='<div style="font-size:12px;line-height:1.45;max-width:240px">'+
    '<div style="font-weight:700;margin-bottom:4px">'+fullName+'</div>'+
    '<div style="color:#666;margin-bottom:8px">Census tract demographics don\\u2019t represent populations on this '+typeLabel+', so we don\\u2019t aggregate stats here.</div>';
  if(hit>=0){
    var geoid=GJ.features[hit].properties.GEOID;
    html+='<div style="font-size:11px;color:#666;margin-bottom:4px">Compare the surrounding tract instead:</div>'+
      '<button class="popup-compare-btn" onclick="map.closePopup();startCompare(\\''+geoid+'\\')">Compare Tract '+geoid.slice(-6)+'</button>';
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
      levelDiff:0,
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
// (scenario, period). Uses the committed build's settings (the panel's before
// the first build), so it matches the main map even while that is stale.
// Does NOT mutate GJ.features. Returns null if no variables enabled.
function computeCompositeFor(scenario, period){
  var savedScn=curScenario, savedPer=curPeriod;
  curScenario=scenario;
  curPeriod=(scenario==='present')?'':(period||'');
  try{
    var st=_builtSettings();
    var res={exposure:exposureScore(st.heatMode)};
    ['sensitivity','adaptive'].forEach(function(pid){
      res[pid]=pillarPCA(pid,st.on[pid]);
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
// (scenario, period), using the build's heat mode (the panel's before a build).
function computeExposureFor(scenario, period){
  var savedScn=curScenario, savedPer=curPeriod;
  curScenario=scenario;
  curPeriod=(scenario==='present')?'':(period||'');
  try{
    var res=exposureScore(_builtSettings().heatMode);
    return res?res.scores:null;
  } finally {
    curScenario=savedScn;
    curPeriod=savedPer;
  }
}

function openScnCompareModal(){
  // Exposure always has a selected heat mode, and the temperature layers carry
  // the climate deltas — so scenario comparison is always meaningful.
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
  // First-time visitors get the explainer automatically; afterwards it's on the "?" button.
  try{
    if(!localStorage.getItem('shrx_scn_info_seen')){
      openScnInfo();
      localStorage.setItem('shrx_scn_info_seen','1');
    }
  }catch(e){}
  setTimeout(function(){
    renderScnCompareSide('a');
    renderScnCompareSide('b');
    _refreshScnCompareLegend();
    setTimeout(_initScnSync,40);
  },30);
}

function openScnInfo(){var m=document.getElementById('scn-info-modal');if(m)m.classList.add('open');}
function closeScnInfo(){var m=document.getElementById('scn-info-modal');if(m)m.classList.remove('open');}

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
    lightTiles(18).forEach(function(t){t.addTo(m);});
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
    var partsA=scnCmpScn.a.split('|');
    var scnA=partsA[0], periodA=partsA[1]||'';
    if(!scnCmpExp.a){
      scnCmpExp.a=computeExposureFor(scnA, periodA);
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
    // Per-tract LOCA2 Δ°C lookup (returns 0 for present day or missing data)
    function _scnDelta(s,p,loca,i){
      if(s==='present'||!DELTAS||!p)return 0;
      var pp=DELTAS[p];if(!pp)return 0;
      var ps=pp[s];if(!ps)return 0;
      var arr=ps[loca];if(!arr)return 0;
      return arr[i]||0;
    }
    var showClimateSignal=(scnA!=='present'||scn!=='present');
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
      var climateBlock='';
      if(showClimateSignal){
        var dTmax=_scnDelta(scn,period,'tasmax',idx)-_scnDelta(scnA,periodA,'tasmax',idx);
        var dTmin=_scnDelta(scn,period,'tasmin',idx)-_scnDelta(scnA,periodA,'tasmin',idx);
        var sMax=(dTmax>=0?'+':''), sMin=(dTmin>=0?'+':'');
        var dTmaxF=dTmax*9/5, dTminF=dTmin*9/5;   // temperature DIFFERENCE: scale only
        climateBlock='<div style="border-top:1px solid #ddd;margin-top:4px;padding-top:4px;color:#666;font-size:11px">'+
          'Underlying climate signal (this tract, B − A):'+
          '<br>Δ mean Tmax: <b>'+sMax+dTmax.toFixed(2)+'\\u00b0C ('+sMax+dTmaxF.toFixed(1)+'\\u00b0F)</b>'+
          '<br>Δ mean Tmin: <b>'+sMin+dTmin.toFixed(2)+'\\u00b0C ('+sMin+dTminF.toFixed(1)+'\\u00b0F)</b>'+
          '</div>';
      }
      lyr.bindTooltip('Tract '+f.properties.GEOID.slice(-6)+
        '<br>Δ Heat Exposure rank: <b>'+sign+d.toFixed(1)+'</b> percentile pts'+
        '<br><span style="color:#888">A rank: '+(expA[idx]!=null?expA[idx].toFixed(1):'—')+
        ' → B rank: '+expB[idx].toFixed(1)+' (0–100)</span>'+
        climateBlock,
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
      metaEl.innerHTML='Change from <b>'+(aLab||scnCmpScn.a)+'</b> to <b>'+(lab||'')+'</b>';
    }
  }
}

function _refreshScnCompareLegend(){
  var la=document.getElementById('scn-cmp-leg-a');
  var lb=document.getElementById('scn-cmp-leg-b');
  if(la){
    var cr=pal('CR');
    var crStops=cr.map(function(c){return'rgb('+c.join(',')+')'}).join(',');
    la.innerHTML=
      '<div class="scn-leg-name">Composite heat risk</div>'+
      '<div class="scn-leg-bar" style="background:linear-gradient(to right,'+crStops+')"></div>'+
      '<div class="scn-leg-ends"><span>Lower</span><span>Higher</span></div>';
  }
  if(lb){
    var divStops=CDIV.map(function(c){return'rgb('+c.join(',')+')'}).join(',');
    // 1px neutral tick overlaid at the 50% (zero) point of the diverging scale
    var zeroMark='linear-gradient(to right,transparent calc(50% - 1px),#9ca3af calc(50% - 1px),#9ca3af calc(50% + 1px),transparent calc(50% + 1px)),';
    var dMax=(window._scnDeltaMax!=null?window._scnDeltaMax:0).toFixed(1);
    lb.innerHTML=
      '<div class="scn-leg-name">Change in heat exposure <span class="scn-leg-sub">(vs left map)</span></div>'+
      '<div class="scn-leg-dirs"><span>◀ cooler · less exposed</span><span>hotter · more exposed ▶</span></div>'+
      '<div class="scn-leg-bar" style="background:'+zeroMark+'linear-gradient(to right,'+divStops+')"></div>'+
      '<div class="scn-leg-ticks"><span>−'+dMax+'</span><span>0</span><span>+'+dMax+'</span></div>'+
      '<div class="scn-leg-unit">percentile points</div>';
  }
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

function _loadScript(src,integrity){
  return new Promise(function(resolve,reject){
    var s=document.createElement('script');
    s.src=src;s.crossOrigin='anonymous';
    // crossOrigin alone verifies nothing; the hash is what pins the bytes.
    if(integrity)s.integrity=integrity;
    s.onload=resolve;s.onerror=function(){reject(new Error('Failed to load '+src));};
    document.head.appendChild(s);
  });
}

function _ensurePdfLibs(){
  var needs=[];
  if(!window.jspdf)needs.push(_loadScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js',
    'sha384-JcnsjUPPylna1s1fvi1u12X5qjY5OL56iySh75FdtrwhO/SWXgMjoVqcKyIIWOLk'));
  return Promise.all(needs);
}

function _scenarioLabel(){
  if(curScenario==='present')return'Present Day';
  var sl={'ssp245':'SSP2-4.5 (Moderate)','ssp370':'SSP3-7.0 (High)','ssp585':'SSP5-8.5 (Very High)'};
  var pl={'2050':'Mid-Century (2035\\u20132064)','2080':'End-of-Century (2065\\u20132094)'};
  return sl[curScenario]+' \\u2022 '+(pl[curPeriod]||'');
}

// Paint the light basemap onto a 2D context for the PDF blitters, which draw
// raster tiles by hand rather than screenshotting Leaflet (see
// _captureStudyAreaMap). `pb` is a Leaflet pixel-bounds at zoom `zoom`; `S` is
// the retina scale of the target canvas. Resolves with the number of base
// tiles that actually loaded, so a caller can tell "blocked" from "drawn".
//
// Two Esri quirks are handled here. It serves no @2x tiles, so print density
// comes from fetching one zoom level deeper and drawing each tile at half its
// map size — 256 source pixels land on 256 canvas pixels at S=2. And it has no
// tiles at all past LIGHT_MAX_NATIVE_ZOOM, so beyond that we sample the
// sub-rectangle of the deepest real ancestor tile and let drawImage upscale
// it, which is what maxNativeZoom does on the live map.
function _paintLightBasemap(ctx,zoom,pb,S){
  var TILE=256;
  // One level deeper == retina density. `zoom` may be fractional (the map PDF
  // fits the county exactly), so fetch the nearest whole level and scale its
  // tiles by k onto zoom's pixel grid. For a whole zoom, k=1/2 as before.
  var rz=Math.round(zoom)+1;
  var k=Math.pow(2,zoom-rz);           // rz px -> zoom px
  var f=S*k;                           // rz px -> physical px
  var min={x:pb.min.x/k,y:pb.min.y/k}, max={x:pb.max.x/k,y:pb.max.y/k};
  var nWorld=Math.pow(2,rz);
  // Source tile + sub-rectangle for one tile of the rz grid.
  function src(x,y){
    var d=Math.max(0,rz-LIGHT_MAX_NATIVE_ZOOM), n=1<<d, sub=TILE/n;
    var sx=Math.floor(x/n), sy=Math.floor(y/n);
    var q={z:rz-d,x:sx,y:sy,sx:(x%n)*sub,sy:(y%n)*sub,sw:sub,sh:sub};
    q.base=LIGHT_BASE_URL.replace('{z}',q.z).replace('{y}',q.y).replace('{x}',q.x);
    q.ref =LIGHT_REF_URL .replace('{z}',q.z).replace('{y}',q.y).replace('{x}',q.x);
    return q;
  }
  function loadImage(url){
    return new Promise(function(res){
      var im=new Image();
      im.crossOrigin='anonymous';
      im.onload=function(){res(im);};
      im.onerror=function(){res(null);};
      im.src=url;
    });
  }
  var tasks=[], got={n:0};
  for(var tx=Math.floor(min.x/TILE);tx<=Math.floor(max.x/TILE);tx++){
    for(var ty=Math.floor(min.y/TILE);ty<=Math.floor(max.y/TILE);ty++){
      if(ty<0||ty>=nWorld)continue;
      var q=src(((tx%nWorld)+nWorld)%nWorld,ty);
      // Round each tile's edges, not its origin + size, so scaled tiles abut
      // with no hairline seams.
      var dx=Math.round((tx*TILE-min.x)*f), dy=Math.round((ty*TILE-min.y)*f);
      var dw=Math.round(((tx+1)*TILE-min.x)*f)-dx, dh=Math.round(((ty+1)*TILE-min.y)*f)-dy;
      (function(x,y,w,h,p){
        // Base first, then the transparent label raster directly over it. Each
        // label tile covers exactly its own base tile, so per-tile ordering is
        // enough — no second pass needed.
        tasks.push(loadImage(p.base).then(function(im){
          if(im){got.n++;ctx.drawImage(im,p.sx,p.sy,p.sw,p.sh,x,y,w,h);}
          return loadImage(p.ref);
        }).then(function(im){
          if(im)ctx.drawImage(im,p.sx,p.sy,p.sw,p.sh,x,y,w,h);
        }));
      })(dx,dy,dw,dh,q);
    }
  }
  return Promise.all(tasks).then(function(){return got.n;});
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
function _captureStudyAreaMap(opts){
  opts=opts||{};
  return new Promise(function(resolve,reject){
    // Logical figure size; canvas is rendered at 2x for retina-print output.
    // opts.width widens the frame to fit a caller's page.
    var TARGET_W=opts.width||1600, TARGET_H=1200;
    var SCALE=2;

    // Off-screen Leaflet map used solely for projection math (zoom-fit +
    // latLngToContainerPoint). It carries no tile layer and is never captured.
    var projDiv=document.createElement('div');
    projDiv.style.cssText='position:absolute;top:-99999px;left:-99999px;'+
      'width:'+TARGET_W+'px;height:'+TARGET_H+'px;';
    document.body.appendChild(projDiv);

    // zoomSnap:0 lets fitBounds land on the exact zoom that fills the frame;
    // snapping to a whole level left the county at ~70% of it.
    var projMap=L.map(projDiv,{zoomControl:false,attributionControl:false,
      fadeAnimation:false,zoomAnimation:false,zoomSnap:0});

    // Frame the requested bounds, else the study area (tract bounds → county bbox)
    var bounds=null;
    if(opts.bounds&&opts.bounds.isValid&&opts.bounds.isValid())bounds=opts.bounds;
    if(!bounds&&tLayer&&tLayer.getBounds){var b=tLayer.getBounds();if(b&&b.isValid())bounds=b;}
    if(!bounds)bounds=L.latLngBounds([32.53,-117.62],[33.51,-116.08]);

    var canvas=document.createElement('canvas');
    canvas.width=TARGET_W*SCALE;
    canvas.height=TARGET_H*SCALE;
    var ctx=canvas.getContext('2d');
    ctx.fillStyle='#ffffff';
    ctx.fillRect(0,0,canvas.width,canvas.height);

    // Fit as tightly as the map furniture allows: the legend gets its own column
    // on the right so it never covers a tract, and opts.padLeft makes room on
    // the left (the map PDF's variable box).
    var leg=opts.minimal?null:drawLegend(true);
    projMap.fitBounds(bounds,{paddingTopLeft:[opts.padLeft||20,20],
      paddingBottomRight:[leg?leg.w+18+16:20,20],animate:false});

    var zoom=projMap.getZoom();
    var pixelBounds=projMap.getPixelBounds();

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
    var tileTasks=[_paintLightBasemap(ctx,zoom,pixelBounds,SCALE)];

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
      // The map PDF stacks its variable box on top of this.
      canvas._scaleBarRect={x:bx*SCALE, y:by*SCALE, w:totalW*SCALE, h:totalH*SCALE};

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

    // With measureOnly, returns the legend's {w,h} (logical px) without drawing,
    // so the fit can keep tracts out from under it.
    function drawLegend(measureOnly){
      // Mirror the live in-page legend (showLegend) for the active view state.
      var sl={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'};
      var title='', subtitle='', stops=null;
      var lowLabel='Low', highLabel='High', lowLabel2='', highLabel2='';
      var driverRows=null, triMix=false;

      if(curLayer==='heatmode'&&!rawVar){
        var md=heatModeDef(heatMode);
        title='Heat Exposure — '+((md&&md.label)||'heat mode');
        subtitle=_scenarioLabel();
        stops=pal('PCC').exposure||pal('CR');
      }else if(rawVar){
        if(typeof getDeltaArr==='function'&&getDeltaArr(rawVar)&&window._deltaRange){
          var dr=window._deltaRange;
          title='Δ '+(varLabel[rawVar]||rawVar);
          subtitle=(sl[curScenario]||curScenario)+' '+curPeriod+' change from present';
          stops=pal('CD');
          lowLabel='+'+dr.min.toFixed(1)+'°C';
          highLabel='+'+dr.max.toFixed(1)+'°C';
          lowLabel2='+'+(dr.min*9/5).toFixed(1)+'°F';   // delta: scale only, no +32 offset
          highLabel2='+'+(dr.max*9/5).toFixed(1)+'°F';
        }else{
          title=varLabel[rawVar]||rawVar;
          subtitle=_scenarioLabel();
          stops=pal('PCC')[varPillar[rawVar]]||pal('CR');
        }
      }else if(curLayer==='composite'){
        title='Composite Risk';
        subtitle=_scenarioLabel();
        stops=pal('CR');
      }else if(curLayer==='exposure'){
        title='Heat Exposure — '+((results&&results.exposure&&results.exposure.label)||'heat mode');
        subtitle=_scenarioLabel();
        stops=pal('PCC').exposure||pal('CR');
      }else if(curLayer==='driver'){
        title='Dominant Driver';
        subtitle=_scenarioLabel();
        var d=pal('DRGB');
        driverRows=[
          {label:'Heat Exposure',rgb:d.exposure},
          {label:'Population Sensitivity',rgb:d.sensitivity},
          {label:'Adaptive Capacity',rgb:d.adaptive}
        ];
      }else if(curLayer==='drivermix'){
        title='Driver Mix';
        subtitle=_scenarioLabel();
        triMix=true;
      }else{
        var m=curLayer.match(/^(exposure|sensitivity|adaptive)_pc(\\d+)$/);
        if(m){
          title=PMETA[m[1]].label+' — Pattern '+m[2];
          subtitle=_scenarioLabel();
          stops=pal('PCC')[m[1]]||pal('CR');
        }
      }
      if(!title)return;

      // City-scoped exports omit the jurisdiction/tribal checklist entirely.
      var cityRows=[];
      if(!opts.jurisName&&typeof JURIS!=='undefined'&&JURIS&&typeof jurisLayers!=='undefined'){
        Object.keys(jurisLayers).sort(function(a,b){return a.localeCompare(b);}).forEach(function(k){
          if(!map.hasLayer(jurisLayers[k]))return;
          cityRows.push({label:k, color:JURIS_COLOR});
        });
      }
      var tribalRows=[];
      if(!opts.jurisName&&typeof TRIBAL!=='undefined'&&TRIBAL&&typeof tribalGroups!=='undefined'){
        Object.keys(tribalGroups).sort(function(a,b){return a.localeCompare(b);}).forEach(function(gk){
          var anyOn=tribalGroups[gk].some(function(name){return map.hasLayer(tribalLayers[name]);});
          if(!anyOn)return;
          tribalRows.push({label:gk,color:TRIBAL_COLORS.reservation,dashed:false});
        });
      }
      var unincorpOn=(!opts.jurisName&&typeof unincorpLayer!=='undefined'&&unincorpLayer&&map.hasLayer(unincorpLayer));

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
      if(unincorpOn){var uw=tw('Unincorporated county',fonts.body);if(uw>maxLabelW)maxLabelW=uw;}
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
        plan.push({op:'gradLabels',y:y}); y+=bodySize+4; if(lowLabel2)y+=bodySize+1;
      }
      if(driverRows){
        driverRows.forEach(function(r){plan.push({op:'drow',y:y,label:r.label,rgb:r.rgb});y+=rowH;});
        plan.push({op:'note',y:y+2,text:'Lighter = lower risk'});
        y+=subSize+8;
      }
      if(triMix){
        var triLW=Math.min(gradW,150), triLH=Math.round(triLW*0.78);
        plan.push({op:'triangle',y:y,w:triLW,h:triLH});
        y+=triLH+6;
        plan.push({op:'note',y:y,text:'Lighter = lower risk'});
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
      if(unincorpOn){
        y+=sectionGap;
        plan.push({op:'sect',y:y,text:'Unincorporated'});
        y+=bodySize+8;
        plan.push({op:'crow',y:y,label:'Unincorporated county',color:UNINCORP_COLOR});
        y+=rowH;
      }
      var boxH=y+pad-4;
      if(measureOnly)return {w:boxW,h:boxH};

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
          if(lowLabel2){
            var py2=(py+bodySize+1)*SCALE;
            ctx.fillStyle='#64748b';
            ctx.textAlign='left';  ctx.fillText(lowLabel2, px*SCALE, py2);
            ctx.textAlign='right'; ctx.fillText(highLabel2, (px+gradW)*SCALE, py2);
          }
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
        }else if(L.op==='triangle'){
          ctx.drawImage(_triCanvas(Math.round(L.w*SCALE),Math.round(L.h*SCALE)), px*SCALE, py*SCALE);
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
      var text='Esri, HERE, Garmin  © OpenStreetMap contributors';
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

    // When scoped to one city, clip tract drawing to that city's polygon so the
    // whole municipal area is filled exactly. (Centroid-based tract assignment
    // dropped boundary-straddling tracts, leaving gaps.) Tracts tile the county
    // with no holes, so clipping + drawing all of them fills the city cleanly.
    var clipFeats=null;
    if(opts.jurisName&&typeof JURIS!=='undefined'&&JURIS){
      clipFeats=JURIS.features.filter(function(f){return f.properties.NAME===opts.jurisName;});
    }
    function pathCity(){
      ctx.beginPath();
      clipFeats.forEach(function(f){
        var g=f.geometry;if(!g)return;
        if(g.type==='Polygon')addRings(g.coordinates);
        else if(g.type==='MultiPolygon')g.coordinates.forEach(addRings);
      });
    }

    Promise.all(tileTasks).then(function(){
      // Tracts — colored using the live `fcolor()` against current pillar/var state
      if(tLayer&&typeof GJ!=='undefined'&&GJ&&GJ.features){
        if(clipFeats&&clipFeats.length){ctx.save();pathCity();ctx.clip('evenodd');}
        GJ.features.forEach(function(f){
          var idx=window._gIdx[f.properties.GEOID]||0;
          drawFeature(f,fcolor(f.properties,idx),'#ffffff',0.5*SCALE,0.85);
        });
        if(clipFeats&&clipFeats.length)ctx.restore();
      }
      // Unincorporated region — above tracts, below city/tribal outlines (skipped on city-scoped exports)
      if(!opts.jurisName&&typeof UNINCORP!=='undefined'&&UNINCORP&&typeof unincorpLayer!=='undefined'&&unincorpLayer&&map.hasLayer(unincorpLayer)){
        UNINCORP.features.forEach(function(f){drawFeature(f,UNINCORP_COLOR,UNINCORP_COLOR,2*SCALE,0.08);});
      }
      // Jurisdiction outlines: just the focused city when scoped, else all visible
      if(typeof JURIS!=='undefined'&&JURIS){
        Object.keys(jurisLayers).forEach(function(k){
          if(opts.jurisName){if(k!==opts.jurisName)return;}
          else if(!map.hasLayer(jurisLayers[k]))return;
          var feats=JURIS.features.filter(function(f){return f.properties.NAME===k;});
          feats.forEach(function(f){drawFeature(f,null,JURIS_COLOR,1.8*SCALE,0);});
        });
      }
      // Replicate visible tribal land outlines (skipped on city-scoped exports)
      if(!opts.jurisName&&typeof TRIBAL!=='undefined'&&TRIBAL&&typeof tribalLayers!=='undefined'){
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
      if(!opts.minimal){drawLegend();drawScaleBar();}
      drawAttribution();
      cleanup();
      resolve(canvas);
    }).catch(function(err){
      cleanup();
      reject(err);
    });
  });
}

function _heroPdf(canvas,opts){
  opts=opts||{};
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

  // No layer/scenario subtitle: the in-map legend already names both.
  pdf.setFont('helvetica','bold');
  pdf.setFontSize(20);pdf.setTextColor(15,23,42);
  pdf.text('San Diego Heat Risk Explorer',M,43);

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

  // Variable box over the lower-left (open ocean at the study-area framing),
  // stacked on the scale bar. A raw-variable or heat-mode preview has no build
  // behind it, so it gets no box.
  if(results&&!rawVar&&curLayer!=='heatmode'){
    var k=imgW/canvas.width, sb=canvas._scaleBarRect||{x:0,y:canvas.height};
    _pdfVarBox(pdf,imgX+sb.x*k,imgY+sb.y*k-6,170,sb.y*k-12);
  }

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

  pdf.setFont('helvetica','normal');pdf.setFontSize(8);pdf.setTextColor(100,116,139);
  var dt=new Date();
  var dStr=dt.toLocaleString(undefined,{year:'numeric',month:'short',day:'2-digit',
                                         hour:'2-digit',minute:'2-digit'});
  pdf.text('Generated '+dStr,M,ph-30);
  pdf.text('Sources: LOCA2 downscaled CMIP6 \\u00b7 USGS Landsat \\u00b7 NOAA HRRR \\u00b7 CDC PLACES & SVI \\u00b7 Healthy Places Index',
    M,ph-18);

  var url='cccia-sio.github.io/socal-heat-hub';
  pdf.setFont('helvetica','bold');pdf.setTextColor(37,99,235);
  var urlW=pdf.getTextWidth(url);
  pdf.text(url,pw-M-urlW,ph-18);

  var ts=dt.toISOString().replace(/[:.]/g,'-').slice(0,19);
  var slug=opts.jurisName?('_'+opts.jurisName.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')):'';
  pdf.save('san_diego_heat_risk'+slug+'_'+ts+'.pdf');
}

// The map PDF's variable box: the inputs behind the build, by pillar, sized to
// its content and anchored by its bottom-left corner. Reads the COMMITTED inputs
// (results[pid]), not the `enabled` checkboxes — after a toggle the composite is
// stale but the map still shows the last build, and the list has to describe
// the map.
function _pdfVarBox(pdf,x,bottom,maxW,maxH){
  var pad=7, indent=9, bulletW=6;
  var groups=['exposure','sensitivity','adaptive'].map(function(pid){
    var r=results[pid], items=[];
    if(r)items=(pid==='exposure')?[r.label]:r.keys.map(function(k){return varLabel[k]||k;});
    return {pid:pid,items:items};
  });
  var maxN=Math.max.apply(null,groups.map(function(g){return g.items.length;}));

  // One pass both measures and paints, so the fit check can't drift from the
  // drawing. Paints only when given a top edge; returns the box size.
  function layout(fs,cap,top){
    var draw=(top!=null), lh=fs*1.3, by=(draw?top:0)+pad, w=0;
    function line(txt,dx,style,rgb,bullet){
      by+=lh;
      pdf.setFont('helvetica',style);
      w=Math.max(w,dx+pdf.getTextWidth(txt));
      if(!draw)return;
      pdf.setTextColor(rgb[0],rgb[1],rgb[2]);
      pdf.text(txt,x+pad+dx,by);
      if(bullet){
        pdf.setFont('helvetica','normal');pdf.setTextColor(100,116,139);
        pdf.text('\\u2022',x+pad+indent,by);
      }
    }
    pdf.setFontSize(fs+1.5);
    line('Selected Variables',0,'bold',[15,23,42]);
    pdf.setFontSize(fs);
    groups.forEach(function(g){
      by+=lh*0.6;
      var c=_hexToRgb(PMETA[g.pid].color), sq=fs*0.7;
      if(draw){pdf.setFillColor(c[0],c[1],c[2]);pdf.rect(x+pad,by+lh-sq,sq,sq,'F');}
      line(PMETA[g.pid].label,indent,'bold',c);
      if(!g.items.length){line('None selected',indent,'italic',[148,163,184]);return;}
      g.items.slice(0,cap).forEach(function(lbl){
        pdf.setFont('helvetica','normal');
        pdf.splitTextToSize(lbl,maxW-2*pad-indent-bulletW).forEach(function(seg,i){
          line(seg,indent+bulletW,'normal',[51,65,85],i===0);
        });
      });
      if(g.items.length>cap)line('+ '+(g.items.length-cap)+' more',indent+bulletW,'italic',[100,116,139]);
    });
    return {w:w+2*pad, h:by+pad-(draw?top:0)};
  }

  // Shrink the type first; if the smallest size still overflows, cap each
  // pillar's list. Overflow is stated, never silent.
  var sizes=[7,6.5,6,5.5], fs=sizes[sizes.length-1], cap=maxN;
  for(var i=0;i<sizes.length;i++){if(layout(sizes[i],maxN).h<=maxH){fs=sizes[i];break;}}
  while(cap>1&&layout(fs,cap).h>maxH)cap--;

  // Same white card and hairline as the in-map legend and scale bar.
  var box=layout(fs,cap), top=bottom-box.h;
  pdf.setFillColor(255,255,255);pdf.rect(x,top,box.w,box.h,'F');
  pdf.setDrawColor(207,209,213);pdf.setLineWidth(0.4);pdf.rect(x,top,box.w,box.h);
  layout(fs,cap,top);
}

function downloadMapPdf(){
  var btn=document.getElementById('dl-pdf-btn');
  btn.disabled=true;
  _pdfShow('Loading PDF library\\u2026');
  _ensurePdfLibs().then(function(){
    _pdfShow('Rendering study-area map\\u2026');
    // width fills _heroPdf's 720x462pt map area (1200*720/462). padLeft pushes
    // the coastline clear of its variable box, over the ocean in the lower-left.
    return _captureStudyAreaMap({width:1870,padLeft:140});
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
      '. If basemap tiles were blocked, allow server.arcgisonline.com, or use the browser\\u2019s Print \\u2192 Save as PDF.';
    bar.style.display='block';
    setTimeout(function(){bar.style.display='none';},9000);
  });
}

// Crisp gradient bar drawn as vector segments (jsPDF has no native gradients).
function _pdfGradientBar(pdf,x,y,w,h,stops){
  var n=64;
  for(var i=0;i<n;i++){
    var c=interpC(stops,i/(n-1));
    pdf.setFillColor(c[0],c[1],c[2]);
    pdf.rect(x+(i/n)*w,y,w/n+0.6,h,'F');
  }
  pdf.setDrawColor(180,188,200);pdf.setLineWidth(0.4);pdf.rect(x,y,w,h);
}

// City-scoped two-panel figure: composite risk (left) + dominant driver (right),
// mirroring the user-guide hero figure but framed and clipped to one city. Builds
// the composite first if it hasn't been computed yet.
function downloadCityPdf(){
  var name=focusedCity;
  if(!name||!jurisLayers[name]){return;}
  var btn=document.getElementById('cf-pdf-btn');
  if(btn)btn.disabled=true;
  function bail(msg){
    if(btn)btn.disabled=false;
    var b=document.getElementById('error-bar');
    b.textContent=msg;b.style.display='block';
    setTimeout(function(){b.style.display='none';},7000);
  }
  // Build the composite first if it hasn't been computed yet.
  if(!results){
    compute();
    if(!results){bail('Turn on at least one variable, then export.');return;}
  }
  var bounds=jurisLayers[name].getBounds();
  var prevLayer=curLayer, prevRaw=rawVar;
  function restore(){curLayer=prevLayer;rawVar=prevRaw;}
  _pdfShow('Loading PDF library\\u2026');
  _ensurePdfLibs().then(function(){
    _pdfShow('Rendering '+name+' \\u2014 composite\\u2026');
    rawVar=null;curLayer='composite';
    return _captureStudyAreaMap({bounds:bounds,jurisName:name,minimal:true});
  }).then(function(canvasL){
    _pdfShow('Rendering '+name+' \\u2014 dominant driver\\u2026');
    rawVar=null;curLayer='driver';
    return _captureStudyAreaMap({bounds:bounds,jurisName:name,minimal:true})
      .then(function(canvasR){return [canvasL,canvasR];});
  }).then(function(pair){
    _pdfShow('Building PDF\\u2026');
    restore();
    _heroPdfTwoPanel(pair[0],pair[1],{jurisName:name});
    _pdfHide();
    if(btn)btn.disabled=false;
  }).catch(function(err){
    restore();
    console.error('City PDF export failed:',err);
    _pdfHide();
    if(btn)btn.disabled=false;
    var bar=document.getElementById('error-bar');
    bar.textContent='PDF export failed: '+err.message+
      '. If basemap tiles were blocked, allow server.arcgisonline.com, or use the browser\\u2019s Print \\u2192 Save as PDF.';
    bar.style.display='block';
    setTimeout(function(){bar.style.display='none';},9000);
  });
}

function _heroPdfTwoPanel(cL,cR,opts){
  opts=opts||{};
  var jsPDF=window.jspdf.jsPDF;
  var pdf=new jsPDF({orientation:'landscape',unit:'pt',format:'letter',compress:true});
  var pw=pdf.internal.pageSize.getWidth();   // 792
  var ph=pdf.internal.pageSize.getHeight();  // 612
  var M=36;

  // ── Title band ─────────────────────────────────────────────
  pdf.setFillColor(247,249,252);pdf.rect(0,0,pw,72,'F');
  pdf.setFillColor(37,99,235);pdf.rect(0,72,pw,2.4,'F');
  pdf.setFont('helvetica','bold');pdf.setFontSize(20);pdf.setTextColor(15,23,42);
  pdf.text('San Diego Heat Risk Explorer',M,36);
  pdf.setFont('helvetica','normal');pdf.setFontSize(11);pdf.setTextColor(71,85,105);
  var lead=(opts.jurisName?opts.jurisName+'  \\u2014  ':'')+'Composite Risk & Dominant Driver';
  pdf.text(lead,M,55);
  pdf.setTextColor(120,130,148);
  pdf.text('  \\u2022  '+_scenarioLabel(),M+pdf.getTextWidth(lead),55);

  // ── Two panels side by side ────────────────────────────────
  var gap=22, capH=15, legH=40;
  var top=90, bottom=ph-58;
  var colW=(pw-2*M-gap)/2;
  var imgAreaH=(bottom-top)-capH-legH;
  var ratio=cL.width/cL.height;

  function placePanel(canvas,x0,caption,kind){
    // caption
    pdf.setFont('helvetica','bold');pdf.setFontSize(11);pdf.setTextColor(30,41,59);
    pdf.text(caption,x0+colW/2,top+11,{align:'center'});
    // image, centered in its area
    var imgW=colW, imgH=imgW/ratio;
    if(imgH>imgAreaH){imgH=imgAreaH;imgW=imgH*ratio;}
    var imgX=x0+(colW-imgW)/2;
    var imgY=top+capH+(imgAreaH-imgH)/2;
    pdf.setFillColor(225,228,235);pdf.rect(imgX+2,imgY+2,imgW,imgH,'F');
    pdf.addImage(canvas.toDataURL('image/png'),'PNG',imgX,imgY,imgW,imgH,undefined,'FAST');
    pdf.setDrawColor(203,213,225);pdf.setLineWidth(0.6);pdf.rect(imgX,imgY,imgW,imgH);
    // legend below image
    var ly=top+capH+imgAreaH+12;
    if(kind==='composite'){
      var barW=Math.min(colW*0.7,190), barX=x0+(colW-barW)/2;
      _pdfGradientBar(pdf,barX,ly,barW,8,pal('CR'));
      pdf.setFont('helvetica','normal');pdf.setFontSize(7.5);pdf.setTextColor(100,116,139);
      pdf.text('Lower',barX,ly+18);
      pdf.text('Higher',barX+barW,ly+18,{align:'right'});
    }else{
      var d=pal('DRGB');
      var items=[['Heat Exposure',d.exposure],['Pop. Sensitivity',d.sensitivity],['Adaptive Capacity',d.adaptive]];
      var sw=9, txtGap=4, itemGap=12;
      pdf.setFont('helvetica','normal');pdf.setFontSize(7.5);
      var total=0;
      items.forEach(function(it){total+=sw+txtGap+pdf.getTextWidth(it[0]);});
      total+=itemGap*(items.length-1);
      var ix=x0+(colW-total)/2, iy=ly;
      items.forEach(function(it){
        var c=it[1];
        pdf.setFillColor(c[0],c[1],c[2]);pdf.rect(ix,iy,sw,sw,'F');
        pdf.setTextColor(71,85,105);
        pdf.text(it[0],ix+sw+txtGap,iy+sw-1.5);
        ix+=sw+txtGap+pdf.getTextWidth(it[0])+itemGap;
      });
      pdf.setFontSize(6.5);pdf.setTextColor(140,150,165);
      pdf.text('Lighter shade = lower composite risk',x0+colW/2,ly+18,{align:'center'});
    }
  }
  placePanel(cL,M,'Composite Risk','composite');
  placePanel(cR,M+colW+gap,'Dominant Driver','driver');

  // ── Footer ─────────────────────────────────────────────────
  pdf.setDrawColor(226,232,240);pdf.setLineWidth(0.5);pdf.line(M,ph-46,pw-M,ph-46);
  pdf.setFontSize(8);pdf.setTextColor(100,116,139);
  var dt=new Date();
  var dStr=dt.toLocaleString(undefined,{year:'numeric',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'});
  pdf.text('Generated '+dStr,M,ph-30);
  pdf.text('Sources: LOCA2 downscaled CMIP6 \\u00b7 USGS Landsat \\u00b7 NOAA HRRR \\u00b7 CDC PLACES & SVI \\u00b7 Healthy Places Index',M,ph-18);
  var url='cccia-sio.github.io/socal-heat-hub';
  pdf.setFont('helvetica','bold');pdf.setTextColor(37,99,235);
  var urlW=pdf.getTextWidth(url);pdf.text(url,pw-M-urlW,ph-18);

  var ts=dt.toISOString().replace(/[:.]/g,'-').slice(0,19);
  var slug=opts.jurisName?('_'+opts.jurisName.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')):'';
  pdf.save('san_diego_heat_risk'+slug+'_'+ts+'.pdf');
}

/* ================================================================
   SINGLE-TRACT PDF
   ================================================================
   One-page portrait handout for one tract, reached from the "Export tract PDF"
   button in the tract popup: risk summary, the popup's rose plot, a satellite
   locator, and the top variables behind each pillar.

   Like the map and the other PDF exports (and unlike the bulk CSV/GeoJSON
   download), this is a RENDERED VIEW, not a dataset — so Healthy Places Index
   variables may appear here, used under HPI's own terms and credited in the
   footer. See DATA_LICENSE.md. */

var _TRACT_PDF_MAX_ROWS=8;   // per pillar; any overflow is stated, never silent

// 'rgb(r,g,b)' -> [r,g,b], so the PDF reuses _dtRiskColor's ramp rather than
// duplicating the stops.
function _rgbTriplet(s){
  var m=/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/.exec(String(s));
  return m?[+m[1],+m[2],+m[3]]:[148,163,184];
}
// PMETA's pillar colors are hex; jsPDF wants channels.
function _hexToRgb(h){
  var m=/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(String(h||'').trim());
  return m?[parseInt(m[1],16),parseInt(m[2],16),parseInt(m[3],16)]:[71,85,105];
}
function _ord(n){
  n=Math.round(n);
  var suf=['th','st','nd','rd'], v=n%100;
  return n+(suf[(v-20)%10]||suf[v]||suf[0]);
}
function _pdfTrunc(pdf,txt,maxW){
  txt=String(txt==null?'':txt);
  if(pdf.getTextWidth(txt)<=maxW)return txt;
  while(txt.length>1&&pdf.getTextWidth(txt+'\\u2026')>maxW)txt=txt.slice(0,-1);
  return txt+'\\u2026';
}

// Rasterize an SVG string to a canvas. Note the rose plot is re-parsed as
// STRICT XML once it becomes an <img> — HTML's lenient parser is not in play —
// so every label radarSVG() injects must be run through _addrEsc first. A bare
// "&" (e.g. the "Age 17 & Under" label) would otherwise fail the load silently.
function _svgToCanvas(svg,w,h,scale){
  return new Promise(function(resolve,reject){
    var img=new Image();
    img.onload=function(){
      var c=document.createElement('canvas');
      c.width=Math.round(w*scale);c.height=Math.round(h*scale);
      var cx=c.getContext('2d');
      cx.fillStyle='#ffffff';cx.fillRect(0,0,c.width,c.height);
      cx.drawImage(img,0,0,c.width,c.height);
      resolve(c);
    };
    img.onerror=function(){reject(new Error('the rose plot could not be rendered'));};
    img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg);
  });
}
// Pin the SVG's intrinsic size to the target pixel size so browsers rasterize
// at full resolution instead of rendering at 336px and upscaling.
function _radarCanvas(fi,pr,scale){
  var S=scale||3, W=400, H=320;   // W/H = radarSVG's viewBox
  var svg=radarSVG(fi,pr).replace('width="336" height="269"',
    'width="'+(W*S)+'" height="'+(H*S)+'"');
  return _svgToCanvas(svg,W,H,S);
}

// Composite the SANDAG land-use parcels onto the inset canvas. protomaps-leaflet
// draws each tile into its own <canvas> inside a Leaflet GridLayer, so we let it
// render off-screen and then blit those tiles across. Position comes from the
// DOM rects rather than re-deriving Leaflet's tile math — the layer may use a
// different tile size / levelDiff than the raster grid, and the rects are
// already exactly where Leaflet put them.
// Resolves false when protomaps is unavailable or the PMTiles fetch yields
// nothing (e.g. file:// preview, where Range requests fail), so the caller can
// drop the panel instead of emitting a blank one.
function _pdfPaintLanduse(pm,ctx,S){
  return new Promise(function(resolve){
    if(!window.protomapsL){resolve(false);return;}
    if(!pm.getPane('landusePane')){
      pm.createPane('landusePane');
      pm.getPane('landusePane').style.zIndex=450;
    }
    var layer=_buildLanduseLayer();
    if(!layer){resolve(false);return;}
    var settled=false;
    function finish(){
      if(settled)return;
      settled=true;
      var painted=false;
      try{
        // Blit onto a scratch canvas first, not straight onto the basemap: a
        // failed PMTiles fetch still leaves protomaps with blank-but-valid tile
        // canvases, so "we drew some tiles" is NOT evidence that any parcels
        // rendered. Only the pixels can tell us, and once they're composited
        // over the basemap we can no longer distinguish them from it.
        var sc=document.createElement('canvas');
        sc.width=ctx.canvas.width;sc.height=ctx.canvas.height;
        var sctx=sc.getContext('2d');
        var mr=pm.getContainer().getBoundingClientRect();
        var tiles=layer._tiles||{};
        Object.keys(tiles).forEach(function(k){
          var el=tiles[k]&&tiles[k].el;
          if(!el||!el.width||!el.height)return;
          var r=el.getBoundingClientRect();
          if(!r.width||!r.height)return;
          try{sctx.drawImage(el,(r.left-mr.left)*S,(r.top-mr.top)*S,r.width*S,r.height*S);}catch(e){}
        });
        // Parcels (incl. water) blanket the SANDAG extent, so a real render
        // covers most of the frame. Sample the alpha channel; a blank layer
        // scores ~0. The scratch canvas only ever holds protomaps' own vector
        // output, so it is never tainted and getImageData is safe.
        var d=sctx.getImageData(0,0,sc.width,sc.height).data;
        var seen=0,hit=0;
        for(var i=3;i<d.length;i+=4*17){seen++;if(d[i]>8)hit++;}
        painted=(seen>0&&hit>seen*0.02);
        if(painted)ctx.drawImage(sc,0,0);
      }catch(e){painted=false;}
      try{pm.removeLayer(layer);}catch(e){}
      resolve(painted);
    }
    // GridLayer fires 'load' once every visible tile is done; give the last
    // canvas paint a frame to land before we read the tiles back.
    layer.on('load',function(){setTimeout(finish,150);});
    try{layer.addTo(pm);}catch(e){resolve(false);return;}
    setTimeout(finish,8000);   // hard cap — a stalled PMTiles fetch must not hang the export
  });
}

// Locator inset for one tract, in either of two views:
//   'sat' — Esri World Imagery
//   'lu'  — Esri light basemap + SANDAG land-use parcels (protomaps/PMTiles)
// Same approach as _captureStudyAreaMap: Leaflet is used purely as a projector
// and the raster tiles are drawn by hand, because html2canvas cannot composite
// Leaflet's tile and vector panes in alignment. Resolves to null (rather than
// rejecting) when the basemap is blocked, so a firewalled network loses the
// panel but still gets the PDF.
function _captureTractInset(feat,mode){
  mode=mode||'sat';
  return new Promise(function(resolve){
    // Wide format: the two insets stack in the PDF's right column, so each is
    // short and wide. Aspect matches that slot (232pt / 107.5pt) so the image
    // fills it without letterboxing. A wide frame isn't wasted space — it shows
    // more of the surrounding neighbourhood, which is what a locator is for.
    var W=680,H=315,S=2,TILE=256;
    var div=document.createElement('div');
    div.style.cssText='position:absolute;top:-99999px;left:-99999px;width:'+W+'px;height:'+H+'px;';
    document.body.appendChild(div);
    var pm=L.map(div,{zoomControl:false,attributionControl:false,
      fadeAnimation:false,zoomAnimation:false});
    function done(v){
      try{pm.remove();}catch(e){}
      try{div.parentNode.removeChild(div);}catch(e){}
      resolve(v);
    }
    var b;
    try{b=L.geoJSON(feat).getBounds();}catch(e){done(null);return;}
    if(!b||!b.isValid()){done(null);return;}
    pm.fitBounds(b,{padding:[30,12],maxZoom:17,animate:false});
    var z=pm.getZoom(), pb=pm.getPixelBounds();

    var canvas=document.createElement('canvas');
    canvas.width=W*S;canvas.height=H*S;
    var ctx=canvas.getContext('2d');
    ctx.fillStyle='#e8eaed';ctx.fillRect(0,0,canvas.width,canvas.height);
    function project(lat,lng){var p=pm.latLngToContainerPoint([lat,lng]);return[p.x*S,p.y*S];}
    function loadImage(src){
      return new Promise(function(res){
        var im=new Image();im.crossOrigin='anonymous';
        im.onload=function(){res(im);};im.onerror=function(){res(null);};
        im.src=src;
      });
    }
    function tileUrl(z,x,y){
      // Esri World Imagery is ordered {z}/{y}/{x}, not the usual {z}/{x}/{y}.
      return 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/'+
             'MapServer/tile/'+z+'/'+y+'/'+x;
    }
    var tasks=[], got={n:0};
    if(mode==='lu'){
      // Two rasters and a native-zoom ceiling to respect — _paintLightBasemap
      // owns that math for both PDF blitters.
      tasks.push(_paintLightBasemap(ctx,z,pb,S).then(function(n){got.n+=n;}));
    }else{
      var minTX=Math.floor(pb.min.x/TILE),maxTX=Math.floor(pb.max.x/TILE);
      var minTY=Math.floor(pb.min.y/TILE),maxTY=Math.floor(pb.max.y/TILE);
      var nW=Math.pow(2,z);
      for(var tx=minTX;tx<=maxTX;tx++){
        for(var ty=minTY;ty<=maxTY;ty++){
          if(ty<0||ty>=nW)continue;
          var wx=((tx%nW)+nW)%nW;
          var url=tileUrl(z,wx,ty);
          var dx=(tx*TILE-pb.min.x)*S, dy=(ty*TILE-pb.min.y)*S;
          (function(x,y,u){
            tasks.push(loadImage(u).then(function(im){
              if(im){got.n++;ctx.drawImage(im,x,y,TILE*S,TILE*S);}
            }));
          })(dx,dy,url);
        }
      }
    }
    Promise.all(tasks).then(function(){
      if(!got.n){done(null);return;}   // basemap blocked — drop the panel
      // Parcels go on top of the light base. If they don't render there is no
      // land-use panel to show, so drop it rather than pass off a plain basemap.
      return (mode==='lu')?_pdfPaintLanduse(pm,ctx,S):true;
    }).then(function(ok){
      if(ok===undefined)return;        // already resolved null above
      if(!ok){done(null);return;}
      var g=feat.geometry;
      if(!g){done(canvas);return;}
      function addRings(rings){
        rings.forEach(function(r){
          for(var i=0;i<r.length;i++){
            var p=project(r[i][1],r[i][0]);   // GeoJSON is [lng,lat]
            if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);
          }
          ctx.closePath();
        });
      }
      ctx.beginPath();
      if(g.type==='Polygon')addRings(g.coordinates);
      else if(g.type==='MultiPolygon')g.coordinates.forEach(addRings);
      else{done(canvas);return;}
      // No wash over the land-use view: a yellow tint would shift every parcel
      // colour away from its legend swatch.
      if(mode!=='lu'){ctx.globalAlpha=0.12;ctx.fillStyle='#fde047';ctx.fill('evenodd');}
      ctx.globalAlpha=1;ctx.lineJoin='round';
      ctx.strokeStyle='#0f172a';ctx.lineWidth=6;ctx.stroke();   // halo for contrast
      ctx.strokeStyle='#fde047';ctx.lineWidth=2.6;ctx.stroke();
      done(canvas);
    }).catch(function(){done(null);});
  });
}

// Compact land-use key for the PDF, laid out as a single row where it fits and
// wrapped when it doesn't. Returns the y below the key.
function _pdfLuLegend(pdf,x,y,maxW){
  var cats=LU_CATS.concat([LU_WATER_CAT]);
  var SW=5, PAD=3, GAP=9, LH=9;
  pdf.setFont('helvetica','normal');pdf.setFontSize(6);
  var cx=x, cy=y;
  cats.forEach(function(c){
    var tw=pdf.getTextWidth(c.name), iw=SW+PAD+tw;
    if(cx>x&&cx+iw>x+maxW){cx=x;cy+=LH;}
    var rgb=_hexToRgb(c.color);   // LU_CATS are hex; _rgbTriplet only reads rgb()
    pdf.setFillColor(rgb[0],rgb[1],rgb[2]);
    pdf.rect(cx,cy-SW+1,SW,SW,'F');
    pdf.setDrawColor(203,213,225);pdf.setLineWidth(0.3);
    pdf.rect(cx,cy-SW+1,SW,SW);
    pdf.setTextColor(100,116,139);
    pdf.text(c.name,cx+SW+PAD,cy);
    cx+=iw+GAP;
  });
  return cy+3;
}

// Variable rows for one pillar, in the same order the rose plot ranks them
// (PCA contribution), capped at _TRACT_PDF_MAX_ROWS.
function _tractPdfPillar(pid,idx,pr){
  var keys=dtPillarKeys(pid)||[], rows=[];
  keys.forEach(function(key){
    var meta=null;
    for(var j=0;j<VM.length;j++){if(VM[j].key===key){meta=VM[j];break;}}
    if(!meta)return;
    var rawV=pr[key];
    if(rawV==null||(typeof rawV==='number'&&isNaN(rawV))){
      rows.push({label:meta.label,val:'no data',pct:null,worse:true});return;
    }
    var dArr=getDeltaArr(key);
    var disp=dArr?getAdj(key,idx):+rawV;
    rows.push({
      label:meta.label,
      val:fmtCmpRaw(key,disp),
      pct:(rawPct[key]&&rawPct[key][idx]!=null)?rawPct[key][idx]:null,
      worse:(POLARITY&&POLARITY[key]!==false)
    });
  });
  return {rows:rows.slice(0,_TRACT_PDF_MAX_ROWS),
          omitted:Math.max(0,rows.length-_TRACT_PDF_MAX_ROWS)};
}

function _tractPdfBuild(idx,feat,radarC,mapC,luC){
  var pr=feat.properties, geoid=pr.GEOID;
  var jsPDF=window.jspdf.jsPDF;
  var pdf=new jsPDF({orientation:'portrait',unit:'pt',format:'letter',compress:true});
  var pw=pdf.internal.pageSize.getWidth();    // 612
  var ph=pdf.internal.pageSize.getHeight();   // 792
  var M=40, CW=pw-2*M;

  _ensureTractJuris();
  var jn=_tractJurisName[idx];
  var place=jn?('City of '+jn):'Unincorporated San Diego County';

  // ── Title band ───────────────────────────────────────────────
  pdf.setFillColor(247,249,252);pdf.rect(0,0,pw,64,'F');
  pdf.setFillColor(37,99,235);pdf.rect(0,64,pw,2.4,'F');
  pdf.setFont('helvetica','bold');pdf.setFontSize(17);pdf.setTextColor(15,23,42);
  pdf.text('Census Tract '+geoid.slice(-6),M,30);
  pdf.setFont('helvetica','normal');pdf.setFontSize(10);pdf.setTextColor(71,85,105);
  pdf.text(place+'  \\u00b7  GEOID '+geoid,M,46);
  pdf.setFontSize(9);pdf.setTextColor(120,130,148);
  pdf.text(_scenarioLabel(),M,58);
  pdf.setFont('helvetica','bold');pdf.setFontSize(9.5);pdf.setTextColor(37,99,235);
  var wm='San Diego Heat Risk Explorer';
  pdf.text(wm,pw-M-pdf.getTextWidth(wm),30);

  // ── Risk summary cards ───────────────────────────────────────
  var y=82, chh=44, cgap=8, cw=(CW-cgap*3)/4;
  // Adaptive is flipped to a capacity deficit so every card is risk-oriented,
  // matching the rose plot printed directly below it (and the detail modal).
  // Accent bars carry the PILLAR colour (PMETA) — the same hue as that pillar's
  // section header below and its rose-plot wedge. Colouring them by risk instead
  // put a red bar on "Adaptive Cap." while its own header was green.
  [['Composite',pr._comp,_DT_CARD_ACCENT],
   ['Heat Exposure',pr._exposure,PMETA.exposure.color],
   ['Sensitivity',pr._sensitivity,PMETA.sensitivity.color],
   ['Adaptive Cap.',(pr._adaptive==null?null:100-pr._adaptive),PMETA.adaptive.color]
  ].forEach(function(c,i){
    var x=M+i*(cw+cgap), v=c[1];
    pdf.setFillColor(248,250,252);pdf.rect(x,y,cw,chh,'F');
    if(v!=null){
      var col=_hexToRgb(c[2]);
      pdf.setFillColor(col[0],col[1],col[2]);pdf.rect(x,y,3,chh,'F');
    }
    pdf.setFont('helvetica','normal');pdf.setFontSize(7);pdf.setTextColor(100,116,139);
    pdf.text(c[0].toUpperCase(),x+9,y+15);
    pdf.setFont('helvetica','bold');pdf.setFontSize(16);pdf.setTextColor(15,23,42);
    var vs=(v!=null)?String(Math.round(v)):'\\u2014';
    // Measure at the size it's actually drawn in — getTextWidth reads the
    // CURRENT font, so measuring after the switch below would under-report and
    // print "/100" on top of the number.
    var vsW=pdf.getTextWidth(vs);
    pdf.text(vs,x+9,y+35);
    if(v!=null){
      pdf.setFont('helvetica','normal');pdf.setFontSize(7.5);pdf.setTextColor(148,163,184);
      pdf.text('/100',x+9+vsW+3,y+35);
    }
  });
  y+=chh+13;
  if(pr._drv){
    pdf.setFont('helvetica','normal');pdf.setFontSize(9);pdf.setTextColor(71,85,105);
    pdf.text('Dominant driver:',M,y);
    var lw=pdf.getTextWidth('Dominant driver:')+4;
    pdf.setFont('helvetica','bold');pdf.setTextColor(15,23,42);
    // PMETA label, matching the Dominant Driver legend and the detail modal —
    // this printed the raw pillar key ("exposure") on the handout.
    pdf.text((PMETA[pr._drv]&&PMETA[pr._drv].label)||String(pr._drv),M+lw,y);
  }else if(pr._comp==null){
    pdf.setFont('helvetica','italic');pdf.setFontSize(8.5);pdf.setTextColor(148,163,184);
    pdf.text('No composite built \\u2014 scores omitted. The readings below are unaffected.',M,y);
  }
  y+=10;

  // ── Rose plot + satellite locator, side by side ──────────────
  // The rose plot drives the row height: it's the figure people read first, so
  // it gets the width it needs. The right column stacks the satellite locator
  // over the SANDAG land-use view, sharing that height between them. Either
  // capture can come back null (imagery blocked, PMTiles unreachable), so the
  // survivors just take the whole column.
  var panTop=y+8;
  var roseW=288, rx=M+300, rw=CW-300;
  var panH=radarC?roseW*(radarC.height/radarC.width):230;
  var insets=[];
  if(mapC)insets.push({c:mapC,label:'Tract location \\u2014 satellite'});
  if(luC) insets.push({c:luC, label:'Land use (SANDAG)'});
  pdf.setFont('helvetica','bold');pdf.setFontSize(8.5);pdf.setTextColor(71,85,105);
  pdf.text('Heat-vulnerability profile',M,panTop);
  if(insets.length)pdf.text(insets[0].label,rx,panTop);
  var imgTop=panTop+7;
  if(radarC){
    pdf.addImage(radarC.toDataURL('image/png'),'PNG',M,imgTop,roseW,panH,undefined,'FAST');
  }
  // The first caption already sits on the panel header row; later ones are
  // drawn inline above their own image, so each slot loses CAPH+IGAP.
  var CAPH=9, IGAP=6;
  var slotH=insets.length?(panH-(insets.length-1)*(CAPH+IGAP))/insets.length:0;
  var iy=imgTop;
  insets.forEach(function(p,i){
    if(i>0){
      pdf.setFont('helvetica','bold');pdf.setFontSize(8.5);pdf.setTextColor(71,85,105);
      pdf.text(p.label,rx,iy+CAPH-2);
      iy+=CAPH;
    }
    var a=p.c.width/p.c.height;
    var w=rw, h=w/a;
    if(h>slotH){h=slotH;w=h*a;}
    var dx=rx+(rw-w)/2, dy=iy+(slotH-h)/2;
    pdf.addImage(p.c.toDataURL('image/png'),'PNG',dx,dy,w,h,undefined,'FAST');
    pdf.setDrawColor(203,213,225);pdf.setLineWidth(0.6);pdf.rect(dx,dy,w,h);
    iy+=slotH+IGAP;
  });
  y=imgTop+panH+12;
  if(luC)y=_pdfLuLegend(pdf,M,y,CW)+4;

  // ── Variables behind each pillar ─────────────────────────────
  var pillars=['exposure','sensitivity','adaptive'].map(function(pid){
    var d=_tractPdfPillar(pid,idx,pr);
    var hdr=PMETA[pid].label;
    if(pid==='exposure'){var md=heatModeDef(_builtSettings().heatMode);if(md)hdr+=' \\u2014 '+md.label;}
    return {pid:pid,hdr:hdr,rows:d.rows,omitted:d.omitted};
  }).filter(function(p){return p.rows.length;});

  // Fit the rows into whatever vertical room is left, rather than assuming.
  var footTop=ph-62;
  var hdrH=15, secGap=7, noteH=10;
  var nRows=0, fixed=0;
  pillars.forEach(function(p){
    nRows+=p.rows.length;
    fixed+=hdrH+secGap+(p.omitted?noteH:0);
  });
  // Let the rows breathe into whatever room is left rather than stranding a
  // band of white above the footer — a tract with 3 enabled variables per
  // pillar leaves far more space than one with 8. Clamped both ways: 16pt so
  // sparse tracts don't look double-spaced, 9.5pt so full ones still fit.
  var avail=footTop-y-4;
  var rowH=nRows?(avail-fixed)/nRows:13;
  if(rowH>16)rowH=16;
  if(rowH<9.5)rowH=9.5;

  var valR=M+300, barX=M+316, barW=118, pctR=M+CW;
  pillars.forEach(function(p){
    var pm=PMETA[p.pid];
    pdf.setFont('helvetica','bold');pdf.setFontSize(9);
    var hc=_hexToRgb(pm.color);
    pdf.setTextColor(hc[0],hc[1],hc[2]);
    pdf.text(p.hdr.toUpperCase(),M,y+8);
    pdf.setDrawColor(226,232,240);pdf.setLineWidth(0.5);
    pdf.line(M,y+11.5,M+CW,y+11.5);
    y+=hdrH;
    p.rows.forEach(function(r){
      var ty=y+rowH-3.5;
      pdf.setFont('helvetica','normal');pdf.setFontSize(8.5);pdf.setTextColor(51,65,85);
      pdf.text(_pdfTrunc(pdf,r.label,196),M,ty);
      pdf.setFont('helvetica','bold');pdf.setTextColor(15,23,42);
      pdf.text(String(r.val),valR-pdf.getTextWidth(String(r.val)),ty);
      if(r.pct!=null){
        var p0=Math.max(0,Math.min(100,r.pct));
        // Bar length AND the printed rank are the RISK percentile, so a
        // protective variable reconciles with the rose plot above (which plots
        // capacity as a deficit). The colour was already risk-oriented.
        var risk=r.worse?p0:(100-p0);
        var rc=_rgbTriplet(_dtRiskColor(risk));
        var by=y+rowH/2-2.5;
        pdf.setFillColor(232,236,241);pdf.rect(barX,by,barW,5,'F');
        pdf.setFillColor(rc[0],rc[1],rc[2]);
        pdf.rect(barX,by,Math.max(1.5,barW*risk/100),5,'F');
        pdf.setFont('helvetica','normal');pdf.setFontSize(7.5);pdf.setTextColor(100,116,139);
        var pt=_ord(risk)+' pct risk';
        pdf.text(pt,pctR-pdf.getTextWidth(pt),ty);
      }else{
        pdf.setFont('helvetica','normal');pdf.setFontSize(7.5);pdf.setTextColor(148,163,184);
        pdf.text('\\u2014',pctR-pdf.getTextWidth('\\u2014'),ty);
      }
      y+=rowH;
    });
    if(p.omitted){
      pdf.setFont('helvetica','italic');pdf.setFontSize(7.5);pdf.setTextColor(148,163,184);
      pdf.text('+ '+p.omitted+' more selected variable'+(p.omitted===1?'':'s')+
               ' not shown (top '+_TRACT_PDF_MAX_ROWS+' by contribution)',M,y+6);
      y+=noteH;
    }
    y+=secGap;
  });

  // ── Footer ───────────────────────────────────────────────────
  pdf.setDrawColor(226,232,240);pdf.setLineWidth(0.5);pdf.line(M,footTop,pw-M,footTop);
  var dt=new Date();
  var dStr=dt.toLocaleString(undefined,{year:'numeric',month:'short',day:'2-digit',
    hour:'2-digit',minute:'2-digit'});
  pdf.setFont('helvetica','normal');pdf.setFontSize(7);pdf.setTextColor(100,116,139);
  pdf.text('Generated '+dStr+'  \\u00b7  '+EXPORT_META.license+
           '  \\u00b7  Percentiles are ranked within San Diego County.',M,footTop+12);
  pdf.text('San Diego Heat Risk Explorer \\u2014 Southern California Extreme Heat Research Hub '+
           '& San Diego Regional Climate Collaborative.',M,footTop+21);
  // Wrap rather than assume: the source list plus the URL overruns the right
  // margin at this size if written as one line.
  pdf.setTextColor(130,140,155);
  var srcTxt='Sources incl. MesoWest/Synoptic, NOAA HRRR, USGS Landsat, CDC PLACES & SVI, '+
             'Healthy Places Index, U.S. Census.'+
             (mapC?' Imagery \\u00a9 Esri, Maxar, Earthstar Geographics.':'')+
             (luC?' Land use \\u00a9 SANDAG; basemap Esri, HERE, Garmin, \\u00a9 OpenStreetMap contributors.':'')+
             ' Full list: '+EXPORT_META.sources_url;
  var fy=footTop+30;
  pdf.splitTextToSize(srcTxt,CW).forEach(function(line){
    pdf.text(line,M,fy);fy+=8;
  });

  pdf.save('san_diego_heat_risk_tract_'+geoid+'.pdf');
}

function downloadTractPdf(geoid){
  if(!geoid)return;
  var idx=(window._gIdx&&window._gIdx[geoid]!=null)?window._gIdx[geoid]:null;
  if(idx==null){
    for(var i=0;i<GJ.features.length;i++){
      if(GJ.features[i].properties.GEOID===geoid){idx=i;break;}
    }
  }
  if(idx==null||!GJ.features[idx])return;
  var feat=GJ.features[idx];
  // Pillar scores come from the composite. Build it if the user hasn't — but
  // don't bail if they've enabled nothing: the rose plot and the raw readings
  // are still worth exporting, and the PDF says the scores are missing.
  if(!results){try{compute();}catch(e){}}

  var radarC=null, satC=null;
  _pdfShow('Loading PDF library\\u2026');
  _ensurePdfLibs().then(function(){
    _pdfShow('Drawing the heat-vulnerability profile\\u2026');
    return _radarCanvas(idx,feat.properties,3);
  }).then(function(c){
    radarC=c;
    _pdfShow('Fetching satellite imagery\\u2026');
    return _captureTractInset(feat,'sat');
  }).then(function(c){
    satC=c;
    _pdfShow('Rendering SANDAG land use\\u2026');
    return _captureTractInset(feat,'lu');
  }).then(function(luC){
    _pdfShow('Building PDF\\u2026');
    _tractPdfBuild(idx,feat,radarC,satC,luC);
    _pdfHide();
  }).catch(function(err){
    console.error('Tract PDF export failed:',err);
    _pdfHide();
    var bar=document.getElementById('error-bar');
    bar.textContent='Tract PDF export failed: '+err.message+
      '. You can also use the browser\\u2019s Print \\u2192 Save as PDF.';
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
  if(_dataLayer && RASTER_META[_dataLayer]) setDataLayer(_dataLayer);  // swap raster cb/default variant
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
function _dictWanted(){
  var b=document.getElementById('exp-incl-dict');
  return b?!!b.checked:true;
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
  // Some sources bar redistribution (e.g. HPI is non-commercial); their columns
  // are omitted from the download though they stay visible on the map.
  var skip={}; (NO_EXPORT||[]).forEach(function(k){skip[k]=1;});
  var cols=[{key:'GEOID',label:'GEOID'},{key:'jurisdiction',label:'jurisdiction'}];
  VM.forEach(function(v){if(!skip[v.key])cols.push({key:v.key,label:v.key,vm:true});});
  var inclPctile=true;
  var pctileBox=document.getElementById('exp-incl-pctile');
  if(pctileBox)inclPctile=!!pctileBox.checked;
  if(inclPctile){
    VM.forEach(function(v){
      if(!skip[v.key])cols.push({key:v.key+'_sd_pctile',label:v.key+'_sd_pctile',pctOf:v.key});
    });
  }
  var hasScn=(curScenario!=='present'&&curPeriod);
  if(hasScn){
    VM.forEach(function(v){
      if(DELTA_MAP[v.key]&&!skip[v.key])cols.push({key:v.key+'_adj',label:v.key+'_adj',adjOf:v.key});
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
  // Attribution, repeated on every row, so the file carries its license and a
  // link to the per-variable source table once it leaves this site.
  Object.keys(EXPORT_META).forEach(function(k){cols.push({key:k,label:k,meta:true});});
  return cols;
}

function _rowValueFor(col,i,pr){
  if(col.meta)return EXPORT_META[col.key];
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
  // Non-standard top-level member. GIS clients ignore it, but it survives a
  // plain JSON read, so the file names its own fields and sources in-band.
  return JSON.stringify({type:'FeatureCollection',
    metadata:_exportManifest(cols,indices.length),
    features:feats});
}

/* ================================================================
   DATA DICTIONARY  (ships with every download)
   ================================================================
   Column headers in the data file are raw keys — EP_NOINT, shututility_prev,
   houseing_pctile — with no units, no polarity and no provenance. Away from
   the map that file is not interpretable, so every download is accompanied by
   this dictionary and GeoJSON embeds it at metadata.fields. */

var _DICT_COLS=['column','label','pillar','group','units','higher_value_means',
                'source','source_url','in_download','notes'];
var _PILLAR_LABEL={exposure:'Heat Exposure',sensitivity:'Population Sensitivity',
                   adaptive:'Adaptive Capacity'};
var _SRC_TIGER='https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html';
var _SRC_LOCA='https://loca.ucsd.edu/';
var _SRC_SELF='https://cccia-sio.github.io/socal-heat-hub/';

// Columns the viewer derives rather than reads from a source table.
var _DICT_DERIVED={
  GEOID:{label:'Census tract identifier',group:'Geography',
    units:'11-digit 2020 Census FIPS code (state+county+tract)',
    source:'U.S. Census Bureau TIGER/Line 2020',source_url:_SRC_TIGER,
    notes:'Join key. Read as text \\u2014 the leading zero is significant.'},
  jurisdiction:{label:'Jurisdiction',group:'Geography',units:'city or tribal land name',
    source:'SANDAG / SanGIS jurisdiction boundaries',
    source_url:'https://sdgis-sandag.opendata.arcgis.com/',
    notes:'Whichever boundary contains the tract centroid. Blank = unincorporated county.'},
  scenario:{label:'Climate scenario',group:'Scenario',units:'SSP label',
    source:'LOCA2 downscaled CMIP6',source_url:_SRC_LOCA,
    notes:'Only present when a future scenario is active.'},
  period:{label:'Scenario period',group:'Scenario',units:'year range',
    source:'LOCA2 downscaled CMIP6',source_url:_SRC_LOCA,
    notes:'Only present when a future scenario is active.'},
  composite_risk:{label:'Composite heat risk',group:'Index output',
    units:'0-100 (mean of the pillar percentiles)',
    higher_value_means:'higher value = HIGHER heat risk',
    source:'Computed by this tool',source_url:_SRC_SELF,
    notes:'Mean of the active pillar scores, with Adaptive Capacity inverted '+
          '(100 minus its score). Not re-ranked. Depends on the heat mode and variables the map was '+
          'built from (listed in the file header) \\u2014 it is not a fixed published index.'},
  composite_driver:{label:'Dominant pillar',group:'Index output',
    units:'exposure | sensitivity | adaptive',
    source:'Computed by this tool',source_url:_SRC_SELF,
    notes:'Whichever of the three pillar scores is highest on the risk side for this tract.'},
  exposure_score:{label:'Heat Exposure score',group:'Index output',
    units:'percentile within San Diego County (0-100)',
    higher_value_means:'higher value = HIGHER heat risk',
    source:'Computed by this tool',source_url:_SRC_SELF,
    notes:'Mean of the z-scored member layers of the selected heat mode, re-ranked '+
          'to 0-100. Not a PCA. See the heat_mode field in this dictionary header.'},
  sensitivity_score:{label:'Population Sensitivity score',group:'Index output',
    units:'percentile within San Diego County (0-100)',
    higher_value_means:'higher value = HIGHER heat risk',
    source:'Computed by this tool',source_url:_SRC_SELF,
    notes:'Variance-weighted blend of the retained principal components (eigenvalue > 1) '+
          'of the sensitivity variables in the build, re-ranked to 0-100.'},
  adaptive_score:{label:'Adaptive Capacity score',group:'Index output',
    units:'percentile within San Diego County (0-100)',
    higher_value_means:'higher value = MORE capacity = LOWER heat risk',
    source:'Computed by this tool',source_url:_SRC_SELF,
    notes:'CAUTION: stored capacity-side, the OPPOSITE direction from exposure_score '+
          'and sensitivity_score. 100 = most adaptive capacity = least risk. '+
          'composite_risk already flips it; use 100 - adaptive_score to put this '+
          'column on the risk side.'},
  citation:{label:'Citation',group:'Attribution',units:'text',
    source:'Computed by this tool',source_url:_SRC_SELF,
    notes:'Repeated on every row so a row that travels alone keeps its attribution.'},
  license:{label:'License',group:'Attribution',units:'text',
    source:'Computed by this tool',source_url:'https://creativecommons.org/licenses/by/4.0/',
    notes:'Applies to this download. Individual sources carry their own terms \\u2014 see source_url per row.'},
  sources_url:{label:'Source list URL',group:'Attribution',units:'text',
    source:'Computed by this tool',source_url:_SRC_SELF,
    notes:'Human-readable source table in the User Guide.'}
};

function _dictRow(o){
  return {column:o.column||'',label:o.label||'',pillar:o.pillar||'',
          group:o.group||'',units:o.units||'',
          higher_value_means:o.higher_value_means||'',
          source:o.source||'',source_url:o.source_url||'',
          in_download:(o.in_download||'yes'),notes:o.notes||''};
}
function _dictPolarity(key){
  if(POLARITY[key]===undefined)return '';
  return POLARITY[key]?'higher value = HIGHER heat risk'
                      :'higher value = LOWER heat risk';
}
function _dictVarRow(v,over){
  var url=VSRC[v.key]||'';
  var r=_dictRow({column:v.key,label:v.label,pillar:_PILLAR_LABEL[v.pillar]||v.pillar,
    group:v.sub||'',units:VUNITS[v.key]||'',higher_value_means:_dictPolarity(v.key),
    source:SRCNAME[url]||'',source_url:url});
  if(over)for(var k in over)r[k]=over[k];
  return r;
}

// One row per column in the data file, in the same order, then one row per
// variable that is on the map but withheld from the download.
function _buildDataDictionary(cols){
  var vmByKey={};VM.forEach(function(v){vmByKey[v.key]=v;});
  var nTracts=GJ.features.length;
  var rows=[];
  cols.forEach(function(c){
    if(_DICT_DERIVED[c.key]){
      var d=_DICT_DERIVED[c.key];
      rows.push(_dictRow({column:c.label,label:d.label,group:d.group,units:d.units,
        higher_value_means:d.higher_value_means,source:d.source,
        source_url:d.source_url,notes:d.notes}));
      return;
    }
    if(c.pctOf){
      var vp=vmByKey[c.pctOf];if(!vp)return;
      rows.push(_dictVarRow(vp,{column:c.label,
        label:vp.label+' \\u2014 county percentile rank',
        units:'percentile within San Diego County (0-100)',
        notes:'Rank of the raw '+c.pctOf+' value against all '+nTracts+
              ' county tracts \\u2014 not just the tracts in this file, and not '+
              'recomputed for a jurisdiction subset. Ties take the average rank. '+
              'This is the value the map colours. Ranked on the RAW value, so the '+
              'higher_value_means column applies here too.'}));
      return;
    }
    if(c.adjOf){
      var va=vmByKey[c.adjOf];if(!va)return;
      rows.push(_dictVarRow(va,{column:c.label,
        label:va.label+' \\u2014 scenario-adjusted',
        notes:'Raw '+c.adjOf+' plus the LOCA2 '+(DELTA_MAP[c.adjOf]||'')+' delta for '+
              curScenario+' '+(curPeriod||'')+'. A uniform per-tract temperature '+
              'offset applied to the observed present-day value; the underlying '+
              'observation is unchanged. LOCA2: '+_SRC_LOCA}));
      return;
    }
    var v=vmByKey[c.key];
    if(v)rows.push(_dictVarRow(v,{column:c.label}));
  });
  // Withheld variables, so an absent column is explained rather than mysterious.
  (NO_EXPORT||[]).forEach(function(k){
    var v=vmByKey[k];if(!v)return;
    rows.push(_dictVarRow(v,{in_download:'no',
      notes:(NO_EXPORT_WHY&&NO_EXPORT_WHY[k])||
            'Shown on the map but omitted from this download: this project '+
            'cannot establish a right to redistribute it. Get it from the source.'}));
  });
  return rows;
}

// Provenance header for the GeoJSON, and the comment block atop the dictionary
// CSV. Records the settings the scores depend on — without these, the index
// columns cannot be reproduced. They're the build's settings, not the panel's,
// which may have changed since the scores in the file were built.
function _exportManifest(cols,nRows){
  var st=_builtSettings(), md=heatModeDef(st.heatMode);
  var onVars=results?st.on.sensitivity.concat(st.on.adaptive):[];
  return {citation:EXPORT_META.citation,license:EXPORT_META.license,
    sources_url:EXPORT_META.sources_url,
    generated:new Date().toISOString(),
    tool:_SRC_SELF,
    tracts:nRows,
    tract_vintage:'2020 Census tracts (TIGER/Line)',
    heat_mode:md?(md.key+' \\u2014 '+md.label):heatMode,
    heat_mode_layers:md?md.layers.join(' + '):'',
    scenario:curScenario,
    period:(curScenario==='present'?'':(curPeriod||'')),
    composite_variables:onVars.join(' '),
    fields:_buildDataDictionary(cols)};
}

function _dictionaryToCSV(cols,nRows){
  var man=_exportManifest(cols,nRows);
  // Leading '#' lines: ignore with pandas.read_csv(..., comment='#').
  var head=[
    '# San Diego Heat Risk Explorer \\u2014 data dictionary',
    '# Describes the columns of the data file downloaded alongside this one.',
    '# generated: '+man.generated,
    '# tool: '+man.tool,
    '# tracts: '+man.tracts+'  ('+man.tract_vintage+')',
    '# heat mode: '+man.heat_mode+'  [layers: '+man.heat_mode_layers+']',
    '# scenario: '+man.scenario+(man.period?'  period: '+man.period:''),
    '# variables in the composite: '+(man.composite_variables||'(none)'),
    '# license: '+man.license,
    '# full source table: '+man.sources_url,
    '#'
  ].join('\\n');
  var lines=[head,_DICT_COLS.join(',')];
  man.fields.forEach(function(r){
    lines.push(_DICT_COLS.map(function(k){return _csvEscape(r[k]);}).join(','));
  });
  return lines.join('\\n')+'\\n';
}

function _slugify(s){return String(s||'').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'');}
function _exportStem(scope,jurisNames){
  var parts=['san_diego_heat_risk'];
  if(scope==='juris'&&jurisNames.length){
    var slugs=jurisNames.map(_slugify).filter(Boolean);
    parts.push(slugs.length<=3?slugs.join('-'):(slugs.length+'_cities'));
  }else{
    parts.push('all_tracts');
  }
  parts.push((curScenario==='present')?'present':(curScenario+(curPeriod?'_'+curPeriod:'')));
  return parts.join('_');
}
function _exportFilename(scope,jurisNames,fmt){
  return _exportStem(scope,jurisNames)+'.'+(fmt==='geojson'?'geojson':'csv');
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
    var stem=_exportStem(scope,jurisNames);
    var fname=_exportFilename(scope,jurisNames,fmt);
    if(fmt==='geojson'){
      _triggerDownload(_featuresToGeoJSON(indices,cols),fname,'application/geo+json');
    }else{
      _triggerDownload(_featuresToCSV(indices,cols),fname,'text/csv');
    }
    // Second file: what each column is, its units, its polarity and its source.
    // On by default — a data file that leaves here without its key is not
    // interpretable. Browsers may ask to "allow multiple downloads"; expected.
    if(_dictWanted()){
      _triggerDownload(_dictionaryToCSV(cols,indices.length),
                       stem+'_data_dictionary.csv','text/csv');
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

/* Welcome popup — shows on every page load. Dismissal is per-session
   (closing/refreshing brings it back) so the framing message lands every
   time someone opens the tool. */
function closeWelcome(){
  var m=document.getElementById('welcome-modal');
  if(!m)return;
  m.classList.remove('open');
}
(function(){
  var m=document.getElementById('welcome-modal');
  if(!m)return;
  m.classList.add('open');
  var ok=document.getElementById('wlc-got-it');
  if(ok)ok.addEventListener('click',closeWelcome);
  var x=document.getElementById('wlc-close-btn');
  if(x)x.addEventListener('click',closeWelcome);
  var tt=document.getElementById('wlc-take-tour');
  if(tt)tt.addEventListener('click',function(){closeWelcome();startTour();});
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'&&m.classList.contains('open'))closeWelcome();
  });
})();

/* ================================================================
   GUIDED TOUR — opt-in walkthrough of the sidebar pillars.
   Each step describes a section, optionally prompts the user to try
   the highlighted control, and advances on Next.
   ================================================================ */
var TOUR_STEPS=[
  {key:"exposure",target:'[data-tour="pillar-exposure"]',side:"right",
   title:"1. Heat Exposure",
   body:"The heat hazard itself. Choose one mode \\u2014 <b>Daytime</b>, <b>Nighttime</b>, <b>Humid heat (WBGT)</b>, or <b>Surface temperature</b>.",
   prompt:"Pick a mode, then Next."},
  {key:"sensitivity",target:'[data-tour="pillar-sensitivity"]',side:"right",
   title:"2. Population Sensitivity",
   body:"Who is more vulnerable \\u2014 age, poverty, language, health. Toggle the header for all of it, or open a subsection to choose variables.",
   prompt:"Toggle a variable, then Next."},
  {key:"adaptive",target:'[data-tour="pillar-adaptive"]',side:"right",
   title:"3. Adaptive Capacity",
   body:"What helps people cope \\u2014 air conditioning, tree canopy, income, health insurance. Low capacity raises risk, so these are inverted in the blend.",
   prompt:"Toggle a variable, then Next."},
  {key:"build",target:'#compute-btn',side:"right",
   title:"4. Build the map",
   body:"Combines your selections into one risk score per tract. Variables are z-scored, Sensitivity and Adaptive Capacity reduced by PCA, then ranked countywide.",
   prompt:"Build the map, then Next."},
  {key:"layer",target:'#layer-select',side:"right",
   title:"5. Map layer",
   body:"Change views without rebuilding. <b>Composite Risk</b> is the blend; <b>Dominant Driver</b> colors each tract by its strongest pillar. Single pillars and variables are available too.",
   prompt:"Pick a layer, then Next."},
  {key:"scenario",target:'#scenario-bar',side:"right",
   title:"6. Climate scenarios",
   body:"Project risk under future warming. Pick <b>2050</b> or <b>2080</b> and an emissions pathway. Only Heat Exposure shifts \\u2014 Sensitivity and Adaptive Capacity stay at today's baseline.",
   prompt:null},
  {key:"done",target:null,side:"center",
   title:"Done",
   body:"Click any tract for its scores and dominant driver. Use <b>Find address</b> (bottom-left) to jump to a place. Reload to retake the tour.",
   prompt:null}
];
/* ----------------------------------------------------------------
   WORKED EXAMPLE — Example 1 from the user guide
   ("Where are children most at risk?"). Mirrors the regular tour
   state machine but each step's `action` auto-fires on Next so the
   user watches the tool perform the example.
   ---------------------------------------------------------------- */
var WE_STEPS=[
  {key:"we-intro",target:null,side:"center",
   title:"Worked example: where are children most at risk?",
   body:"We'll narrow Population Sensitivity to one variable \\u2014 children under 18 \\u2014 to find where kids face the most heat stress. Click <b>Next</b> and the tool does each step.",
   prompt:null},
  {key:"we-clear",target:'[data-tour="pillar-sensitivity"]',side:"right",
   title:"1. Clear Population Sensitivity",
   body:"Uncheck every variable here so we can isolate one.",
   prompt:"Next clears the pillar.",
   action:function(){togPillar('sensitivity',false);}},
  {key:"we-age",target:'#vr_EP_AGE17',side:"right",
   title:"2. Enable Age 17 & Under",
   body:"Turn on just the under-18 variable. Sensitivity now tracks where children live.",
   prompt:"Next enables it.",
   // `prepare` runs before the spotlight positions, so the Demographics
   // subsection is open and the target row is laid out (non-zero height)
   // by the time getBoundingClientRect reads it.
   prepare:function(){
     var sec=document.querySelector('[data-tour="pillar-sensitivity"]');
     if(!sec)return;
     var ds=sec.querySelectorAll('details.subsec');
     for(var i=0;i<ds.length;i++){
       if(ds[i].querySelector('#vr_EP_AGE17')){ds[i].open=true;break;}
     }
   },
   action:function(){if(!enabled['EP_AGE17'])tog('EP_AGE17');}},
  {key:"we-build",target:'#compute-btn',side:"right",
   title:"3. Build the map",
   body:"Blend Heat Exposure, children-only Sensitivity, and Adaptive Capacity into the risk index.",
   prompt:"Next builds the map.",
   action:function(){
     // Safety net: ensure Adaptive Capacity has variables on so the
     // composite is meaningful (matches the user-guide instruction
     // "select all variables for Heat Exposure and Adaptive Capacity").
     // Heat Exposure is a single-select radio, so a mode is already chosen.
     var anyAdaptive=false;
     VM.forEach(function(v){if(v.pillar==='adaptive'&&enabled[v.key])anyAdaptive=true;});
     if(!anyAdaptive)togPillar('adaptive',true);
     compute();
   }},
  {key:"we-done",target:null,side:"center",
   title:"Done \\u2014 children at risk",
   body:"Dark-red tracts are where children face the most heat stress \\u2014 hot, many kids under 18, limited capacity to cope. Click any tract for its breakdown.",
   prompt:null}
];

var _tourIdx=0;
var _inWorkedExample=false;
var _tourResizeObs=null;
function _currentSteps(){return _inWorkedExample?WE_STEPS:TOUR_STEPS;}

function startTour(){_inWorkedExample=false;_tourIdx=0;_showTourStep(0);}
function startWorkedExample(){_inWorkedExample=true;_tourIdx=0;_showTourStep(0);}

// Re-position the spotlight whenever the highlighted element changes size
// (e.g. user expands a <details> subsection, revealing more variables).
function _attachTourResizeObs(el,side){
  _detachTourResizeObs();
  if(typeof ResizeObserver==='undefined')return;
  _tourResizeObs=new ResizeObserver(function(){_positionAtTarget(el,side);});
  _tourResizeObs.observe(el);
}
function _detachTourResizeObs(){
  if(_tourResizeObs){try{_tourResizeObs.disconnect();}catch(e){}_tourResizeObs=null;}
}

function endTour(){
  _detachTourResizeObs();
  _inWorkedExample=false;
  var ov=document.getElementById('tour-overlay');
  if(ov)ov.classList.remove('open');
}

function _advanceTour(){
  var steps=_currentSteps();
  if(_tourIdx>=steps.length-1){endTour();return;}
  var step=steps[_tourIdx];
  if(step&&typeof step.action==='function'){
    try{step.action();}catch(e){console.warn('Tour step action failed:',e);}
    // Brief pause so the user sees the action take effect before the card moves on.
    setTimeout(function(){_tourIdx++;_showTourStep(_tourIdx);},420);
    return;
  }
  _tourIdx++;_showTourStep(_tourIdx);
}

function _tourBack(){
  if(_tourIdx<=0)return;
  _tourIdx--;_showTourStep(_tourIdx);
}

function _showTourStep(i){
  var steps=_currentSteps();
  var step=steps[i];if(!step)return endTour();
  var ov=document.getElementById('tour-overlay');
  ov.classList.add('open');
  var progLabel=_inWorkedExample?'Example':'Step';
  document.getElementById('tc-progress').textContent=progLabel+' '+(i+1)+' of '+steps.length;
  document.getElementById('tc-title').textContent=step.title;
  document.getElementById('tc-body').innerHTML=step.body;
  var pr=document.getElementById('tc-prompt');
  if(step.prompt){pr.textContent=step.prompt;pr.style.display='flex';}
  else pr.style.display='none';
  // Footer buttons
  var back=document.getElementById('tc-back');
  back.disabled=(i===0);
  back.style.visibility=(i===0?'hidden':'visible');
  var next=document.getElementById('tc-next');
  next.textContent=(i===steps.length-1?'Finish':'Next');
  // "See worked example" button: only visible on the last step of the main tour.
  var ex=document.getElementById('tc-example');
  if(ex)ex.style.display=(!_inWorkedExample&&i===TOUR_STEPS.length-1)?'inline-block':'none';
  // Detach the previous step's ResizeObserver before any DOM mutations in
  // `prepare` so we don't briefly reposition the old spotlight on stale targets.
  _detachTourResizeObs();
  if(typeof step.prepare==='function'){try{step.prepare();}catch(e){console.warn('Tour prepare failed:',e);}}
  // Position card + spotlight
  if(step.target){
    var el=document.querySelector(step.target);
    if(!el){_positionCenter();}
    else{
      try{el.scrollIntoView({behavior:'instant',block:'nearest'});}catch(e){el.scrollIntoView();}
      // Double-rAF so the sidebar scroll + layout fully settle before
      // we read getBoundingClientRect — a single setTimeout occasionally
      // measured stale coords and left the spotlight a few px off.
      requestAnimationFrame(function(){
        requestAnimationFrame(function(){
          _positionAtTarget(el,step.side);
          _attachTourResizeObs(el,step.side);
        });
      });
    }
  }else{
    _positionCenter();
  }
}

function _positionCenter(){
  _detachTourResizeObs();
  // Hide dim rects and spotlight; center the card.
  ['tour-dim-top','tour-dim-right','tour-dim-bottom','tour-dim-left'].forEach(function(id){
    var d=document.getElementById(id);d.style.cssText='position:fixed;background:rgba(15,23,42,0.55);z-index:2400;left:0;top:0;width:0;height:0';
  });
  // Full-screen dim using the top rect
  var top=document.getElementById('tour-dim-top');
  top.style.cssText='position:fixed;background:rgba(15,23,42,0.55);z-index:2400;left:0;top:0;right:0;bottom:0;width:auto;height:auto';
  document.getElementById('tour-spotlight').style.display='none';
  var card=document.getElementById('tour-card');
  card.className='tour-card centered';
  card.style.left='';card.style.top='';card.style.right='';card.style.bottom='';
}

function _positionAtTarget(el,side){
  var r=el.getBoundingClientRect();
  // Pad the spotlight slightly so the ring sits proud of the element.
  var pad=6;
  var x=r.left-pad,y=r.top-pad,w=r.width+pad*2,h=r.height+pad*2;
  var vw=window.innerWidth,vh=window.innerHeight;
  // Four-rectangle dim
  var top=document.getElementById('tour-dim-top'),
      right=document.getElementById('tour-dim-right'),
      bottom=document.getElementById('tour-dim-bottom'),
      left=document.getElementById('tour-dim-left');
  top.style.cssText='position:fixed;background:rgba(15,23,42,0.55);z-index:2400;left:0;top:0;width:'+vw+'px;height:'+Math.max(0,y)+'px;backdrop-filter:blur(1px)';
  bottom.style.cssText='position:fixed;background:rgba(15,23,42,0.55);z-index:2400;left:0;top:'+(y+h)+'px;width:'+vw+'px;height:'+Math.max(0,vh-(y+h))+'px;backdrop-filter:blur(1px)';
  left.style.cssText='position:fixed;background:rgba(15,23,42,0.55);z-index:2400;left:0;top:'+Math.max(0,y)+'px;width:'+Math.max(0,x)+'px;height:'+h+'px;backdrop-filter:blur(1px)';
  right.style.cssText='position:fixed;background:rgba(15,23,42,0.55);z-index:2400;left:'+(x+w)+'px;top:'+Math.max(0,y)+'px;width:'+Math.max(0,vw-(x+w))+'px;height:'+h+'px;backdrop-filter:blur(1px)';
  // Spotlight ring
  var sp=document.getElementById('tour-spotlight');
  sp.style.display='block';
  sp.style.left=x+'px';sp.style.top=y+'px';
  sp.style.width=w+'px';sp.style.height=h+'px';
  // Card placement
  var card=document.getElementById('tour-card');
  card.className='tour-card';
  card.style.right='';card.style.bottom='';
  var cw=card.offsetWidth||320,ch=card.offsetHeight||200;
  var gap=18;
  var placeLeft=r.left,placeTop=r.top;
  var pos=side||'right';
  if(pos==='right'&&r.right+gap+cw<=vw-8){
    placeLeft=r.right+gap;placeTop=Math.max(12,Math.min(r.top,vh-ch-12));
    card.classList.add('arrow-left');
  }else if(r.left-gap-cw>=8){
    placeLeft=r.left-gap-cw;placeTop=Math.max(12,Math.min(r.top,vh-ch-12));
    card.classList.add('arrow-right');
  }else if(r.bottom+gap+ch<=vh-8){
    placeLeft=Math.max(12,Math.min(r.left,vw-cw-12));placeTop=r.bottom+gap;
    card.classList.add('arrow-top');
  }else{
    placeLeft=Math.max(12,Math.min(r.left,vw-cw-12));placeTop=Math.max(12,r.top-ch-gap);
    card.classList.add('arrow-bottom');
  }
  card.style.left=placeLeft+'px';card.style.top=placeTop+'px';
}

// Tour control wiring + keyboard + resize handling
(function(){
  var nb=document.getElementById('tc-next');
  if(nb)nb.addEventListener('click',_advanceTour);
  var bb=document.getElementById('tc-back');
  if(bb)bb.addEventListener('click',_tourBack);
  var sb=document.getElementById('tc-skip');
  if(sb)sb.addEventListener('click',endTour);
  var ex=document.getElementById('tc-example');
  if(ex)ex.addEventListener('click',startWorkedExample);
  document.addEventListener('keydown',function(e){
    var ov=document.getElementById('tour-overlay');
    if(!ov||!ov.classList.contains('open'))return;
    if(e.key==='Escape')endTour();
    else if(e.key==='ArrowRight')_advanceTour();
    else if(e.key==='ArrowLeft')_tourBack();
  });
  window.addEventListener('resize',function(){
    var ov=document.getElementById('tour-overlay');
    if(!ov||!ov.classList.contains('open'))return;
    var step=_currentSteps()[_tourIdx];if(!step||!step.target){_positionCenter();return;}
    var el=document.querySelector(step.target);
    if(el)_positionAtTarget(el,step.side);
  });
  // Reposition while the sidebar scrolls (e.g. user scrolls between steps).
  var sb2=document.getElementById('sidebar');
  if(sb2)sb2.addEventListener('scroll',function(){
    var ov=document.getElementById('tour-overlay');
    if(!ov||!ov.classList.contains('open'))return;
    var step=_currentSteps()[_tourIdx];if(!step||!step.target)return;
    var el=document.querySelector(step.target);
    if(el)_positionAtTarget(el,step.side);
  });
})();

console.log("DIY PCA loaded. Features:",GJ.features.length,"Vars:",VM.length);
</script>
<script>
/* --- mobile: controls drawer + tap-to-reveal tooltips --- */
function toggleSidebar(force){
  var open = (typeof force === 'boolean') ? force : !document.body.classList.contains('sidebar-open');
  document.body.classList.toggle('sidebar-open', open);
  var btn = document.getElementById('menu-toggle');
  if(btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}
/* Touch has no hover, so a tap toggles the info bubble; tapping another
   icon or anywhere else closes the open one. blur() drops the :focus
   state so a second tap can actually hide it. Keyboard focus still works. */
document.addEventListener('click', function tapInfo(e){
  var ic = e.target.closest ? e.target.closest('.info-ic') : null;
  document.querySelectorAll('.info-ic.open').forEach(function(el){
    if(el !== ic){ el.classList.remove('open'); el.blur(); }
  });
  if(ic){
    var willOpen = !ic.classList.contains('open');
    ic.classList.toggle('open', willOpen);
    if(!willOpen) ic.blur();
  }
});
/* On mobile the on-map controls are reparented into the drawer
   (#drawer-tools-slot) so the map is unobstructed; on desktop they live on the
   map (bottom-left stack + bottom-right stack). The bottom-left stack is appended
   first so the bottom-right controls land last in the drawer. Re-runs on viewport
   changes such as phone rotation. */
(function(){
  var mq = window.matchMedia('(max-width:768px),(max-height:500px) and (orientation:landscape)');
  var els = ['map-controls-bl','map-controls-br']
    .map(function(id){return document.getElementById(id);}).filter(Boolean);
  var slot = document.getElementById('drawer-tools-slot');
  var mapWrap = document.getElementById('map-wrap');
  if(!els.length || !slot || !mapWrap) return;
  function place(){
    var target = mq.matches ? slot : mapWrap;
    els.forEach(function(el){
      if(el.parentNode !== target) target.appendChild(el);
    });
  }
  if(mq.addEventListener) mq.addEventListener('change', place);
  else if(mq.addListener) mq.addListener(place);
  place();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

