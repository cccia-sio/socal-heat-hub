# Data & content license

The **software** in this repository (build pipeline, scripts, front-end code) is
licensed under the **MIT License** — see [`LICENSE`](LICENSE).

The **data, map output, and written documentation** that this project *produces*
— the composite risk scores, the within-county percentile ranks, and the layers
computed here (impervious surface, energy burden, land-surface temperature,
WBGT) — are licensed under:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**
<https://creativecommons.org/licenses/by/4.0/>

You may share and adapt this material, including commercially, **as long as you
give appropriate credit.** Suggested attribution:

> San Diego Heat Risk Explorer — Southern California Extreme Heat Research Hub &
> San Diego Regional Climate Collaborative. Licensed CC BY 4.0.

## Third-party source data retains its own terms

This tool incorporates data from external sources. **CC BY 4.0 applies only to
this project's own contributions, not to the underlying third-party data**, each
of which keeps its original license. Most sources are U.S. federal public-domain
works (Census/ACS, CDC PLACES & SVI, USGS Landsat, NOAA HRRR, NLCD, DOE LEAD),
but several have their own conditions:

| Source | Terms | Handling |
|---|---|---|
| [Healthy Places Index (HPI)](https://www.healthyplacesindex.org/terms-and-conditions) | **Non-commercial** + attribution; mark if modified | **Excluded from the "Download Data" export.** Shown on the map under HPI's terms with attribution; obtain the data from HPI directly. |
| [CalHeatScore (CalEPA / OEHHA)](https://calheatscore.calepa.ca.gov/) | **No published license for the per-ZCTA effect estimates** | **Excluded from the export** (2026-07-27 audit). See below. |
| [SANDAG / SanGIS](https://www.sangis.org/) | Raw parcel/land-base layers need a SanGIS sublicense | Only a derived per-tract green-space percentage is used; raw parcels are not redistributed. The land-use input came from the open [SANDAG Open Data portal](https://opendata.sandag.org/). |
| [MesoWest / Synoptic](https://synopticdata.com/tcs/) | Non-commercial; summary statistics may be published only if the underlying data "cannot be decompiled, reverse engineered, or otherwise extracted" | Derived per-tract temperature statistics only, each a blend of six stations with no station exceeding 40% of the value. See below. |
| [Romitti et&nbsp;al. 2022](https://doi.org/10.7910/DVN/HWFVP6) | CC BY article; data deposited on Harvard Dataverse | Open. Cited to the Dataverse DOI (the redistributable artifact), not the article page. |
| [UCLA Heat Maps](https://uclaheatmaps.org) | Public research product, aggregated from California HCAI emergency-department records | Display-only map layer; **not included in the export.** |

## Provenance audit, 2026-07-27

Every column reaching the "Download Data" export was traced to its source file
and that file's origin checked. Findings:

**Verified open** — CDC PLACES, CDC/ATSDR SVI, ACS (B25034 / B25003 / C24030 /
B11007), Census LACE 2023, USGS Landsat, NOAA HRRR, NLCD/MRLC, DOE LEAD,
TIGER/Line, LOCA2, SANDAG Open Data, and Romitti et al. via Harvard Dataverse.
All are U.S. federal public-domain works, state open-data releases, or CC BY.

**Withdrawn from the export** — `heat_hosp_day` and `heat_hosp_night`. Their
input (`data/hospitalizations/joined_rel.csv`) arrived as an emailed
attachment; it carries no license, has no public download URL, and its column
schema (`max99_2`, `hi95_SNR`, …) is analyst-internal rather than a published
product. CalHeatScore's public release is daily scores plus a REST API — the
per-ZCTA effect estimates are not part of it. Absent a demonstrable right to
redistribute, they are now treated like HPI: on the map, out of the download.
Request them from `CalHeatScore@oehha.ca.gov`.

### Resolved: single-station recoverability in the MesoWest columns

Synoptic permits publishing summary statistics *provided the underlying
observations cannot be extracted from them*. The audit found two places where
our tract aggregates failed that test. Both are fixed.

**What was wrong.** Of the 1,093 contributing stations, 511 are citizen networks
(Tempest 228, CWOP 226, WXM 57) and 414 professional (SDG&E mesonet 220,
SGXWFO 110). CWOP is public domain; Tempest, WXM, SDG&E and SC-Edison are
privately operated.

- **Daytime** (`mean_tmax`, `p85_tmax`, `p99_tmax`, `max_tmax`): the pipeline
  pooled stations into their containing tract, ran IDW from those ~212 tract
  centroids, then overwrote every station-containing tract with the pooled
  direct computation. 131 of those 212 tracts held exactly **one** station, so
  the published value *was* that station's statistic — at least 14 of them on
  the private SDG&E mesonet.
- **Nighttime** (`mean_tmin_overnight`, `p90_tmin_overnight`,
  `no_relief_fraction`): one-hop IDW (k=6) normally blends six stations, but in
  35 tracts a single station carried >90% of the inverse-distance weight (one
  centroid sits 14 m from its station). 24 of those were private networks.

**What changed** (both in the `heat-risk-sd` repo):

1. `analyze_extreme_heat.py` now writes `station_heat_metrics.csv` — per-station
   heat statistics for the 503 stations with at least a year of record. A stub
   for this existed but returned an empty DataFrame, which is why interpolation
   ran from tract centroids in the first place.
2. `map_tract_heat_risk.py` interpolates in **one hop from those 503 station
   points** and no longer overwrites station-containing tracts. Control points
   went from 212 to 503, so this also raises resolution where stations are
   densest — the urban cores.
3. Both `map_tract_heat_risk.py` and `compute_nighttime_tract_metrics.py` cap
   any single station at **40%** of a tract's interpolated value, redistributing
   the excess across the remaining neighbours. Verified: 0 of 737 tracts exceed
   the cap on either path, and every tract now blends six stations.

The cap is enforced by `tests/test_idw_weight_cap.py` in that repo.

Effect on the published values: daytime `mean_tmax` rose 0.48 °C on average
(Spearman 0.83 against the old field), `max_tmax` 1.59 °C (Spearman 0.68 — the
old field was a station maximum passed straight through, which the blend now
smooths). Nighttime shifted by −0.03 to +0.05 °C on average.

Because HPI is non-commercial and CC BY permits commercial use, HPI-derived
columns are **omitted from the bulk data download** to keep the downloadable
dataset CC BY-clean. They remain visible on the map, used under HPI's own terms
with attribution. If you redistribute the raw inputs (not the normal "view/use
the map" path), check each source's terms.

The exported data dictionary lists every withheld variable with
`in_download = no` and the reason, so an absent column is explained rather than
mysterious. Exclusions are driven by a single table —
`RESTRICTED_EXPORT_SOURCES` in `scripts/build_diy_pca_viewer.py` — and enforced
by `tests/test_export_licensing.py`.

### Why HPI is excluded but MesoWest/Synoptic is not

Both sources are non-commercial, so the differing treatment is deliberate. The
HPI columns on the map are HPI's *published product* — their percentile values,
passed through unchanged; redistributing them is redistributing HPI. The
MesoWest columns (`mean_tmax`, `p85_tmax`, `p99_tmax`, `max_tmax`,
`mean_tmin_overnight`, `p90_tmin_overnight`, `no_relief_fraction`) are new
statistics this project computes from raw station observations and interpolates
to tracts, which is the case Synoptic's terms expressly allow publishing. The
same reasoning covers the derived SanGIS green-space percentage.

An earlier version of this file claimed the observations "cannot be recovered
from the aggregates," and extended the reasoning to the CalHeatScore variables.
The 2026-07-27 audit found both claims wrong as written: the aggregates *were*
invertible in ~166 single-station tracts, and the CalHeatScore input has no
license at all. The CalHeatScore variables have been withdrawn from the export,
and the interpolation was rebuilt so the claim now holds — every tract value is
a six-station blend with a 40% per-station ceiling.

### Attribution travels with the download

Every row of the CSV and every feature of the GeoJSON produced by "Download
Data" carries three constant columns — `citation`, `license` (CC BY 4.0), and
`sources_url` — so a file that leaves this site still states its terms and links
back to the per-variable source table. These are stamped from `EXPORT_META` in
`scripts/build_diy_pca_viewer.py`.

See the full source list in the [User Guide](userguide/index.html).
