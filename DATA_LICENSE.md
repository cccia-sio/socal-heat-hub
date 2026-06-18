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
| [SANDAG / SanGIS](https://www.sangis.org/) | Raw parcel/land-base layers need a SanGIS sublicense | Only a derived per-tract green-space percentage is used; raw parcels are not redistributed. |
| [MesoWest / Synoptic](https://synopticdata.com/) | Open for non-commercial / research use | Only derived per-tract temperature statistics are redistributed. |
| [CalHeatScore (CalEPA / OEHHA)](https://calheatscore.calepa.ca.gov/) | State aggregated product | Only tract-level relative rate ratios are used. |

Because HPI is non-commercial and CC BY permits commercial use, HPI-derived
columns are **omitted from the bulk data download** to keep the downloadable
dataset CC BY-clean. They remain visible on the map, used under HPI's own terms
with attribution. If you redistribute the raw inputs (not the normal "view/use
the map" path), check each source's terms.

See the full source list in the [User Guide](userguide/index.html).
