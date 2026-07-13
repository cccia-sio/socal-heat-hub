# San Diego Heat Risk Explorer

### [**Landing Page &rarr;**](https://cccia-sio.github.io/socal-heat-hub/)

Custom composite heat-risk map for every census tract in San Diego County.

Pick the variables that matter to you — temperatures, chronic disease rates,
tree canopy, AC access, poverty — and the tool combines them into a
neighborhood-level risk map. Toggle climate scenarios to see how risk shifts
under mid-century and end-of-century warming. The tool runs entirely in the
browser — no server, no account, and no analytics or tracking of any kind.

Two things do leave your browser, both to third parties, neither logged by us:
map tiles are fetched from CARTO and Esri as you pan, and **anything you type
into the address search is sent to OpenStreetMap's Nominatim** geocoder to be
turned into a location. If you'd rather not send an address to a third party,
click the map instead of searching for it.

See the [User Guide](https://cccia-sio.github.io/socal-heat-hub/userguide/)
for variables, methodology, and map interpretation.

## Three pillars of heat risk

- **Exposure** — How much heat does this area face? Air temperature, humidity,
  number of extreme-heat days, and satellite-measured land surface temperature.
- **Sensitivity** — How vulnerable are the people living here? Age, chronic
  disease rates, social vulnerability, and environmental health.
- **Adaptive Capacity** — What resources help residents cope with heat? Air
  conditioning access, tree canopy, income, healthcare access, and housing
  quality.

## Data sources

- Weather stations: [MesoWest / Synoptic](https://synopticdata.com/)
- Land surface temperature: [USGS Landsat](https://www.usgs.gov/landsat-missions)
- Chronic disease: [CDC PLACES](https://www.cdc.gov/places/index.html)
- Social vulnerability: [CDC SVI 2022](https://www.atsdr.cdc.gov/place-health/php/svi/index.html)
- Healthy Places Index: [HPI](https://www.healthyplacesindex.org/)
- AC prevalence: [Romitti et al. 2022](https://academic.oup.com/pnasnexus/article/1/4/pgac210/6713629), [Ahn et al. 2025](https://doi.org/10.1038/s41597-025-06104-3)
- Demographics & housing: [ACS 5-year estimates](https://www.census.gov/programs-surveys/acs)
- Climate projections: [LOCA2](https://loca.ucsd.edu/) downscaled CMIP6 (mid-century 2035–2064 and end-of-century 2065–2094)

## Acknowledgments

This tool was created through a joint research program between the
[Southern California Extreme Heat Research Hub](https://socalheathub.ucsd.edu/)
and the
[San Diego Regional Climate Collaborative](https://www.sandiego.edu/soles/centers-and-institutes/nonprofit-institute/signature-programs/climate-collaborative/).
Funding was provided through the
[NSF CoPe program](https://www.nsf.gov/funding/opportunities/cope-coastlines-people-hubs-research-broadening-participation).

Issues and feedback welcome via GitHub issues.

## Citation

If you use this tool, its methodology, or its derived data, please cite it —
see [CITATION.cff](CITATION.cff).

## License

- **Code** (build pipeline, scripts, front-end source) — [MIT](LICENSE).
- **Data & content** (composite scores, percentile ranks, and the layers
  computed here) — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
  reuse freely, including commercially, with attribution.

The MIT grant covers the **code**, not the tract data baked into the generated
`main/heat_risk_map.html`. That file embeds third-party data (HPI, UCLA Heat
Maps) that this project cannot sublicense commercially — so copying the
generated HTML is not an MIT-covered act. See the scope note in
[LICENSE](LICENSE).

Incorporated third-party data keeps its own terms. In particular, **Healthy
Places Index** data is non-commercial, so HPI-derived columns are shown on the
map but **excluded from the "Download Data" export**. See
[DATA_LICENSE.md](DATA_LICENSE.md) for the full breakdown.
