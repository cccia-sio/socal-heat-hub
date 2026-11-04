# San Diego Heat Risk Explorer

### [**Landing Page &rarr;**](https://cccia-sio.github.io/socal-heat-hub/)

Custom composite heat-risk map for every census tract in San Diego County.

Pick the variables that matter to you — temperatures, chronic disease rates,
tree canopy, AC access, poverty — and the tool combines them into a
neighborhood-level risk map. Toggle climate scenarios to see how risk shifts
under mid-century and end-of-century warming. The tool runs entirely in the
browser — no server, no account, and no analytics or tracking of any kind.

Two things do leave your browser, both to third parties, neither logged by us:
map tiles are fetched from Esri as you pan, and **anything you type
into the address search is sent to OpenStreetMap's Nominatim** geocoder to be
turned into a location. If you'd rather not send an address to a third party,
click the map instead of searching for it.

See the [User Guide](https://cccia-sio.github.io/socal-heat-hub/userguide/)
for variables, methodology, and map interpretation.

## Three pillars of heat risk

- **Heat Exposure** — How much heat does this area face? Daytime and overnight
  air temperature, wet-bulb globe temperature, and satellite land surface
  temperature.
- **Population Sensitivity** — How vulnerable are the people living here? Age, chronic
  disease rates, social vulnerability, and environmental health.
- **Adaptive Capacity** — What resources help residents cope with heat? Air
  conditioning access, tree canopy, income, health insurance, and housing
  quality.

## Data sources

Every variable's source is listed in the User Guide's
[Data sources](https://cccia-sio.github.io/socal-heat-hub/userguide/#data)
table, and linked from each variable in the tool.

## Acknowledgments

This tool was created through a joint research program between the
[Southern California Extreme Heat Research Hub](https://socalheathub.ucsd.edu/)
and the
[San Diego Regional Climate Collaborative](https://www.sandiego.edu/soles/centers-and-institutes/nonprofit-institute/signature-programs/climate-collaborative/).
Funding was provided through the
[NSF CoPe program](https://www.nsf.gov/funding/opportunities/cope-coastlines-people-hubs-research-broadening-participation).

Issues and feedback welcome via GitHub issues.

## Citation

If you use this tool, its methodology, or its derived data, please cite it:

<!-- TODO: after the v1.0.0 release, replace XXXXXXX (here and in CITATION.cff)
     with the Zenodo concept DOI, which always resolves to the latest version. -->

```text
Mack, C., Hale, M., Gong, A., Berry, D., & Merrifield, M. (2026). San Diego Heat Risk Explorer (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

```bibtex
@software{mack2026sdheatrisk,
  author    = {Mack, Connor and Hale, Maren and Gong, A-bel and Berry, Darbi and Merrifield, Mark},
  title     = {San Diego Heat Risk Explorer},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

GitHub's "Cite this repository" button gives the same citation in other
formats, from [CITATION.cff](CITATION.cff).

## License

- **Code** (build pipeline, scripts, front-end source) — [MIT](LICENSE).
- **Data & content** (composite scores, percentile ranks, and the layers
  computed here) — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
  reuse freely, including commercially, with attribution.

The MIT grant covers the **code**, not the tract data baked into the generated
`main/heat_risk_map.html`. That file embeds third-party data (HPI, UCLA Heat
Maps) that this project cannot sublicense commercially — so copying the
generated HTML is not an MIT-covered act. See
[what the MIT license does not cover](DATA_LICENSE.md#what-the-mit-license-does-not-cover).

Incorporated third-party data keeps its own terms. In particular, **Healthy
Places Index** data is non-commercial, so HPI-derived columns are shown on the
map but **excluded from the "Download Data" export**. See
[DATA_LICENSE.md](DATA_LICENSE.md) for the full breakdown.
