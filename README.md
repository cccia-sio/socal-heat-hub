# San Diego Heat Risk Explorer

Interactive web tool for exploring heat risk across San Diego County census
tracts. Users select which exposure, sensitivity, and adaptive-capacity
variables to include, then build a composite risk map using per-pillar
principal component analysis. Climate scenarios (SSP2-4.5, SSP3-7.0,
SSP5-8.5) apply downscaled LOCA2 CMIP6 deltas to temperature variables.

**Live tool:** [main/diy_pca.html](main/diy_pca.html)

**User guide (PDF):** [userguide/userguide.pdf](userguide/userguide.pdf)

## About

This repository hosts the publicly accessible build of the viewer. The tool
runs entirely in the browser — no server, no tracking, no account needed.

Data sources:
- **Exposure:** MesoWest/Synoptic weather stations, NASA Landsat LST
- **Sensitivity:** CDC PLACES, CDC Social Vulnerability Index
- **Adaptive capacity:** Healthy Places Index, Romitti et al. AC prevalence,
  Ahn et al. AC type prevalence, ACS demographics
- **Climate projections:** LOCA2 CMIP6 ensemble (2035–2064 vs. 1985–2014)

## Citation

If you use this tool in published work, please cite:
> Mack, C. J. (2026). San Diego Heat Risk Explorer. Interactive web tool.
> https://connorjmack.github.io/heat-risk-viewer/

## Contact

Issues and feedback welcome via GitHub issues.
