# Urban Heat Island Index — Methodology Spec (draft)

Status: draft for review. No code written yet. Target product: a 100 m
surface-UHI-intensity raster for San Diego County, delivered as a PNG overlay in
the existing viewer (reusing the `build_lst_overlay.py` pipeline) plus a
per-tract summary table.

This spec covers (1) what the index measures and does not measure, (2) the
de-trending math that separates the urban signal from San Diego's marine and
elevation gradient, (3) driver attribution, (4) the raster and tract outputs,
(5) validation against the transect data, and (6) the decisions that need your
call before implementation.

---

## 1. Scope

The index estimates **surface urban heat island intensity (SUHII)**: the
clear-sky daytime land-surface temperature of a pixel minus the temperature
expected for non-urban land at the same distance from the coast and the same
elevation. Units are °C. Positive values mean the surface is hotter than the
natural background; values near zero mean the surface behaves like its rural
surroundings.

It does **not** measure air temperature, and it does not measure the nighttime
canopy-layer UHI that drives overnight health load (Oke 1982). Surface and air
UHI differ in magnitude, sign of the day/night cycle, and spatial texture
(Voogt & Oke 2003). Daytime SUHII from Landsat is the data-rich starting point;
a nighttime air-UHI layer from ECOSTRESS night mosaics and `mean_tmin_overnight`
is a separate product (Section 7).

It is not the existing composite heat-risk score. That score blends hazard,
sensitivity, and adaptive capacity into one percentile rank. SUHII is a physical
anomaly in °C with no sociodemographic content. The two are combined only at the
end, and kept on separate axes (Section 6).

### Conventions

- Native grid: 100 m, EPSG:32611 (UTM 11N), matching the `*_100m.tif` layers in
  `heat-risk-sd`.
- Output overlay: reprojected to EPSG:4326 for Leaflet, same as
  `annual_lst_4326.png`.
- Elevation datum: as delivered in `context/terrain/dem_*.tif` (confirm NAVD88).
- Distance to coast: `context/terrain/dist_coast_100m.tif`, km, positive inland.
- Temperature: °C throughout. SUHII reported to 0.1 °C.
- Domain: San Diego County land pixels; water masked.
- Time: Landsat overpass, approximately 10:30 local, clear-sky compositing
  window defined in Section 4 (decision D-3).

---

## 2. Inputs

All inputs exist in `../heat-risk-sd` at 100 m unless noted. No new acquisition
is required for v1.

| Symbol | Layer | File | Units | Role |
|---|---|---|---|---|
| `T_s` | Land-surface temperature | `exposure/geotiffs/heat_composites/lst_normal.tif` (+ regime variants) | °C | measurement |
| `T_s,night` | ECOSTRESS LST (incl. night) | `exposure/processed/ecostress/mosaics` | °C | v2 nighttime layer |
| `d` | Distance to coast | `context/terrain/dist_coast_100m.tif` | km | background covariate |
| `z` | Elevation | `context/terrain/dem_*.tif` | m | background covariate |
| `SVF` | Sky-view factor | `context/terrain/svf_100m.tif` | 0–1 | driver / covariate |
| `IMP` | Impervious fraction | `context/nlcd/impervious_100m.tif` | 0–1 | driver |
| `CAN` | Tree canopy | `context/nlcd/treecanopy_100m.tif`, CALFIRE `Canopy_2022_13.TIF` | 0–1 | driver (protective) |
| `ALB` | Albedo | `albedo/albedo_100m.tif`, `albedo/composites/albedo_JJA_median.tif` | 0–1 | driver |
| `HAG` | Height above ground | `context/terrain/hag_100m.tif` | m | driver / morphology |
| `z0` | Roughness length | `context/terrain/z0_100m.tif` | m | covariate |
| — | Land/water + reference mask | derive from NLCD land cover | class | masking |
| — | Mobile transect traverse | `data/transects/20260318_San_Diego_Transect` | °C, lat/lon | validation |

You already compute `lst_anomaly_*.tif` (santa_ana, coastal_heat,
dry_heat_dome, monsoonal). Those anomalies are deviations of LST across synoptic
regimes, not deviations from a coast/elevation background within one regime. The
SUHII residual defined below is a different quantity. Reconciling the two is
decision D-2.

---

## 3. The confound, stated concretely

San Diego's mean daytime LST rises roughly monotonically from coast to inland
over tens of km, driven by marine-layer cooling and the elevation gradient, not
by urbanization. A raw LST rank therefore labels coastal pavement "cool" and
inland chaparral "hot." The existing build script already observed this failure
mode: PCA on exposure variables produced an inland/coastal axis (r = -0.18
between daytime and nighttime exposure) rather than a heat axis. The
`Surface temperature` mode in the current viewer inherits the same problem,
because it percentile-ranks `mean_lst` directly.

SUHII removes the gradient by conditioning on `d` and `z`, so the residual
reflects land cover rather than geography.

---

## 4. Index definition

Let `i` index 100 m land pixels.

### Step 0 — Common grid

Resample all inputs to the 100 m UTM 11N grid. Continuous fields (`T_s`, `d`,
`z`, `IMP`, `CAN`, `ALB`, `SVF`) use bilinear; categorical land cover uses
nearest neighbor. Mask water and any pixel missing `T_s` after compositing.

### Step 1 — Background expectation

Fit the expected non-urban LST as a function of the two natural covariates.
Fit on **reference pixels only** (non-urban, non-water), then predict
everywhere:

```
reference set R = { i : IMP(i) < t_imp  and  land(i) != water }     (t_imp ≈ 0.05, decision D-1)

T_bg(i) = f(d(i), z(i))
```

Two candidate forms for `f`:

- Linear with curvature:
  `T_bg = β0 + β1·d + β2·z + β3·d² + β4·z²`
- Penalized smooth (GAM):
  `T_bg = β0 + s(d) + s(z)`, thin-plate splines.

The GAM is preferred because the coast-to-inland LST rise is nonlinear (steep in
the first ~15 km, flattening inland), and a smooth avoids imposing a functional
shape. Report the fit (R², residual std) and a partial-dependence plot of `s(d)`
and `s(z)` so the background is auditable.

Fitting on reference pixels (not all pixels) prevents dense urban cores from
biasing the background warm, which would shrink the very anomaly we are trying to
measure.

### Step 2 — SUHII raster

```
ΔT_s(i) = T_s(i) − T_bg(i)          [°C]
```

This is the deliverable raster. Interpretation: at this distance from the coast
and this elevation, the surface is `ΔT_s` °C hotter (or cooler) than natural
land. The field is signed; the color scale diverges about 0 (Section 6).

### Step 3 — Driver attribution

Regress the residual on the modifiable land-cover drivers, over all developed
pixels:

```
ΔT_s(i) = γ0 + γ1·IMP(i) + γ2·CAN(i) + γ3·ALB(i) + γ4·SVF(i) + ε(i)
```

Expected signs from surface energy balance: `γ1 > 0` (impervious warms),
`γ2 < 0` (canopy cools via shade and transpiration), `γ3 < 0` (higher albedo
reflects more shortwave), `γ4` sign depends on whether daytime trapping
(positive) or nighttime sky cooling dominates; for daytime LST expect `γ4`
small. Report coefficients in °C per unit driver (e.g. °C per 0.1 increase in
impervious fraction) with confidence intervals. If a sign contradicts the energy
balance, that is a data or collinearity problem to resolve, not a finding to
publish.

Do not run PCA on the drivers. Their physical signs are known; a data-driven
axis would re-extract the coastal gradient and destroy interpretability. A
transparent linear model with fixed-sign expectations is the correct tool here.

### Step 4 — Cooling potential (the actionable layer)

For each developed pixel, predict the SUHII reduction from a counterfactual
land-cover change, using the fitted `γ`:

```
ΔΔT_s(i) = γ2·(CAN_target − CAN(i)) + γ3·(ALB_target − ALB(i)) + γ1·(IMP_target − IMP(i))
```

with feasible targets (e.g. canopy raised to the 75th-percentile of comparable
land use, albedo to a cool-roof value). This mirrors what
`treatment_priority/el_cajon/cooling_potential_3m.tif` already does at 3 m, and
generalizes it county-wide at 100 m. Targets are a policy choice (decision D-5).

### Step 5 — Tract summary

For the existing tract viewer, summarize each tract `T`:

```
SUHII_mean(T) = mean_{i in T} ΔT_s(i)
SUHII_range(T) = p90 − p10 of ΔT_s over i in T
```

`SUHII_range` is reported alongside the mean because UHI lives below the tract:
a tract spanning a park and a strip mall has a wide range, and the mean alone
hides the hotspot. The 100 m raster overlay is the primary product; the tract
numbers are the secondary, tool-compatible summary.

---

## 5. Outputs

| Product | Format | Pipeline |
|---|---|---|
| SUHII intensity, 100 m | GeoTIFF (EPSG:32611) + PNG (EPSG:4326) + `*_bounds.json` | extend `build_lst_overlay.py` |
| Driver coefficients | table (°C per unit), JSON for legend | new |
| Cooling potential, 100 m | GeoTIFF + PNG | extend `build_lst_overlay.py` |
| Tract SUHII mean + range | CSV keyed by GEOID | merge in `build_diy_pca_viewer.py` `load_data()` |

The PNG-overlay path is already wired end-to-end (`build_lst_overlay.py`
produces turbo + colorblind PNGs and a bounds JSON; the viewer reads bounds at
build time and fetches PNGs at runtime). A diverging SUHII overlay is the same
pipeline with a diverging colormap and a symmetric range.

---

## 6. Color and the two-axis product

SUHII is signed, so the overlay uses a diverging map (cool/blue negative, warm/red
positive) anchored at 0 °C, with a symmetric range set to a robust quantile
(e.g. ±the 98th percentile of |ΔT_s|) to keep outliers from flattening the
midrange. Provide a colorblind-safe variant, as the LST overlay already does.

The heat-equity product is **bivariate**, not a blend: SUHII intensity on one
axis, the existing tract vulnerability score on the other, rendered as a 3×3
class grid. "High SUHII and high vulnerability" is the intervention-priority
class. Collapsing both into one percentile would hide which of the two drives a
tract's score, which is the question a planner is asking. The
`treatment_priority_100m.tif` raster you already have is the raster analog of
this. The bivariate layer is deferred to a second pass; v1 ships the physical
SUHII raster and the tract summary.

---

## 7. Daytime surface vs nighttime air

v1 is daytime surface UHI because Landsat compositing gives clean, repeated
clear-sky coverage. The health-relevant island is the nighttime air UHI, which
peaks overnight and correlates with lack of relief. Two paths for v2, not in
scope here:

- ECOSTRESS night mosaics give surface temperature at variable overpass times,
  including pre-dawn, allowing a nighttime SUHII with the same residual method.
- `mean_tmin_overnight` and `no_relief_fraction` (already in the tract tool) are
  air-based and can anchor an air-UHI proxy, but at tract resolution.

Label v1 explicitly as daytime surface UHI in the legend so it is not read as
the experienced nighttime island.

---

## 8. Validation

- **Transects.** `data/transects/20260318_San_Diego_Transect` is a mobile
  traverse with air temperature along a path. Compare the SUHII spatial pattern
  to transect anomalies along the same route. Expect spatial-pattern agreement,
  not value agreement, because transects measure air and SUHII measures surface.
  Report rank correlation along the route and call out where surface and air
  diverge (parks read cooler in surface than air; dry bare soil reads hot in
  surface, neutral in air).
- **Driver signs.** Confirm `γ1 > 0`, `γ2 < 0`, `γ3 < 0`. A wrong sign flags
  collinearity (e.g. impervious and albedo are correlated) needing a variance
  check or a partial-residual look.
- **Background fit.** Inspect `s(d)`, `s(z)` for monotonic, physically plausible
  shapes; report reference-set residual std as the noise floor of ΔT_s.
- **Cross-check** against your existing `lst_anomaly_*` rasters: the SUHII
  pattern should resemble the warm-regime anomaly in urban cores but differ in
  the coastal strip, where the regime anomaly carries marine signal that SUHII
  removes.

---

## 9. Known limitations

- Clear-sky bias. Landsat LST exists only for cloud-free overpasses; the coastal
  marine layer is systematically under-sampled, so coastal `T_bg` rests on fewer,
  warmer-than-typical scenes. Quantify scene counts per pixel and flag low-count
  coastal pixels.
- Surface ≠ air. Stated above; the legend must say so.
- Tract aggregation discards the within-tract hotspot; `SUHII_range` partly
  recovers it, the 100 m raster fully recovers it.
- Emissivity. LST retrieval assumes a land-cover emissivity; bare soil and dry
  vegetation bias high. Confirm how `lst_*.tif` was retrieved before trusting
  absolute °C.
- Collinearity among drivers (impervious, albedo, canopy co-vary) limits causal
  reading of individual `γ`. The cooling-potential layer is a first-order
  estimate, not a guaranteed intervention yield.
- Temporal mismatch. NLCD/CALFIRE canopy, albedo composites, and LST composites
  span different years. Record each layer's date; the index is a multi-year
  climatology, not a single-date snapshot.

---

## 10. Decisions needed before implementation

- **D-1.** Reference-pixel threshold `t_imp` for the background fit (proposed
  0.05 impervious). Also: exclude agriculture and irrigated turf from the
  reference set? Irrigated land reads anomalously cool and would warp `T_bg`.
- **D-2.** Reconcile with existing `lst_anomaly_*.tif`. Build SUHII on
  `lst_normal.tif`, or on a warm-season composite, or produce one SUHII per
  synoptic regime (santa_ana, coastal_heat, dry_heat_dome, monsoonal)?
- **D-3.** Compositing window: annual median, warm-season (JJA) median, or
  heat-event composite. Affects which `T_s` and which `albedo` composite feed in.
- **D-4.** Background covariate set: `d` and `z` only, or add latitude/SVF?
  Adding SVF risks absorbing part of the urban signal we want to keep.
- **D-5.** Cooling-potential targets (canopy/albedo/impervious goals). Policy
  choice; affects the actionable layer only, not SUHII itself.
- **D-6.** Viewer integration: replace the current `Surface temperature` exposure
  mode with the de-trended SUHII, add SUHII as a separate overlay toggle, or
  both. Replacing it fixes a misleading layer; adding it preserves the raw LST
  view for users who want it.

---

## References

- Oke, T.R. (1982). The energetic basis of the urban heat island.
  *Q. J. R. Meteorol. Soc.* 108, 1–24.
- Voogt, J.A., Oke, T.R. (2003). Thermal remote sensing of urban climate.
  *Remote Sens. Environ.* 86, 370–384.
- Stewart, I.D., Oke, T.R. (2012). Local Climate Zones for urban temperature
  studies. *Bull. Am. Meteorol. Soc.* 93, 1879–1900.

(Verify these against the specific claims they support before citing in any
public-facing text.)
