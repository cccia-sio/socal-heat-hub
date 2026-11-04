#!/usr/bin/env python3
"""User-guide hero figure: composite Heat Risk Index + dominant risk driver.

Reproduces the viewer's **default view** — the map a user sees on opening
`main/heat_risk_map.html` before touching a control — so the figure in the user
guide matches the tool it illustrates:

  - Heat Exposure   : the Daytime heat mode, scored as the mean of its
                      z-scored member layers, percentile-ranked (NOT PCA).
  - Sensitivity     : PCA over the viewer's three default-on variables.
  - Adaptive Cap.   : PCA over the viewer's three default-on variables.
  - Composite       : (exposure + sensitivity + (100 - adaptive)) / 3,
                      percentile-ranked
  - Dominant driver : argmax(exposure, sensitivity, 100 - adaptive)

The scoring functions below are direct ports of `exposureScore()`,
`pillarPCA()` and `compute()` in `build_diy_pca_viewer.py`'s HTML_TEMPLATE,
down to the sample-std z-scoring, the tie-averaged percentile ranks, and JS
`Math.round` half-up semantics. Tract data comes from that script's
`load_data()`, so both read the same CSVs.

PC ORIENTATION
--------------
Eigenvector signs are arbitrary, so any PC past the first can enter the
variance-weighted blend with inverted risk polarity, cancelling risk instead of
adding it. Both this script and `pillarPCA()` orient each retained PC to the
pillar's risk reference before weighting. The viewer used to anchor only the
finished blend, which made multi-PC pillars (sensitivity at the default
selection) diverge from this figure; that was fixed in `pillarPCA()`. The run
logs each pillar's PC count.

Usage:
    python3 scripts/build_hero_figure.py
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_diy_pca_viewer import (  # noqa: E402
    HEAT_MODES,
    INVERTED_VARS,
    VARIABLES,
    load_data,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "userguide" / "figure_hero_composite.png"

# Mirrors DEFAULT_ON in the viewer template. Exposure is absent by design: it is
# a single-select heat mode, not a variable toggle set.
DEFAULT_ON = {
    "sensitivity": ["RPL_THEMES", "EP_AGE65"],
    "adaptive": ["ac_prob", "treecanopy_pctile", "income_pctile"],
}
DEFAULT_MODE = "daytime"

# Sign-reference variable per pillar (viewer's SREF). Missing from the enabled
# set -> the template falls back to index 0; _pillar_pca does the same.
SREF = {
    "exposure": "mean_tmax",
    "sensitivity": "bphigh_prev",
    "adaptive": "income_pctile",
}

# Viewer palettes (default, non-colorblind): DRGB for the driver layer, CR for
# the composite ramp. Keep in step with build_diy_pca_viewer.py.
DRGB = {
    "exposure": np.array([192, 57, 43], float),
    "sensitivity": np.array([106, 81, 163], float),
    "adaptive": np.array([22, 160, 82], float),
}
CR = np.array(
    [[255, 255, 178], [254, 204, 92], [253, 141, 60], [240, 59, 32], [189, 0, 38]],
    float,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ports of the viewer's numeric helpers
# ---------------------------------------------------------------------------
def _js_round(x, decimals=0):
    """JS `Math.round` (half away from zero); numpy's rint is half-to-even."""
    f = 10.0**decimals
    return np.floor(np.asarray(x, float) * f + 0.5) / f


def pct_rank(x):
    """Port of pctRank(): tie-averaged rank as a 0-100 percentile, 1 decimal."""
    x = np.asarray(x, float)
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return _js_round(ranks / len(x) * 100.0, 1)


def zscore(x):
    """Port of the template's z-scoring: sample std (N-1), tiny-sd guard."""
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    if sd < 1e-10:
        sd = 1.0
    return (x - x.mean()) / sd


def polarity_map():
    """Port of the POLARITY dict main() bakes into the template."""
    return {
        key: (pillar in ("exposure", "sensitivity")) ^ (key in INVERTED_VARS)
        for key, _label, pillar, _sub in VARIABLES
    }


def jacobi_eigen(A):
    """Port of jacobiEigen(): cyclic Jacobi rotation for symmetric matrices.

    numpy's `eigh` would give the same eigenvalues, but eigenvector *sign* is
    solver-specific and arbitrary. A pillar retaining more than one PC blends
    raw (not yet sign-anchored) components, so a flipped PC2 silently rewrites
    the score — measured at r=0.33 against the viewer before this port existed.
    Mirroring the template's routine keeps the signs identical by construction.
    """
    A = np.asarray(A, float)
    n = len(A)
    if n == 0:
        return np.array([]), np.zeros((0, 0))
    if n == 1:
        return np.array([A[0][0]]), np.ones((1, 1))

    S = A.copy()
    V = np.eye(n)
    for _sweep in range(100):
        off = sum(S[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off < 1e-20:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(S[p][q]) < 1e-15:
                    continue
                tau = (S[q][q] - S[p][p]) / (2 * S[p][q])
                if abs(tau) > 1e15:
                    t = 1 / (2 * tau)
                else:
                    t = (1 if tau >= 0 else -1) / (abs(tau) + np.sqrt(1 + tau * tau))
                c = 1 / np.sqrt(1 + t * t)
                s = t * c
                Spq = S[p][q]
                S[p][p] -= t * Spq
                S[q][q] += t * Spq
                S[p][q] = S[q][p] = 0.0
                for r in range(n):
                    if r in (p, q):
                        continue
                    rp, rq = S[r][p], S[r][q]
                    S[r][p] = S[p][r] = c * rp - s * rq
                    S[r][q] = S[q][r] = s * rp + c * rq
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq

    return np.diag(S).copy(), V


def _interp_c(stops, t):
    """Port of interpC(): piecewise-linear over equally spaced RGB stops."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    idx = t * (len(stops) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(stops) - 1)
    f = (idx - lo)[:, None]
    return np.floor(stops[lo] * (1 - f) + stops[hi] * f + 0.5)


# ---------------------------------------------------------------------------
# Pillar scores
# ---------------------------------------------------------------------------
def exposure_score(df, mode_key=DEFAULT_MODE):
    """Port of exposureScore(): mean of the mode's z-scored layers, ranked."""
    mode = next(m for m in HEAT_MODES if m[0] == mode_key)
    _key, label, layers, _desc = mode
    keys = [c for c in layers if c in df.columns and df[c].notna().any()]
    if not keys:
        raise SystemExit(f"No usable layers for heat mode '{mode_key}'")

    acc = np.zeros(len(df))
    for k in keys:
        acc += zscore(df[k].fillna(0).to_numpy(float))
    acc /= len(keys)

    logger.info(f"  Exposure: {label} — {len(keys)} layers ({', '.join(keys)})")
    return pct_rank(acc), label


def pillar_pca(df, pid, keys):
    """Port of pillarPCA(): z-score, polarity-align, PCA, variance-weight, rank."""
    pol = polarity_map()
    keys = [k for k in keys if k in df.columns]
    if not keys:
        raise SystemExit(f"No usable variables for pillar '{pid}'")

    X = np.column_stack([zscore(df[k].fillna(0).to_numpy(float)) for k in keys])

    # Point every column the same way as its pillar's risk semantics, so a
    # protective variable does not get inverted twice.
    pillar_higher_is_risk = pid in ("exposure", "sensitivity")
    for j, k in enumerate(keys):
        if pol[k] != pillar_higher_is_risk:
            X[:, j] = -X[:, j]

    if len(keys) == 1:
        logger.info(f"  {pid.title()}: 1 variable ({keys[0]}) — no PCA")
        return pct_rank(X[:, 0])

    evals, evecs = jacobi_eigen(np.cov(X, rowvar=False, ddof=1))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    vr = np.maximum(evals, 0) / np.maximum(evals, 0).sum()
    nret = max(1, int((evals > 1).sum()))

    scores = X @ evecs
    ri = keys.index(SREF[pid]) if SREF[pid] in keys else 0
    rc = X[:, ri]

    # Orient EACH retained PC to the pillar's risk reference before weighting.
    # Eigenvector signs are arbitrary, so an unanchored PC can enter the blend
    # with inverted polarity and cancel risk instead of adding it. pillarPCA()
    # in the viewer does the same, so the two agree.
    for k in range(nret):
        if np.corrcoef(rc, scores[:, k])[0, 1] < 0:
            scores[:, k] = -scores[:, k]

    weights = vr[:nret] / vr[:nret].sum()
    w = (scores[:, :nret] * weights).sum(axis=1)
    if np.corrcoef(rc, w)[0, 1] < 0:
        w = -w

    logger.info(
        f"  {pid.title()}: {len(keys)} variables, {nret} PC(s) retained, "
        f"{vr[:nret].sum() * 100:.1f}% variance, sign-anchored to {keys[ri]}"
    )
    return pct_rank(w)


def build_composite(tracts):
    """Port of compute(): pillar scores -> composite risk + dominant driver."""
    for pid, keys in DEFAULT_ON.items():
        missing = [k for k in keys if k not in tracts.columns]
        if missing:
            logger.warning(f"  {pid}: missing default variable(s) {missing} — skipped")

    exposure, mode_label = exposure_score(tracts)
    sensitivity = pillar_pca(tracts, "sensitivity", DEFAULT_ON["sensitivity"])
    adaptive = pillar_pca(tracts, "adaptive", DEFAULT_ON["adaptive"])

    out = tracts[["GEOID", "geometry"]].copy()
    out["exposure_score"] = exposure
    out["sensitivity_score"] = sensitivity
    out["adaptive_score"] = adaptive
    out["adaptive_deficit"] = 100.0 - adaptive
    # Mean of the pillar percentiles, ranked again so the composite is one too.
    out["composite_risk"] = pct_rank((exposure + sensitivity + (100.0 - adaptive)) / 3.0)

    e, s, a = exposure, sensitivity, 100.0 - adaptive
    out["driver"] = np.where(
        (e >= s) & (e >= a), "exposure", np.where(s >= a, "sensitivity", "adaptive")
    )
    return out, mode_label


def driver_colors(composite):
    """Port of fcolor()'s driver branch: pillar hue whitened by composite risk."""
    intensity = 0.15 + 0.85 * (composite["composite_risk"].to_numpy() / 100.0)
    base = np.array([DRGB[d] for d in composite["driver"]])
    rgb = 255.0 - intensity[:, None] * (255.0 - base)
    return np.clip(rgb / 255.0, 0, 1)


def composite_colors(composite):
    """Port of fcolor()'s composite branch: interpC(CR, (comp - 2) / 96)."""
    t = (composite["composite_risk"].to_numpy() - 2.0) / 96.0
    return np.clip(_interp_c(CR, t) / 255.0, 0, 1)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_hero(composite, mode_label, output_path):
    """Two-panel hero: composite risk (left) + dominant driver (right).

    Colorbar and legend sit directly below each map's rendered extent, measured
    after `apply_aspect()` so the equal-aspect shrink is already committed and
    no dead vertical space opens up between map and caption.
    """
    fig = plt.figure(figsize=(14, 7.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04, wspace=0.04)
    ax_comp = fig.add_subplot(1, 2, 1)
    ax_drv = fig.add_subplot(1, 2, 2)

    bounds = composite.total_bounds
    x_pad = (bounds[2] - bounds[0]) * 0.025
    y_pad = (bounds[3] - bounds[1]) * 0.025
    xlim = (bounds[0] - x_pad, bounds[2] + x_pad)
    ylim = (bounds[1] - y_pad, bounds[3] + y_pad)

    # ---- Left: composite risk ----
    composite.plot(ax=ax_comp, color=composite_colors(composite),
                   edgecolor="#3a3a3a", linewidth=0.18)
    ax_comp.set_title("Composite Heat Risk Index", fontsize=18,
                      fontweight="bold", color="#2c3e50", pad=10)
    ax_comp.set_xlim(xlim)
    ax_comp.set_ylim(ylim)
    ax_comp.set_aspect("equal")
    ax_comp.axis("off")

    # ---- Right: dominant driver ----
    composite.plot(ax=ax_drv, color=driver_colors(composite),
                   edgecolor="#3a3a3a", linewidth=0.18)
    ax_drv.set_title("Dominant Risk Driver", fontsize=18,
                     fontweight="bold", color="#2c3e50", pad=10)
    ax_drv.set_xlim(xlim)
    ax_drv.set_ylim(ylim)
    ax_drv.set_aspect("equal")
    ax_drv.axis("off")
    ax_drv.annotate("Lighter shade = lower composite risk",
                    xy=(0.98, 0.015), xycoords="axes fraction", fontsize=9,
                    ha="right", va="bottom", fontstyle="italic", color="#666666")

    ax_comp.apply_aspect()
    ax_drv.apply_aspect()
    pos_comp = ax_comp.get_position()
    pos_drv = ax_drv.get_position()

    # ---- Colorbar: directly below the left map ----
    gap, cbar_height = 0.012, 0.025
    cbar_width = pos_comp.width * 0.7
    cax = fig.add_axes([
        pos_comp.x0 + (pos_comp.width - cbar_width) / 2,
        pos_comp.y0 - gap - cbar_height,
        cbar_width,
        cbar_height,
    ])
    # The map colors tracts via interpC over the full 0-100 percentile range, so
    # the bar spans 0-100 too rather than the data's own min/max.
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("viewer_CR", CR / 255.0),
        norm=plt.Normalize(vmin=0, vmax=100),
    )
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=9)
    cbar.locator = mticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.set_label("Composite risk percentile", fontsize=10, labelpad=4)

    # ---- Legend: directly below the right map ----
    # "Low Adaptive Capacity" rather than the viewer's bare "Adaptive Capacity":
    # green marks a shortfall in cooling resources, and a static figure has no
    # tooltip to carry that qualifier.
    legend_items = [
        mpatches.Patch(facecolor=DRGB["exposure"] / 255.0, edgecolor="#777",
                       linewidth=0.5, label="Heat Exposure"),
        mpatches.Patch(facecolor=DRGB["sensitivity"] / 255.0, edgecolor="#777",
                       linewidth=0.5, label="Population Sensitivity"),
        mpatches.Patch(facecolor=DRGB["adaptive"] / 255.0, edgecolor="#777",
                       linewidth=0.5, label="Low Adaptive Capacity"),
    ]
    fig.legend(handles=legend_items, loc="upper center",
               bbox_to_anchor=(pos_drv.x0 + pos_drv.width / 2, pos_drv.y0 - gap),
               ncol=3, fontsize=10, frameon=False, handlelength=1.6,
               handleheight=1.2, columnspacing=1.8, labelspacing=0.4)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved → {output_path}")


def main():
    logger.info("Loading tract data (shared with the viewer build)...")
    tracts = load_data()

    logger.info("Scoring pillars at the viewer's default selection...")
    composite, mode_label = build_composite(tracts)

    counts = composite["driver"].value_counts()
    logger.info(
        f"  Dominant driver: {counts.get('exposure', 0)} exposure, "
        f"{counts.get('sensitivity', 0)} sensitivity, "
        f"{counts.get('adaptive', 0)} low-adaptive-capacity "
        f"({len(composite)} tracts)"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_hero(composite, mode_label, OUTPUT_PATH)


if __name__ == "__main__":
    main()
