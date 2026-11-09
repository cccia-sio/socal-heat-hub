"""The composite is a COMMITTED artifact: only Build writes it.

Picking a heat mode used to call compute(), which silently re-ran all three
pillars and rewrote every tract's _comp. The map redrew the instant you touched
the radio, so it looked like the composite was live-calculating itself and there
was no way to tell that a rebuild had happened at all — you couldn't distinguish
"this is the composite I built" from "this is a composite the app built for me".

Meanwhile toggling a Sensitivity or Adaptive variable did the *opposite*: it
never recomputed and never said so, so the composite just went quietly stale.

So the rule, in both directions:

    * compute() is the ONLY thing that may write _comp / the pillar scores.
    * Every change to an input marks the composite stale and highlights Build.
    * Previewing a heat mode reads its score live and writes nothing.
    * A scenario switch re-scores the COMMITTED build under the new scenario.
      It used to call a bare compute(), which quietly applied any pending
      variable or heat-mode changes and cleared the stale flag.
    * Anything that describes or re-scores the build (radar, detail drawer,
      tract PDF, export manifest, scenario compare) reads the settings saved on
      `results`, never the live panel. While the composite is stale the two
      differ, and reading the panel labels last build's scores with this
      build's heat mode and variables.
"""

import re

import pytest


def test_selecting_a_heat_mode_does_not_rebuild_the_composite(js_code):
    """The headline regression. selectHeatMode() must never call compute()."""
    src = js_code("selectHeatMode")
    assert not re.search(r"\bcompute\s*\(\s*\)", src), (
        "selectHeatMode() calls compute() again. Picking a heat mode then silently "
        "rewrites every tract's composite, so the map looks like it is "
        "live-calculating and the user cannot tell a rebuild happened. It must "
        "preview the mode and mark the composite stale instead."
    )
    assert "previewHeatMode()" in src, "selectHeatMode() no longer previews the picked mode."
    assert "markCompositeStale()" in src, (
        "selectHeatMode() no longer marks the composite stale, so the Build button "
        "won't tell the user their choice isn't in the map yet."
    )


def test_variable_toggles_mark_the_composite_stale(js_code):
    """The opposite bug: these never recomputed and never said so."""
    for fn in ("tog", "togPillar"):
        assert "markCompositeStale()" in js_code(fn), (
            f"{fn}() changes the enabled variables without marking the composite stale. "
            "The map would keep showing a composite built from a different variable set, "
            "with nothing on screen saying so."
        )


def test_build_is_the_only_writer_of_the_composite(viewer_html_text):
    """`_comp` is assigned in exactly one place: compute()."""
    writers = [
        ln.strip()
        for ln in viewer_html_text.splitlines()
        if re.search(r"\bpr\._comp\s*=(?!=)", ln) and not ln.lstrip().startswith("//")
    ]
    assert len(writers) == 1, (
        f"_comp is written in {len(writers)} places: {writers}. The composite must only "
        "ever be written by compute(), or it can change without the user pressing Build."
    )


def test_building_clears_the_stale_flag(js_code):
    src = js_code("compute")
    assert "compositeStale=false" in src.replace(" ", ""), (
        "compute() no longer clears the stale flag, so the Build button would stay "
        "amber forever after the first change."
    )
    assert "syncBuildBtn()" in src, "compute() no longer refreshes the Build button."


def test_building_commits_the_heat_mode_preview(js_code):
    """After Build the composite must actually appear — the thing the user asked for.

    Without this the map stays on the heat-mode preview after a rebuild, still
    captioned "click Build to fold this into the composite", and the composite
    never pops up.
    """
    src = js_code("compute")
    assert re.search(r'curLayer\s*===?\s*"heatmode"\s*\)\s*curLayer\s*=\s*"composite"', src), (
        "compute() no longer switches out of the heat-mode preview, so clicking Build "
        "leaves the user staring at the preview instead of the composite they just built."
    )


def test_heat_mode_preview_is_read_only(js_code):
    """The preview must not write pillar scores — that's what made it look auto-built.

    It reads expModeScores() (the same score that becomes Heat Exposure on Build),
    so what you preview is what you get, but nothing is persisted to the features.
    """
    src = js_code("previewHeatMode")
    assert "expModeScores()" in src, (
        "previewHeatMode() no longer reads the live per-mode score, so the preview may "
        "not match what Build will actually produce."
    )
    for written in ("_exposure", "_comp", "_sensitivity", "_adaptive"):
        assert not re.search(rf"\.{written}\s*=", src), (
            f"previewHeatMode() writes {written} onto the features. Previewing must not "
            "mutate the committed composite — that is the bug this whole flow removed."
        )


def test_preview_layer_does_not_read_the_stale_pillar_score(js_code):
    """The 'heatmode' layer must colour from expModeScores(), not pr._exposure.

    pr._exposure belongs to the last build and may be a completely different mode,
    so reading it would show the OLD mode's map under the NEW mode's name.
    """
    src = js_code("fcolor")
    branch = re.search(r'if\(curLayer==="heatmode"\)\{(.*?)\}\s*if\(curLayer==="composite"', src, re.S)
    assert branch, "fcolor() has no 'heatmode' preview branch."
    body = branch.group(1)
    assert "expModeScores()" in body, (
        "the heat-mode preview layer no longer reads expModeScores()."
    )
    assert "_exposure" not in body, (
        "the heat-mode preview colours from pr._exposure, which is the LAST BUILD's "
        "pillar score — very possibly a different heat mode. It would show the old "
        "map under the new mode's name."
    )


def test_stale_state_is_visible_on_the_button(js_code, viewer_html_text):
    """A silent stale flag is no better than a silent rebuild."""
    src = js_code("syncBuildBtn")
    assert "needs-build" in src, "syncBuildBtn() no longer toggles the stale class."
    assert "Rebuild" in src, (
        "the Build button no longer relabels when stale, so nothing tells the user "
        "their pending changes aren't in the map."
    )
    assert "#compute-btn.needs-build{" in viewer_html_text, (
        "the .needs-build style rule is gone — the button would relabel but look "
        "identical, which is the failure mode this whole feature exists to prevent."
    )


# ── The build's own settings ────────────────────────────────────────────────
# The panel (heatMode, enabled) is what the user is editing; results.settings is
# what the map was built from. After any unbuilt change they disagree.


def test_build_saves_the_settings_it_scored(js_code):
    src = js_code("compute")
    assert re.search(r"\.settings\s*=\s*settings", src), (
        "compute() no longer saves the settings it scored on `results`, so nothing "
        "can tell which heat mode and variables the map was built from."
    )
    assert re.search(r"exposureScore\(\s*settings\.heatMode\s*\)", src), (
        "compute() scores exposure from something other than the settings it saves."
    )


def test_build_clears_the_previous_builds_pillar_fields(js_code):
    """A pillar left out of a rebuild used to keep its old score on every tract.

    compute() skips pillars with no variables, so `_sensitivity` and its PCs from
    the last build survived, and the radar, detail drawer, compare views and tract
    PDF went on showing them next to a composite that no longer included them.
    """
    assert re.search(r"delete\s+pr\[", js_code("compute")), (
        "compute() no longer clears the previous build's pillar fields before "
        "writing, so an emptied pillar keeps showing its old score."
    )


def test_build_drops_a_layer_it_no_longer_produces(js_code):
    """Viewing Sensitivity Pattern 3, then rebuilding with fewer variables, drew every tract grey.

    The new build had fewer patterns, but curLayer still pointed at the old one and
    fcolor() found no `_sensitivity_pc3` to color by.
    """
    src = js_code("compute")
    fallback = [ln for ln in src.splitlines() if "pcs.length" in ln and 'curLayer="composite"' in ln]
    assert fallback, (
        "compute() no longer falls back to the composite when the active pillar "
        "layer isn't in the new build, so the map draws every tract grey."
    )
    assert src.index(fallback[0]) < src.index("drawTracts()"), (
        "compute() resets the layer after drawing; the grey map is already on screen."
    )


def test_scenario_switch_rescores_the_committed_build(js_code):
    src = js_code("setScenario")
    assert not re.search(r"\bcompute\(\s*\)", src), (
        "setScenario() calls a bare compute() again. That is a Build: it applies "
        "pending checkbox and heat-mode changes and clears the stale flag, so the "
        "scenario buttons become a hidden 'apply'."
    )
    assert re.search(r"compute\(\s*results\.settings\s*\)", src), (
        "setScenario() no longer re-scores the committed build under the new scenario."
    )


def test_rescoring_leaves_the_stale_flag_alone(js_code):
    """Re-scoring for a new scenario doesn't apply the panel, so it isn't a Build."""
    src = js_code("compute")
    assert re.search(r"if\(\s*!rescore\s*\)\s*compositeStale\s*=\s*false", src), (
        "compute() clears the stale flag on a scenario re-score too, hiding the "
        "user's pending changes."
    )


@pytest.mark.parametrize(
    "fn",
    [
        "dtPillarKeys",
        "openDetailDrawer",
        "_tractPdfBuild",
        "radarSpokes",
        "radarSVG",
        "_exportManifest",
        "computeCompositeFor",
        "computeExposureFor",
        "exposureScore",
    ],
)
def test_built_scores_are_described_by_the_build_not_the_panel(js_code, fn):
    """Each of these labels or re-scores the build, so must not read the live panel.

    Build with Daytime, pick Nighttime, export: the tract PDF's exposure card was
    the daytime score under a "Nighttime heat" header, and the export manifest
    said `nighttime` above daytime `exposure_score` columns.
    """
    src = js_code(fn)
    live = [
        p
        for p in (r"\benabled\s*\[", r"heatModeDef\(\s*heatMode\s*\)", r"===\s*heatMode\b", r"\[\s*heatMode\s*\]")
        if re.search(p, src)
    ]
    assert not live, (
        f"{fn}() reads the live panel ({', '.join(live)}). While the composite is "
        "stale the panel no longer matches the build, so this would describe the "
        "last build's scores with settings they weren't built from. Read "
        "_builtSettings() instead."
    )


# ── The composite is a percentile ───────────────────────────────────────────
# Everything else in the tool is a county percentile and the guide says the
# index is one. A plain mean of three percentiles isn't: it bunches around 50
# and a "75" was really the 99th percentile.


@pytest.mark.parametrize("fn", ["compute", "computeCompositeFor"])
def test_composite_is_ranked_not_just_averaged(js_code, fn):
    assert re.search(r"pctRank\(\s*avg\s*\)", js_code(fn)), (
        f"{fn}() writes the raw mean of the pillar scores as the composite. It "
        "must percentile-rank the means across the county, or a composite of 75 "
        "isn't the 75th percentile the guide promises."
    )
