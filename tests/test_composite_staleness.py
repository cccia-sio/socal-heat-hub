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

Known gap, deliberately not asserted here: setScenario() still calls compute()
when a composite exists, so switching to SSP5-8.5 silently rebuilds — the same
bug, and it makes the scenario buttons a hidden "apply" for pending variable
changes. Left as-is pending a product decision; do not enshrine it in a test.
"""

import re


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
