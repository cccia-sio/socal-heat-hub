"""Structural checks on the "Download Map as PDF" export's variable box.

A built map is exported with a box over its lower-left corner listing the
variables behind it, by pillar, so a figure lifted into a grant or report
carries its own inputs.

Static guards on the generated viewer, not a render test. The failure modes are
silent ones:

* The box must describe the MAP, not the checkboxes. Toggling a variable after
  Build marks the composite stale but leaves the map on the old build, so reading
  `enabled[...]` would print a variable list that doesn't match the figure.
* A raw-variable or heat-mode preview has no build behind it; listing the last
  build's inputs next to it would mislabel the figure.
* A pillar with more variables than fit must say so rather than truncate silently.
"""

import re


def test_map_pdf_draws_the_variable_box(js_code):
    assert "_pdfVarBox(" in js_code("_heroPdf"), (
        "_heroPdf() no longer draws the selected-variables box."
    )


def test_box_lists_the_committed_build_not_the_live_checkboxes(js_code):
    src = js_code("_pdfVarBox")
    assert "results" in src and ".keys" in src, (
        "_pdfVarBox() should list results[pillar].keys — the inputs of the build "
        "the map is showing."
    )
    assert not re.search(r"\benabled\s*\[", src), (
        "_pdfVarBox() reads the live `enabled` checkboxes. After a toggle the "
        "composite is stale and the map still shows the previous build, so the "
        "printed list would not match the figure."
    )


def test_box_is_skipped_for_previews(js_code):
    src = js_code("_heroPdf")
    assert "rawVar" in src and '"heatmode"' in src.replace("'", '"'), (
        "_heroPdf() must skip the variable box for raw-variable and heat-mode "
        "previews, which have no build behind them."
    )


def test_box_overflow_is_stated(js_code):
    assert " more" in js_code("_pdfVarBox"), (
        "_pdfVarBox() must say how many variables it left out when a pillar's "
        "list doesn't fit, never truncate silently."
    )


def test_box_stacks_on_the_scale_bar(js_fn):
    """The box is placed from the scale bar's rect, which only the canvas knows.

    If the renderer stops publishing it, the box falls back to the map's corner
    and lands on top of the scale bar.
    """
    assert "canvas._scaleBarRect=" in js_fn("_captureStudyAreaMap"), (
        "drawScaleBar() no longer publishes canvas._scaleBarRect."
    )
    assert "_scaleBarRect" in js_fn("_heroPdf"), (
        "_heroPdf() no longer anchors the variable box to the scale bar."
    )


# ── Framing ─────────────────────────────────────────────────────────────────
# Leaflet snaps fitBounds to a whole zoom level by default, which left the county
# filling ~70% of the exported frame. The fit is now exact, so the basemap has to
# be painted at a fractional zoom, and the furniture has to be planned for.


def test_map_fit_is_not_snapped_to_a_whole_zoom(js_fn):
    assert "zoomSnap:0" in js_fn("_captureStudyAreaMap"), (
        "The export's projection map snaps to whole zoom levels again; the county "
        "will shrink back to ~70% of the frame."
    )


def test_basemap_tiles_are_requested_at_a_whole_zoom(js_code):
    """A fractional z in a tile URL 404s, and the export ships a blank basemap."""
    src = js_code("_paintLightBasemap")
    assert re.search(r"var rz=Math\.round\(zoom\)", src), (
        "_paintLightBasemap() must fetch tiles at a whole zoom level and scale them "
        "onto the (possibly fractional) display zoom."
    )


def test_fit_reserves_room_for_the_legend(js_fn):
    src = js_fn("_captureStudyAreaMap")
    assert "drawLegend(true)" in src and "paddingBottomRight:[leg?" in src, (
        "The fit no longer measures the legend and reserves its column, so a "
        "tight frame puts tracts under it."
    )


def test_every_exported_layer_gets_a_legend_title(js_fn):
    """The map PDF's header names no layer, so the in-map legend must.

    The heat-mode preview had no branch in the export legend; with the old header
    subtitle gone, its PDF would have carried no label at all.
    """
    assert "curLayer==='heatmode'" in js_fn("_captureStudyAreaMap"), (
        "The export legend has no heat-mode-preview branch."
    )
