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

import json
import re

import pytest


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
    src = js_code("_paintBasemap")
    assert re.search(r"var rz=Math\.round\(zoom\)", src), (
        "_paintBasemap() must fetch tiles at a whole zoom level and scale them "
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


# ── Jurisdiction names ──────────────────────────────────────────────────────
# With several cities on, a legend listing their names next to identical black
# squares couldn't say which outline was which. The cities and the unincorporated
# county are named on the map instead, the cities at the live map's label points.
# The legend keeps the layer's color key; the variable box keeps the inputs.


def _juris(viewer_html_text, var):
    match = re.search(rf"\bvar {var}\s*=\s*", viewer_html_text)
    return json.JSONDecoder().raw_decode(viewer_html_text, match.end())[0]


def test_map_names_the_visible_jurisdictions(js_fn):
    src = js_fn("_captureStudyAreaMap")
    assert "drawJurisLabels()" in src, "The map export no longer names the jurisdictions."
    assert "jurisLabels[k].getLatLng()" in src, (
        "drawJurisLabels() should place each city's name at the live map's label "
        "point, so the PDF matches what's on screen."
    )
    assert "'Unincorporated county'" in src, (
        "drawJurisLabels() no longer names the unincorporated county."
    )


def test_one_jurisdiction_export_draws_no_name(js_fn):
    """A one-jurisdiction PDF names it in the title; a label would cover its tracts.

    The city-focus PDF is the one-city case of the same scope.
    """
    src = js_fn("_captureStudyAreaMap")
    assert "if(!scope||scope.names.length>1)drawJurisLabels();" in src
    assert "opts.jurisName?{cities:[opts.jurisName]" in src, (
        "The city-focus PDF no longer runs through the scoped export."
    )


def test_names_dont_overprint(js_fn):
    """Coronado's and National City's points are ~36 px apart at county scale."""
    assert "hits(" in js_fn("_captureStudyAreaMap"), (
        "drawJurisLabels() no longer moves a name that lands on one already drawn."
    )


def test_legend_does_not_list_the_named_jurisdictions(js_fn):
    src = js_fn("_captureStudyAreaMap")
    assert "cityRows" not in src and "text:'Unincorporated'" not in src, (
        "The export legend lists the cities or the unincorporated county again; "
        "they're named on the map."
    )


def test_unincorporated_county_is_outlined_not_tinted(js_fn):
    """An indigo wash over the east county shifts its risk colors off the legend."""
    assert "drawFeature(f,null,UNINCORP_COLOR," in js_fn("_captureStudyAreaMap"), (
        "The map export fills the unincorporated county again; the tracts under it "
        "no longer print in the legend's colors."
    )


def test_map_pdf_still_lists_the_inputs_by_pillar(js_fn):
    """The box over the lower-left corner, not the legend, carries the pillars."""
    assert "_pdfVarBox(" in js_fn("_heroPdf")


@pytest.mark.parametrize("var,key", [("JURIS", "NAME"), ("UNINCORP", "name")])
def test_label_points_are_inside_their_shape(viewer_html_text, var, key):
    """A bounding-box center can miss the shape: Coronado's is in the bay."""
    shape = pytest.importorskip("shapely.geometry").shape
    outside = [
        f["properties"][key]
        for f in _juris(viewer_html_text, var)["features"]
        if not shape(f["geometry"]).contains(
            shape({"type": "Point", "coordinates": [f["properties"]["label_lng"], f["properties"]["label_lat"]]})
        )
    ]
    assert not outside, f"Labels placed outside their shape: {outside}"


# ── Clipping to jurisdictions ───────────────────────────────────────────────
# "Download Map as PDF" opens a dialog: the whole county (the regional snapshot)
# or the map cut to the jurisdictions ticked there, any mix of cities, the
# unincorporated county and tribal lands.


def test_sidebar_button_opens_the_pdf_dialog(viewer_html_text):
    btn = re.search(r'<button id="dl-pdf-btn"[^>]*>', viewer_html_text).group(0)
    assert 'onclick="openPdfModal()"' in btn, (
        "The sidebar PDF button exports straight away again; it should open the "
        "dialog that offers the clipped map."
    )


@pytest.mark.parametrize("source", ["JURIS", "UNINCORP", "tribalGroups"])
def test_dialog_lists_every_kind_of_jurisdiction(js_fn, source):
    assert source in js_fn("_pdfRenderJurisList"), f"The PDF dialog no longer lists {source}."


def test_clipping_to_nothing_is_blocked(js_code):
    """An empty selection would print a map with no tracts on it."""
    assert "pdf-download-btn" in js_code("_pdfSyncScope")


def test_clip_is_a_mask_not_one_evenodd_path(js_code):
    """One even-odd clip path over several shapes turns every overlap into a hole.

    Cities are simplified one at a time, so neighbours overlap in slivers along
    their shared border; ticking both would print a strip of basemap between them.
    """
    src = js_code("_captureStudyAreaMap")
    assert "destination-in" in src, "The scoped export no longer masks the tracts."
    assert ".clip(" not in src, "The scoped export clips with a single path again."


def test_scoped_export_outlines_only_the_selection(js_code):
    src = js_code("_captureStudyAreaMap")
    for check in ("scope.cities.indexOf(k)", "scope.unincorp", "scope.tribal.indexOf(tribalGroupKey(k))"):
        assert check in src, f"The scoped export no longer filters its outlines with {check}."


def test_clipped_map_reserves_a_column_for_the_box(js_code):
    """The county framing parks the variable box over open ocean; a city has none."""
    assert "padLeft:box?" in js_code("downloadMapPdf"), (
        "The clipped map no longer reserves the left column, so the variable box "
        "and scale bar cover the selection."
    )


def test_clipped_pdf_names_its_jurisdictions(js_code):
    assert "opts.clipNames" in js_code("_heroPdf"), "The clipped PDF's title no longer names the selection."
    assert " more" in js_code("_pdfNameList"), (
        "A selection too long for the title must say how many it left out."
    )


def test_escape_closes_the_pdf_dialog(viewer_html_text):
    assert re.search(r"e\.key==='Escape'&&document\.getElementById\('pdf-modal'\)", viewer_html_text)


# ── Basemap ─────────────────────────────────────────────────────────────────
# The dialog also picks the PDF's basemap: the light map (the default, and what
# every export printed before), satellite imagery, or none.


@pytest.mark.parametrize("name", ["light", "satellite", "none"])
def test_dialog_offers_the_basemap(viewer_html_text, name):
    assert f'name="pdf-basemap" value="{name}"' in viewer_html_text, (
        f"The PDF dialog no longer offers the {name} basemap."
    )


def test_chosen_basemap_reaches_the_render(js_code):
    assert "pdf-basemap" in js_code("doPdfDownload"), "doPdfDownload() ignores the basemap choice."
    assert "basemap:basemap" in js_code("downloadMapPdf"), "downloadMapPdf() drops the basemap choice."


def test_attribution_follows_the_basemap(js_code):
    """Satellite tiles are Maxar/Earthstar imagery; printing the light map's credit under them is wrong."""
    src = js_code("_captureStudyAreaMap")
    assert "BASEMAPS[bm].opts.attribution" in src
    assert "HERE, Garmin" not in src, "The export's attribution is hardcoded to the light basemap again."


def test_tracts_are_opaque_off_the_light_basemap(js_code):
    """Imagery showing through the fills would shift them off the legend's colors."""
    assert "bm==='light'?0.85:1" in js_code("_captureStudyAreaMap")
