"""Structural checks on the single-tract PDF export.

The export is driven from the tract popup's "Export tract PDF" button and is
assembled entirely in the browser, so these are static guards on the generated
viewer, not a render test. They cover the failure modes that are silent:

* The rose plot is rasterized by loading it into an <img> as a data URI, which
  re-parses it as STRICT XML. Any label radarSVG() interpolates without escaping
  (e.g. the "Age 17 & Under" variable label) makes the image fail to load and
  the export dies in onerror. So every interpolated label must go through
  _addrEsc.
* The layout must survive a tract where a pillar has more enabled variables than
  the row cap, and must say so rather than truncating silently.
* The SANDAG land-use inset fails in a way that LOOKS like success: protomaps
  hands back blank-but-valid canvas tiles when the PMTiles fetch dies, so the
  export can cheerfully ship a bare basemap captioned "Land use (SANDAG)". See
  the land-use section at the bottom.

The numbers the PDF prints are guarded separately, in test_risk_orientation.py.
"""

import re

import pytest


@pytest.fixture(scope="module")
def tract_pdf_js(viewer_html_text):
    """The SINGLE-TRACT PDF block of the generated viewer."""
    start = viewer_html_text.find("function _svgToCanvas(")
    end = viewer_html_text.find("function toggleCbMode(")
    assert start > 0 and end > start, "single-tract PDF block not found in viewer"
    return viewer_html_text[start:end]


@pytest.fixture(scope="module")
def radar_svg_js(viewer_html_text):
    start = viewer_html_text.find("function radarSVG(")
    end = viewer_html_text.find("function _hvtgStyle(")
    assert start > 0 and end > start, "radarSVG() not found in viewer"
    return viewer_html_text[start:end]


def test_export_button_wired_into_tract_popup(viewer_html_text):
    assert "popup-pdf-btn" in viewer_html_text, "the popup's PDF button is missing"
    assert "downloadTractPdf(" in viewer_html_text, "downloadTractPdf() is not wired up"
    assert ".popup-pdf-btn{" in viewer_html_text, "the PDF button has no style rule"


@pytest.mark.parametrize("fn", [
    "_svgToCanvas", "_radarCanvas", "_captureTractInset",
    "_tractPdfPillar", "_tractPdfBuild", "downloadTractPdf",
    "_hexToRgb", "_rgbTriplet", "_ord", "_pdfTrunc",
])
def test_pdf_helper_defined(viewer_html_text, fn):
    assert "function " + fn + "(" in viewer_html_text, f"{fn}() missing from viewer"


def test_radar_labels_are_xml_escaped(radar_svg_js):
    """Every label interpolated into the rose plot's SVG must pass through _addrEsc.

    Without this the SVG is valid HTML but invalid XML, and rasterizing it for
    the PDF fails silently. "Age 17 & Under" is a live example (EP_AGE17 is a
    default sensitivity spoke), so this is not hypothetical.
    """
    # <text ...>VALUE</text> and <title>VALUE</title> interpolations
    interpolated = re.findall(r"</?(?:text|title)[^>]*>'\+([A-Za-z_][\w.\[\]]*)", radar_svg_js)
    unescaped = [v for v in interpolated if not v.startswith("_addrEsc")]
    # sec.label is a static literal from _RADAR_SECTORS; everything else is data.
    unescaped = [v for v in unescaped if v != "sec.label"]
    assert not unescaped, (
        f"Rose-plot SVG interpolates unescaped values: {sorted(set(unescaped))}. "
        "Wrap them in _addrEsc() or the PDF export's rasterization will fail on "
        "any label containing '&', '<' or '>'."
    )


def test_radar_svg_size_is_patchable(viewer_html_text, radar_svg_js):
    """_radarCanvas pins the SVG's intrinsic size by string-replacing the width/
    height attrs radarSVG() emits. If that literal drifts, the replace silently
    no-ops and the rose plot rasterizes at 336px and prints blurry."""
    assert 'width="336" height="269"' in radar_svg_js, (
        "radarSVG() no longer emits width=\"336\" height=\"269\" — update the "
        "replace() in _radarCanvas() or the PDF rose plot will be upscaled."
    )
    assert 'width="336" height="269"' in viewer_html_text


def test_row_cap_is_declared_and_overflow_reported(tract_pdf_js):
    """More enabled variables than fit must be reported, never silently dropped."""
    assert "_TRACT_PDF_MAX_ROWS" in tract_pdf_js
    assert "more selected variable" in tract_pdf_js, (
        "the PDF drops variables past the row cap without saying so"
    )


def test_pdf_is_single_page(tract_pdf_js):
    """The handout is one page by construction: nothing may call addPage()."""
    assert "addPage(" not in tract_pdf_js, (
        "the tract PDF added a page break — it is specified as a one-pager"
    )


def test_satellite_imagery_is_attributed(tract_pdf_js):
    """Esri's World Imagery terms require attribution wherever it's shown."""
    assert "arcgisonline.com" in tract_pdf_js, "satellite inset no longer fetches Esri tiles"
    assert "Esri" in tract_pdf_js and "Maxar" in tract_pdf_js, (
        "the PDF embeds Esri World Imagery without crediting it"
    )


def test_pdf_carries_license_and_sources(tract_pdf_js):
    """Same provenance guarantee as the CSV/GeoJSON export."""
    assert "EXPORT_META.license" in tract_pdf_js, "the PDF states no license"
    assert "EXPORT_META.sources_url" in tract_pdf_js, "the PDF links to no source list"


def test_inset_failure_is_non_fatal(tract_pdf_js):
    """A blocked tile host must cost the map panel, not the whole export."""
    assert "if(!got.n){done(null);return;}" in tract_pdf_js, (
        "_captureTractInset no longer degrades to null when imagery is blocked; "
        "a firewalled network would lose the entire PDF instead of one panel."
    )


# ------------------------------------------------------------------------
# SANDAG land-use inset
#
# The satellite locator draws raster tiles it fetches itself, so "did any tile
# load?" is a sound success test. The land-use view can't work that way: it is a
# vector PMTiles layer rendered by protomaps into canvas tiles inside a Leaflet
# GridLayer, and when the PMTiles fetch fails (file://, a server with no Range
# support, a 404) protomaps still creates the tile canvases — they are just
# empty. A tile-count check therefore passes, and the PDF ships a plain CARTO
# basemap captioned "Land use (SANDAG)", credited to SANDAG. That is worse than
# no panel: it is a plausible-looking lie. Only the pixels can tell us.
# ------------------------------------------------------------------------

def test_landuse_helpers_defined(viewer_html_text):
    for fn in ("_pdfPaintLanduse", "_pdfLuLegend"):
        assert "function " + fn + "(" in viewer_html_text, f"{fn}() missing from viewer"


def test_landuse_inset_is_a_mode_of_the_shared_capture(js_code):
    """Both insets come from _captureTractInset, so they stay in projection lockstep."""
    src = js_code("_captureTractInset")
    assert "mode" in src.split("{")[0], (
        "_captureTractInset() no longer takes a mode — the land-use and satellite "
        "insets must share one capture path or they can drift out of alignment."
    )
    assert "basemaps.cartocdn.com" in src, (
        "the land-use inset no longer draws the CARTO basemap under the parcels"
    )


def test_landuse_panel_is_dropped_when_no_parcels_render(js_code):
    """The whole point: prove the parcels painted, don't assume it.

    Guards the exact regression this shipped with — `resolve(drew > 0)`, where
    `drew` counted tile canvases. Blank tiles count. The check must read pixels.

    Reads comment-stripped source: the code here *explains* getImageData in a
    comment, so asserting against the raw text passes even with the real check
    ripped out.
    """
    src = js_code("_pdfPaintLanduse")

    assert "getImageData" in src, (
        "_pdfPaintLanduse() no longer inspects pixels to decide whether the parcels "
        "actually rendered. Counting tiles is NOT enough: protomaps produces "
        "blank-but-valid canvas tiles when the PMTiles fetch fails, so a tile count "
        "passes and the PDF ships a bare basemap captioned 'Land use (SANDAG)'."
    )
    assert re.search(r"resolve\(\s*painted\s*\)", src), (
        "_pdfPaintLanduse() must resolve on whether pixels were painted, not on how "
        "many tiles it drew."
    )
    assert not re.search(r"resolve\(\s*drew\s*>", src), (
        "_pdfPaintLanduse() is back to resolving on a tile COUNT (`drew > 0`). Empty "
        "tiles count, so this reports success for a blank layer."
    )


def test_landuse_pixel_check_uses_a_scratch_canvas(js_code):
    """The parcels must be tested BEFORE they're composited over the basemap.

    Once the tiles are drawn onto the CARTO base, their pixels are
    indistinguishable from it and the emptiness check becomes impossible.
    """
    src = js_code("_pdfPaintLanduse")
    scratch = re.search(r"(\w+)\s*=\s*document\.createElement\('canvas'\)", src)
    assert scratch, (
        "_pdfPaintLanduse() no longer composites to its own scratch canvas; it cannot "
        "tell parcel pixels from basemap pixels once they're merged."
    )


def test_landuse_caption_and_credit_are_gated_on_the_capture(js_code):
    """A dropped panel must take its caption AND its attribution with it.

    Crediting SANDAG for a panel that shows no SANDAG data is a licensing claim
    about content that isn't there.

    EVERY line mentioning SANDAG has to be gated, not just the first one found —
    the caption and the footer credit are separate lines, and checking only the
    first lets an ungated credit through while the caption still looks fine.
    """
    src = js_code("_tractPdfBuild")
    mentions = [ln for ln in src.splitlines() if "SANDAG" in ln]
    assert mentions, "_tractPdfBuild() no longer mentions SANDAG at all."

    ungated = [ln.strip() for ln in mentions if "luC" not in ln]
    assert not ungated, (
        f"SANDAG is referenced without checking that the land-use capture succeeded: "
        f"{ungated}. Every mention — the panel caption AND the source credit — must be "
        "gated on luC, or a PDF whose land-use capture failed still carries a caption "
        "and a SANDAG credit for a panel that isn't there."
    )


def test_landuse_legend_reads_the_shared_category_table(js_code):
    """One source of truth for the parcel colours: the map and PDF must not diverge."""
    src = js_code("_pdfLuLegend")
    assert "LU_CATS" in src and "LU_WATER_CAT" in src, (
        "_pdfLuLegend() no longer builds itself from LU_CATS/LU_WATER_CAT, so the PDF "
        "legend can drift from the colours the parcels are actually drawn in."
    )


def test_landuse_legend_parses_hex_not_rgb(js_code):
    """LU_CATS colours are hex ('#f4a460'). _rgbTriplet only parses 'rgb(r,g,b)'.

    Feeding hex to _rgbTriplet doesn't throw — it silently returns the fallback
    grey, so every swatch in the legend comes out the same colour and the legend
    becomes useless while looking fine. Hit this exact bug once already.
    """
    assert "_hexToRgb" in js_code("_pdfLuLegend"), (
        "_pdfLuLegend() must convert LU_CATS' hex colours with _hexToRgb()."
    )
    assert "_rgbTriplet" not in js_code("_pdfLuLegend"), (
        "_pdfLuLegend() uses _rgbTriplet on LU_CATS' HEX colours. _rgbTriplet only "
        "matches 'rgb(r,g,b)' and falls back to grey on anything else, so every legend "
        "swatch would silently render the same colour."
    )


def test_landuse_view_is_not_tinted_by_the_tract_highlight(js_code):
    """The satellite inset washes the tract in translucent yellow. The land-use one
    must not: a wash shifts every parcel away from the colour its legend swatch
    promises, which is exactly the reading the panel exists to support."""
    src = js_code("_captureTractInset")
    fill = next((ln for ln in src.splitlines() if "fill('evenodd')" in ln), "")
    assert fill, "_captureTractInset() no longer fills the tract outline."
    assert "mode!=='lu'" in fill.replace(" ", ""), (
        f"the yellow tract wash is applied in land-use mode too: {fill.strip()!r}. "
        "It tints every parcel off its legend colour."
    )
