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
