"""Structural checks on the generated viewer HTML.

The viewer at main/heat_risk_map.html is built by scripts/build_diy_pca_viewer.py.
These tests guard against the most common ways a build can silently regress:
unreplaced placeholder tokens, missing DOM hooks, or malformed inlined JSON.
"""

import html
import json
import re

import pytest


# DOM IDs the JS code wires up. If any disappears, large parts of the viewer
# break silently. Add new entries when introducing critical new controls.
REQUIRED_DOM_IDS = [
    "map",
    "map-wrap",
    "controls",
    "legend",
    "compute-btn",
    "layer-select",
    "layer-buttons",
    "welcome-modal",
    "wlc-got-it",
    "wlc-close-btn",
    "compare-modal",
    "export-modal",
    "error-bar",
]

# Variables spliced in by the build script — every name here must correspond to
# a `__<NAME>_PLACEHOLDER__` token replaced in build_diy_pca_viewer.py::main.
INLINED_JSON_VARS = [
    "GJ",
    "VM",
    "VSRC",
    "NO_EXPORT",
    "EXPORT_META",
    "DELTAS",
    "DELTA_MAP",
    "JURIS",
    "TRIBAL",
    "LST",
    "POLARITY",
]


def test_no_unreplaced_placeholders(viewer_html_text):
    """Every `__FOO_PLACEHOLDER__` token must be substituted at build time."""
    leftovers = re.findall(r"__[A-Z0-9_]+_PLACEHOLDER__", viewer_html_text)
    assert not leftovers, (
        f"Unreplaced placeholders in main/heat_risk_map.html: {sorted(set(leftovers))}. "
        "Rebuild with: python3 scripts/build_diy_pca_viewer.py"
    )


@pytest.mark.parametrize("dom_id", REQUIRED_DOM_IDS)
def test_required_dom_id_present(viewer_html_text, dom_id):
    assert f'id="{dom_id}"' in viewer_html_text, (
        f"Required DOM id '{dom_id}' missing from generated viewer."
    )


@pytest.mark.parametrize("var_name", INLINED_JSON_VARS)
def test_inlined_json_blob_parses(viewer_html_text, var_name):
    """Each `var X = ...;` payload must be valid JSON (or `null`).

    Catches the case where a build silently emits a Python repr or unquoted
    placeholder text where a JSON literal should be.
    """
    match = re.search(rf"\bvar {var_name}\s*=\s*", viewer_html_text)
    assert match, f"Variable declaration `var {var_name}` not found."
    start = match.end()

    # Allow JS literal `null` (which is also valid JSON, so json handles it).
    decoder = json.JSONDecoder()
    try:
        decoder.raw_decode(viewer_html_text, start)
    except json.JSONDecodeError as exc:
        snippet = viewer_html_text[start : start + 80].replace("\n", "\\n")
        pytest.fail(
            f"`var {var_name}` payload is not valid JSON: {exc}\n"
            f"  starts with: {snippet!r}"
        )


# Every `var X = __..._PLACEHOLDER__;` payload, i.e. everything spliced in at build.
SPLICED_VARS = INLINED_JSON_VARS + [
    "SRCNAME", "VUNITS", "NO_EXPORT_WHY", "UNINCORP", "COOLZONES", "POP",
    "DS_ENC", "DS_GEOIDS", "DS_DENS", "IMPERVIOUS", "CANOPY", "HEAT_MODES",
]


@pytest.mark.parametrize("var_name", SPLICED_VARS)
def test_inlined_data_cannot_close_its_script_tag(viewer_html_text, var_name):
    """A `</` inside spliced data ends the <script> early, so it must be escaped."""
    match = re.search(rf"\bvar {var_name}\s*=\s*", viewer_html_text)
    assert match, f"Variable declaration `var {var_name}` not found."
    _, end = json.JSONDecoder().raw_decode(viewer_html_text, match.end())
    assert "</" not in viewer_html_text[match.end() : end], (
        f"`var {var_name}` contains `</`. Pass it through _script_safe() in the build."
    )


def test_ocean_tract_is_not_in_the_viewer(viewer_html_text):
    """Tract 9901 is the Pacific Ocean (ALAND=0) and draws as a blob offshore."""
    assert "06073990100" not in viewer_html_text, (
        "The ocean tract 06073990100 is back in the viewer. The ALAND filter in "
        "load_data() must drop it."
    )


def test_viewer_html_is_nontrivial(viewer_html_path):
    """A viewer that's <100 KB almost certainly means the data splice failed."""
    size_kb = viewer_html_path.stat().st_size / 1024
    assert size_kb > 100, (
        f"main/heat_risk_map.html is only {size_kb:.0f} KB — expected >100 KB once "
        "tract data is spliced in. Did the build run with empty inputs?"
    )


def test_welcome_modal_shows_every_load(viewer_html_text):
    """The welcome popup is intentionally per-session: it opens on every page
    load with no localStorage gating, so the framing message lands each visit.

    (Replaces an older test that required a versioned WELCOME_KEY; that gating
    was removed when the popup switched to always-on-load.)
    """
    assert 'id="welcome-modal"' in viewer_html_text, (
        "welcome-modal markup missing from generated HTML."
    )
    assert "hrv-welcome-seen" not in viewer_html_text, (
        "Stale welcome-modal storage key found — the popup is now per-session "
        "(shows every load); don't reintroduce localStorage gating."
    )


def test_copy_citation_copies_the_citation_shown(viewer_html_text, js_fn):
    """The sidebar's Copy button must copy the citation on screen, not a second hand-kept copy."""
    assert 'id="cite-text"' in viewer_html_text
    assert 'id="cite-copy-btn"' in viewer_html_text
    assert "getElementById('cite-text')" in js_fn("copyCitation")


def test_one_citation_everywhere(viewer_html_text, js_fn):
    """The sidebar, every exported row, the data dictionary, and all three PDFs cite the same thing."""
    m = re.search(r'<p id="cite-text">(.*?)</p>', viewer_html_text, re.S)
    assert m, "sidebar #cite-text missing"
    shown = " ".join(html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).split())
    m = re.search(r"\bvar EXPORT_META\s*=\s*", viewer_html_text)
    meta, _ = json.JSONDecoder().raw_decode(viewer_html_text, m.end())
    assert meta["citation"] == shown, "Download Data rows cite something other than the sidebar"
    for fn in ("_heroPdf", "_heroPdfTwoPanel", "_tractPdfBuild"):
        assert "EXPORT_META.citation" in js_fn(fn), f"{fn}() PDF footer carries no citation"
    assert "man.citation" in js_fn("_dictionaryToCSV"), "data dictionary header carries no citation"
