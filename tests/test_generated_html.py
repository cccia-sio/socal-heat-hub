"""Structural checks on the generated viewer HTML.

The viewer at main/heat_risk_map.html is built by scripts/build_diy_pca_viewer.py.
These tests guard against the most common ways a build can silently regress:
unreplaced placeholder tokens, missing DOM hooks, or malformed inlined JSON.
"""

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


def test_viewer_html_is_nontrivial(viewer_html_path):
    """A viewer that's <100 KB almost certainly means the data splice failed."""
    size_kb = viewer_html_path.stat().st_size / 1024
    assert size_kb > 100, (
        f"main/heat_risk_map.html is only {size_kb:.0f} KB — expected >100 KB once "
        "tract data is spliced in. Did the build run with empty inputs?"
    )


def test_welcome_modal_storage_key_versioned(viewer_html_text):
    """Bumping WELCOME_KEY is how returning users see updated popup copy.

    If this assertion fails the popup gating logic was rewritten — that's fine,
    just remove or update the test. But unintentional edits should be caught.
    """
    assert re.search(r"WELCOME_KEY\s*=\s*'hrv-welcome-seen-v\d+'", viewer_html_text), (
        "Expected a versioned WELCOME_KEY (e.g. 'hrv-welcome-seen-v3'). "
        "Bump the suffix whenever the popup copy changes so returning users see it."
    )
