"""The display-only Data Layers built by scripts/build_context_layers.py.

Unsheltered people (RTFH point-in-time count), electricity-dependent residents
(HHS emPOWER) and power shutoffs (SDG&E's filings with the CPUC) are context for
the map, not inputs to the index:

* each must be selectable in the Data Layers dropdown and drawn by setDataLayer();
* its values must reach the tract data the page draws from;
* none may be a scoring variable. VARIABLES feeds both the PCA and the
  "Download Data" export, and these are shown on the map only, like the UCLA
  excess-ER layer.
"""

import json
import re

import pytest


LAYERS = {"unsheltered": "unsheltered_n", "empower": "empower_rate", "psps": "psps_events"}

CONTEXT_COLUMNS = [
    "unsheltered_n",
    "empower_rate", "empower_n", "empower_zip",
    "psps_events", "psps_accounts", "psps_mbl", "psps_max_hrs",
]


@pytest.fixture(scope="module")
def tract_props(viewer_html_text):
    match = re.search(r"\bvar GJ\s*=\s*", viewer_html_text)
    gj, _ = json.JSONDecoder().raw_decode(viewer_html_text, match.end())
    return [f["properties"] for f in gj["features"]]


@pytest.mark.parametrize("layer_id", LAYERS)
def test_layer_is_in_the_dropdown(viewer_html_text, layer_id):
    assert f'<option value="{layer_id}">' in viewer_html_text, (
        f"The '{layer_id}' Data Layer is missing from the dropdown."
    )


def test_set_data_layer_draws_them(js_code):
    assert "CTX_LAYER[id]" in js_code("setDataLayer"), (
        "setDataLayer() no longer hands the context layers to their builder; "
        "picking one would fall through to the raster branch and show nothing."
    )


@pytest.mark.parametrize("layer_id,col", LAYERS.items())
def test_layer_values_reach_the_tracts(tract_props, layer_id, col):
    have = [p[col] for p in tract_props if p.get(col) is not None]
    assert have, (
        f"No tract carries `{col}`, so the '{layer_id}' layer draws empty. "
        "Run scripts/build_context_layers.py, then rebuild the viewer."
    )
    assert any(v for v in have), f"Every tract's `{col}` is zero."


def test_empty_tracts_cover_the_composite(js_code):
    """A tract with none counted gets a gray fill, not an empty one.

    The composite map is drawn underneath at full strength. Left clear, those
    tracts showed the composite's reds and yellows, which read as data.
    """
    src = js_code("_ctxColor")
    assert "return null" not in src and "CTX_NONE" in src, (
        "_ctxColor() leaves tracts with no value unfilled, so the composite "
        "underneath shows through them."
    )


@pytest.mark.parametrize("col", CONTEXT_COLUMNS)
def test_context_columns_are_not_scoring_variables(build_script_text, col):
    variables = re.search(r"^VARIABLES\s*=\s*\[(.*?)^\]", build_script_text, re.S | re.M)
    assert variables, "VARIABLES not found in the build script."
    assert f'"{col}"' not in variables.group(1), (
        f"`{col}` is in VARIABLES. That puts a display-only layer into the PCA "
        "and the Download Data export."
    )
