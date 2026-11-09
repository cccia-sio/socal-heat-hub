"""Every tract surface must speak the same language: RISK percentiles.

The popup rose plot is the reference. It plots protective variables
(POLARITY[key] === false — tree canopy, income, AC prevalence) as a *deficit*,
100 - pct, so that "further from the centre" always means "more risk". The
`_adaptive` pillar score is stored capacity-side (higher = MORE capacity), so it
is likewise flipped to 100 - _adaptive; compute() already does this when it
averages the composite.

Three separate surfaces render those numbers, each with its own copy of the
percentile logic:

    1. the popup rose plot   — radarSVG() / _radarPct()
    2. the "Show more info"  — openDetailDrawer() / _dtRow()
    3. the tract PDF export  — _tractPdfBuild() / _tractPdfPillar()

They have drifted apart twice. The detail modal spent a long time printing the
RAW percentile, so a tract with the county's best tree canopy read "100th
percentile" in the drawer and "0" on the rose plot — the same number, inverted,
under the same label. The PDF export then shipped with the identical bug, and
because it embeds the rose plot on the same page it printed "ADAPTIVE CAP. 100"
in red directly above a rose plot reading "Adaptive Capacity 0".

The subtle part, and the reason a "does it mention POLARITY?" check is not
enough: the old code got the *colour* right and the *number* wrong. It computed
a risk percentile, used it for the colour ramp, and then drew the bar and
printed the rank from the raw percentile. So these tests pin the bar length and
the printed rank specifically, not just the presence of a flip.
"""

import re

import pytest


def _line_containing(src: str, needle: str) -> str:
    """First non-comment line containing `needle`.

    Comments are skipped deliberately: the code that fixes these bugs also
    *describes* them, so a naive search matches the explanatory comment above
    the card rather than the card.
    """
    for line in src.splitlines():
        if line.lstrip().startswith("//"):
            continue
        if needle in line:
            return line
    return ""


# ---------------------------------------------------------------- rose plot
# The reference implementation. If this flips, everything else is wrong.

def test_radar_flips_protective_variables(js_code):
    """_radarPct plots protective variables as a deficit."""
    src = js_code("_radarPct")
    assert "inv?100-b:b" in src.replace(" ", ""), (
        "_radarPct() no longer plots protective variables as a deficit (100-pct). "
        "The rose plot's whole premise is that outward = more risk."
    )


def test_radar_polarity_comes_from_the_polarity_table(js_code):
    src = js_code("_radarInv")
    assert "POLARITY[key]===false" in src.replace(" ", ""), (
        "_radarInv() must decide 'is this protective?' from POLARITY, not a hand-kept list."
    )


def test_radar_adaptive_wedge_is_inverted(viewer_html_text):
    """The adaptive wedge plots the capacity GAP, matching the composite."""
    sectors = viewer_html_text[
        viewer_html_text.index("var _RADAR_SECTORS") : viewer_html_text.index("function _radarPct")
    ]
    adaptive = _line_containing(sectors, "pillar:'adaptive'")
    assert "invScore:true" in adaptive.replace(" ", ""), (
        "The rose plot's adaptive wedge is no longer inverted. _adaptive is stored "
        "capacity-side, so without invScore the best-capacity tract plots as highest risk."
    )


# ------------------------------------------------------------ detail drawer

def test_drawer_row_bar_and_rank_both_use_the_risk_percentile(js_code):
    """Bar length AND printed rank must derive from riskPct, not the raw percentile.

    This is the exact shape of the original bug: colour was risk-oriented while
    the bar width and the printed number were raw, so a high-canopy tract drew a
    long green bar labelled "100th percentile".
    """
    src = js_code("_dtRow")

    assert re.search(r"var riskPct\s*=\s*higherIsWorse\s*\?\s*p\s*:\s*\(100 - p\)", src), (
        "_dtRow() no longer derives a risk percentile from polarity."
    )

    width = re.search(r"var width\s*=\s*([^;]+);", src)
    assert width, "_dtRow() no longer computes a bar width."
    assert "riskPct" in width.group(1), (
        f"_dtRow() draws the bar from the RAW percentile: {width.group(1).strip()!r}. "
        "It must use riskPct, or a protective variable renders a long bar for LOW risk "
        "while its colour says low risk — the bar and the colour would contradict."
    )

    printed = re.search(r"""dt-var-pct">'\+([^+]+)\+'""", src)
    assert printed, "_dtRow() no longer prints a percentile."
    assert "riskPct" in printed.group(1), (
        f"_dtRow() prints the RAW percentile: {printed.group(1).strip()!r}. It must print "
        "riskPct so the drawer reconciles with the rose plot (canopy 0, not 100)."
    )


def test_drawer_polarity_note_sits_with_the_value_not_the_percentile(js_code):
    """"86th percentile risk · higher value = less risk" read as "86 is good".

    A City of San Diego user read a low Neighborhood Quality score (17.4, 86th
    percentile risk) as the 86th percentile of quality, because the polarity
    note sat beside the risk rank. The note describes the raw value, so it must
    live in the value span and the percentile line must carry only the rank.
    """
    src = js_code("_dtRow")

    pct_line = _line_containing(src, 'dt-var-pct">')
    assert "dirNote" not in pct_line, (
        f"The percentile line carries the polarity note again: {pct_line.strip()!r}. "
        "Put it beside the value (dt-var-dir)."
    )
    val_line = _line_containing(src, "dt-var-dir")
    assert "dirNote" in val_line, "_dtRow() no longer prints the polarity note beside the value."


def test_drawer_polarity_comes_from_the_polarity_table(js_code):
    src = js_code("openDetailDrawer")
    assert "POLARITY[key]!==false" in src.replace(" ", ""), (
        "openDetailDrawer() must read polarity from POLARITY when building its rows."
    )


# ------------------------------------------------------------- PDF export

def test_pdf_row_bar_and_rank_both_use_the_risk_percentile(js_code):
    """Same invariant in the PDF, which embeds the rose plot on the same page."""
    src = js_code("_tractPdfBuild")

    assert re.search(r"var risk\s*=\s*r\.worse\s*\?\s*p0\s*:\s*\(100-p0\)", src.replace(" ", "")) or \
        "varrisk=r.worse?p0:(100-p0)" in src.replace(" ", ""), (
        "_tractPdfBuild() no longer derives a risk percentile from polarity."
    )

    assert "barW*risk/100" in src.replace(" ", ""), (
        "The PDF draws its row bars from the RAW percentile (barW*p0/100). It must use "
        "the risk percentile — the rose plot is printed on the same page and would disagree."
    )
    assert "barW*p0/100" not in src.replace(" ", ""), (
        "The PDF still sizes a row bar from the raw percentile p0."
    )

    assert "_ord(risk)" in src.replace(" ", ""), (
        "The PDF prints the RAW percentile rank. It must print the risk rank so a "
        "high-canopy tract reads '0th pct risk', matching its rose plot spoke."
    )
    assert "_ord(p0)" not in src.replace(" ", ""), (
        "The PDF still prints a raw percentile rank via _ord(p0)."
    )


def test_pdf_polarity_comes_from_the_polarity_table(js_code):
    src = js_code("_tractPdfPillar")
    assert "POLARITY[key]!==false" in src.replace(" ", ""), (
        "_tractPdfPillar() must read polarity from POLARITY."
    )


# ------------------------------------------- adaptive pillar score, both cards

@pytest.mark.parametrize("fn_name", ["openDetailDrawer", "_tractPdfBuild"])
def test_adaptive_summary_card_shows_the_capacity_gap(js_code, fn_name):
    """`_adaptive` is capacity-side; the summary card must flip it.

    compute() builds the composite as (exposure + sensitivity + (100 - adaptive)) / 3,
    which is what makes _adaptive capacity-oriented. A card that prints it raw shows
    100 for the county's BEST-capacity tract and colours it as maximum risk.
    """
    card = _line_containing(js_code(fn_name), "'Adaptive Cap.'")
    assert card, f"{fn_name}() no longer renders an 'Adaptive Cap.' summary card."
    assert "100-pr._adaptive" in card.replace(" ", ""), (
        f"{fn_name}() prints _adaptive raw on its summary card: {card.strip()!r}. "
        "It must show 100 - _adaptive (the capacity gap) to match the rose plot's "
        "adaptive wedge and the composite."
    )


def test_compare_modal_pillar_scores_show_the_capacity_gap(js_code):
    """The compare modal printed _adaptive raw: the best-capacity tract read 100 in red."""
    src = js_code("openCompareModal").replace(" ", "")
    assert "aV=100-aV" in src and "bV=100-bV" in src, (
        "openCompareModal() prints the Adaptive Capacity pillar score raw. It must show "
        "100 - _adaptive, like the rose plot, the drawer and the tract PDF."
    )


def test_compute_inverts_adaptive_into_the_composite(js_code):
    """The premise the two cards above rely on: adaptive enters the composite inverted."""
    compute = js_code("compute")
    assert re.search(r'if\s*\(\s*pid\s*===\s*"adaptive"\s*\)\s*s\s*=\s*100\s*-\s*s', compute), (
        "compute() no longer inverts the adaptive pillar when averaging the composite. "
        "If adaptive is now stored risk-side instead, every 100-_adaptive flip in the "
        "drawer, the PDF and the rose plot is wrong and must be removed together."
    )


# ------------------------------------------------- card accents = pillar hues

@pytest.mark.parametrize(
    "fn_name,risk_ramp_call",
    [("openDetailDrawer", "_dtRiskColor(c.val)"), ("_tractPdfBuild", "_dtRiskColor(v)")],
)
def test_summary_cards_are_coloured_by_pillar_not_by_score(js_code, fn_name, risk_ramp_call):
    """Card accents come from PMETA, so each card matches its own pillar header.

    Colouring them with the green-to-red risk ramp made the accent encode the score
    instead of the category — "Adaptive Cap." got a red bar sitting directly above an
    ADAPTIVE CAPACITY section header in green.
    """
    src = js_code(fn_name)
    assert risk_ramp_call.replace(" ", "") not in src.replace(" ", ""), (
        f"{fn_name}() colours a summary card with the risk ramp ({risk_ramp_call}). "
        "Card accents must come from PMETA so they match their pillar."
    )
    for pillar in ("exposure", "sensitivity", "adaptive"):
        assert f"PMETA.{pillar}.color" in src, (
            f"{fn_name}()'s summary cards don't take the {pillar} accent from PMETA."
        )


def test_variable_row_bars_still_use_the_risk_ramp(js_code):
    """The row bars SHOULD stay green-to-red — they encode magnitude, not category.

    Guards the over-correction: someone reading the card fix above might strip
    _dtRiskColor everywhere, which would flatten the per-variable bars to one hue and
    throw away the low/high risk signal.
    """
    assert "_dtRiskColor(riskPct)" in js_code("_dtRow").replace(" ", ""), (
        "_dtRow()'s bar no longer uses the risk colour ramp."
    )
    assert "_dtRiskColor(risk)" in js_code("_tractPdfBuild").replace(" ", ""), (
        "The PDF's row bars no longer use the risk colour ramp."
    )
