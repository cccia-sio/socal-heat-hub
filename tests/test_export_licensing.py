"""Licensing guarantees for the viewer's "Download Data" export.

Two invariants, both of which are easy to break by accident when adding a
variable:

1. Variables from a source whose terms bar unrestricted redistribution — today
   Healthy Places Index (non-commercial) — must never reach the export: not as
   raw values, not as `_sd_pctile` ranks, not as `_adj` climate-adjusted
   variants. They stay on the map, used under the source's own terms. Data we
   have no right to publish at all (the emailed heat-hospitalisation
   estimates) must not be in the build.
2. Every exported file must carry its own attribution, so a CSV that travels
   away from the site still states its license and links to the source table.

See DATA_LICENSE.md for the reasoning behind which sources are restricted.
"""

import ast
import json
import re

import pytest


def _literals_from_build_script(build_script_text: str) -> dict:
    """Eval the build script's module-level constants without importing it.

    Importing would pull in geopandas/sklearn; we only need the tables.
    """
    tree = ast.parse(build_script_text)
    src_consts: dict = {}
    wanted = ("VARIABLES", "VAR_SOURCES", "RESTRICTED_EXPORT_SOURCES",
              "RESTRICTED_REASONS", "EXPORT_META")
    out: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], "id", None)
        if name is None:
            continue
        if name.startswith("_SRC_"):
            src_consts[name] = ast.literal_eval(node.value)
        elif name in wanted:
            out[name] = eval(  # noqa: S307 - our own source, _SRC_* names only
                compile(ast.Expression(node.value), "<build>", "eval"), {}, src_consts
            )
    missing = [w for w in wanted if w not in out]
    assert not missing, f"Constants missing from build script: {missing}"
    return out


def _json_var(html: str, name: str):
    match = re.search(rf"\bvar {name}\s*=\s*", html)
    assert match, f"`var {name}` not found in generated viewer."
    return json.JSONDecoder().raw_decode(html, match.end())[0]


@pytest.fixture(scope="module")
def consts(build_script_text):
    return _literals_from_build_script(build_script_text)


@pytest.fixture(scope="module")
def restricted_keys(consts):
    """Variable keys whose source bars unrestricted redistribution."""
    keys = [
        key
        for key, _label, _pillar, _sub in consts["VARIABLES"]
        if consts["VAR_SOURCES"].get(key) in consts["RESTRICTED_EXPORT_SOURCES"]
    ]
    assert keys, "No restricted variables found — RESTRICTED_EXPORT_SOURCES wired up wrong?"
    return keys


def test_known_gated_sources_stay_restricted(consts):
    """Healthy Places Index (non-commercial, audited 2026-07-27) stays out of
    the download; its terms conflict with CC BY 4.0."""
    url = "https://www.healthyplacesindex.org/"
    assert url in consts["RESTRICTED_EXPORT_SOURCES"], (
        f"{url} was audited as not redistributable but is no longer in "
        "RESTRICTED_EXPORT_SOURCES. Its variables would ship in the download."
    )


def test_emailed_hospitalization_estimates_never_ship(consts, build_script_text,
                                                      viewer_html_text):
    """The per-ZCTA heat-hospitalisation estimates must not be built in at all.

    They arrived as an emailed attachment with no licence and no public
    download. Withholding them from the export was not enough, because the map
    itself published the tract values. Removed from the viewer 2026-11-04.
    """
    keys = [k for k, *_ in consts["VARIABLES"] if k.startswith("heat_hosp")]
    assert not keys, f"Hospitalization variables are back in VARIABLES: {keys}"
    assert "data/hospitalizations" not in build_script_text, (
        "build_diy_pca_viewer.py reads data/hospitalizations/ again."
    )
    assert "heat_hosp" not in viewer_html_text, (
        "The built viewer contains heat_hosp_* fields."
    )


def test_restricted_sources_explain_themselves(consts):
    """Every restricted source needs a reason shown in the data dictionary."""
    missing = sorted(
        consts["RESTRICTED_EXPORT_SOURCES"] - set(consts["RESTRICTED_REASONS"])
    )
    assert not missing, (
        f"Restricted sources with no RESTRICTED_REASONS entry: {missing}. The "
        "data dictionary would fall back to a generic note instead of telling "
        "users where to obtain the data."
    )


def test_no_export_reasons_reach_the_page(viewer_html_text):
    """NO_EXPORT_WHY must cover every key in NO_EXPORT."""
    no_export = set(_json_var(viewer_html_text, "NO_EXPORT"))
    why = _json_var(viewer_html_text, "NO_EXPORT_WHY")
    uncovered = sorted(no_export - set(why))
    assert not uncovered, f"Withheld variables with no explanation: {uncovered}"


def test_every_variable_has_a_source(consts):
    """A variable with no VAR_SOURCES entry can't be cited or license-checked."""
    orphans = [
        key
        for key, _label, _pillar, _sub in consts["VARIABLES"]
        if key not in consts["VAR_SOURCES"]
    ]
    assert not orphans, (
        f"Variables with no source mapping: {orphans}. Add them to VAR_SOURCES so they "
        "can be attributed (and screened for restrictive terms)."
    )


def test_restricted_vars_are_all_in_no_export(viewer_html_text, restricted_keys):
    """Every restricted variable must land in the built page's NO_EXPORT list."""
    no_export = set(_json_var(viewer_html_text, "NO_EXPORT"))
    leaked = sorted(set(restricted_keys) - no_export)
    assert not leaked, (
        f"Restricted variables absent from NO_EXPORT: {leaked}. These would be included "
        "in the Download Data export, violating their source's terms."
    )


def test_export_column_spec_skips_restricted_in_every_column_family(viewer_html_text):
    """The raw, `_sd_pctile` and `_adj` column loops must each honour the skip set.

    A new column family added without a `skip[...]` guard would quietly
    re-introduce restricted data into the download.
    """
    start = viewer_html_text.index("function _exportColumnSpec()")
    end = viewer_html_text.index("function _rowValueFor(")
    body = viewer_html_text[start:end]

    assert "NO_EXPORT" in body, "_exportColumnSpec() no longer consults NO_EXPORT."

    pushes = re.findall(r"cols\.push\(\{key:v\.key[^}]*\}\)", body)
    assert len(pushes) >= 3, (
        f"Expected >=3 per-variable column families (raw, _sd_pctile, _adj); found "
        f"{len(pushes)}. Did a family get added or removed?"
    )
    unguarded = [p for p in pushes if "skip[v.key]" not in _enclosing_guard(body, p)]
    assert not unguarded, (
        f"Per-variable column push(es) not guarded by the restricted-source skip set: "
        f"{unguarded}. Restricted variables would leak into the export."
    )


def _enclosing_guard(body: str, push: str) -> str:
    """Return the source line containing `push` (guards are written inline)."""
    for line in body.splitlines():
        if push in line:
            return line
    return ""


def test_export_meta_present_and_complete(viewer_html_text, consts):
    """Every exported row carries citation / license / sources_url."""
    meta = _json_var(viewer_html_text, "EXPORT_META")
    assert meta == consts["EXPORT_META"], (
        "EXPORT_META in the built page differs from the build script — stale HTML? "
        "Rebuild with: python3 scripts/build_diy_pca_viewer.py"
    )
    for field in ("citation", "license", "sources_url"):
        assert meta.get(field), f"EXPORT_META missing/empty field: {field}"
    assert meta["license"] == "CC BY 4.0"
    assert meta["sources_url"].startswith("http")


def test_export_meta_is_ascii(consts):
    """The CSV ships without a BOM; non-ASCII would mojibake in Excel."""
    for field, value in consts["EXPORT_META"].items():
        assert value.isascii(), (
            f"EXPORT_META[{field!r}] contains non-ASCII characters: {value!r}. "
            "Use plain hyphens/quotes so the CSV opens cleanly in Excel."
        )


def test_export_meta_columns_are_emitted(viewer_html_text):
    """_exportColumnSpec() must actually append the attribution columns."""
    start = viewer_html_text.index("function _exportColumnSpec()")
    end = viewer_html_text.index("function _rowValueFor(")
    body = viewer_html_text[start:end]
    assert "EXPORT_META" in body, (
        "_exportColumnSpec() no longer emits the attribution columns — downloads "
        "would ship with no citation or license."
    )


def test_row_value_resolves_meta_columns(viewer_html_text):
    """_rowValueFor() must resolve the attribution columns to their values."""
    start = viewer_html_text.index("function _rowValueFor(")
    end = viewer_html_text.index("function _csvEscape(")
    body = viewer_html_text[start:end]
    assert "col.meta" in body and "EXPORT_META" in body, (
        "_rowValueFor() doesn't resolve meta columns — citation/license/sources_url "
        "would export as empty cells."
    )
