"""Guard the GitHub Pages deploy allow-list.

The workflow stages the public site by copying an explicit list of files into
`_site/`. That is an allow-list *on purpose*. The previous version was a
deny-list — it excluded `scripts/`, `.github/`, and most of `data/`, then
shipped everything else — which silently published ARCHITECTURE.md, CLAUDE.md,
the test suite, internal PCA methodology docs, and a partner's private review
PDF to the open internet. The repo being private did nothing to stop it, because
Pages serves the *deployed* artifact, not the repo.

The invariant is: **adding a file to the repo must never publish it.** It ships
only if it is named in pages.yml.

These tests enforce both directions:
  - nothing internal is staged (leak guard), and
  - everything the published pages reference IS staged (breakage guard).
"""

import re

import pytest


# Tracked files that are fine in a private repo but must never reach the public
# site. Each was verifiably live at cccia-sio.github.io before the allow-list.
MUST_NOT_DEPLOY = [
    "ARCHITECTURE.md",
    "CLAUDE.md",
    "README.md",
    "pytest.ini",
    "tests/",
    "docs/",
    "_mockup_exposure_abs.png",
    "UserGuide_SDRCC",              # internal partner review comments
    "main/hospitalizations.html",   # unlinked page; not part of the public site
    "scripts/",
]

# Assets the three linked pages fetch at runtime. If one of these stops being
# staged the site deploys broken (blank basemap / missing overlay), so the
# allow-list has to be checked in this direction too. Named file by file rather
# than by directory: the staging step enumerates them individually, so a
# directory prefix would keep passing after someone dropped one from the list.
MUST_DEPLOY = [
    "index.html",
    "main/heat_risk_map.html",
    "userguide/index.html",
    "data/context/sandag_landuse.pmtiles",
    "data/context/overlays/population_4326.png",
    "data/context/overlays/population_4326_cb.png",
    "data/context/overlays/impervious_4326.png",
    "data/context/overlays/impervious_4326_cb.png",
    "data/context/overlays/canopy_4326.png",
    "data/context/overlays/canopy_4326_cb.png",
    "data/exposure/processed/landsat/annual_lst_4326.png",
    "data/exposure/processed/landsat/annual_lst_4326_cb.png",
]

_STAGED_DATA = re.compile(r"(?<!_site/)\bdata/\S+")
_VIEWER_DATA_REF = re.compile(r"\.\./(data/[^\"']+)")


@pytest.fixture(scope="session")
def staging_block(pages_workflow_text: str) -> str:
    """The `Stage site` step body — the only place that decides what ships.

    Shell comments are stripped first. The step documents which files are
    deliberately left unstaged, so naming a path in a comment must neither
    satisfy "this asset IS staged" nor trip "this path must NEVER ship" — it
    would do both, in opposite directions, on the same text. Same reasoning as
    strip_js_comments() in conftest: the code that fixes a bug usually also
    describes it.
    """
    match = re.search(
        r"- name: Stage site.*?(?=\n      - )", pages_workflow_text, re.DOTALL
    )
    assert match, "No `Stage site` step found in pages.yml."
    return "\n".join(
        line for line in match.group(0).splitlines()
        if not line.lstrip().startswith("#")
    )


@pytest.mark.parametrize("path", MUST_NOT_DEPLOY)
def test_internal_paths_never_staged(staging_block: str, path: str):
    assert path not in staging_block, (
        f"`{path}` is staged for deploy in pages.yml — it would be served "
        f"publicly at cccia-sio.github.io. The deploy is an allow-list; only "
        "the linked pages and the assets they fetch belong in it."
    )


@pytest.mark.parametrize("path", MUST_DEPLOY)
def test_required_assets_staged(staging_block: str, path: str):
    assert path in staging_block, (
        f"`{path}` is no longer staged in pages.yml. The viewer fetches it at "
        "runtime, so the site would deploy broken."
    )


def test_staged_data_matches_what_the_viewer_fetches(
    staging_block: str, viewer_html_text: str
):
    """The staged `data/` files must be exactly the ones the viewer asks for.

    Both sides are literals — every asset URL in the built viewer is a hardcoded
    string, nothing is concatenated at runtime — so the correct list is knowable
    and this can be checked in both directions at once:

      * staged but never fetched — dead weight, and the symptom of a recursive
        copy. The bounds JSON and `downscaled_*` rasters live in these same
        directories but are read at BUILD time and inlined into the HTML; a
        `cp -R` shipped all 1.1 MB of them a second time.
      * fetched but not staged — the site deploys visibly broken.

    Checking membership by filename can't catch the first case, because under
    `cp -R` the files that ship are never named anywhere. Comparing sets can.
    """
    staged = set(_STAGED_DATA.findall(staging_block))
    fetched = set(_VIEWER_DATA_REF.findall(viewer_html_text))
    assert staged == fetched, (
        "pages.yml and the built viewer disagree about which data files ship.\n"
        f"  staged, never fetched: {sorted(staged - fetched) or 'none'}\n"
        f"  fetched, never staged: {sorted(fetched - staged) or 'none'}\n"
        "If the first list is non-empty and the paths are whole directories, the "
        "staging step went back to `cp -R`."
    )


def test_data_dirs_are_not_copied_recursively(staging_block: str):
    """`cp -R` on a data directory re-creates the deny-list default.

    The allow-list's promise is that adding a file to the repo never publishes
    it. A recursive copy silently breaks that promise for everything under the
    directory it targets, which is how the build-time-only assets above started
    shipping in the first place.
    """
    recursive = re.findall(r"cp\s+-[a-zA-Z]*[rR][a-zA-Z]*\s+\S*data/\S*", staging_block)
    assert not recursive, (
        f"Recursive copy of a data directory in pages.yml: {recursive}. "
        "Name each file the viewer fetches instead — every asset path in the "
        "built viewer is a literal, so the list is knowable."
    )


def test_no_blanket_copy(staging_block: str):
    """A recursive copy of the repo root re-creates the original leak."""
    blanket = re.search(r"(rsync[^\n]*|cp\s+-[a-zA-Z]*r[a-zA-Z]*)\s+\./?\s+_site", staging_block)
    assert not blanket, (
        f"pages.yml copies the repo root into _site ({blanket.group(0)!r}). "
        "That is the deny-list pattern this allow-list replaced — it publishes "
        "every tracked file, including ARCHITECTURE.md, CLAUDE.md, and tests/."
    )


def test_staging_fails_loudly(staging_block: str):
    """`set -e` means a renamed asset breaks the build instead of vanishing."""
    assert "set -euo pipefail" in staging_block, (
        "The staging step should `set -euo pipefail` so a missing or renamed "
        "source file fails the deploy rather than silently dropping from the site."
    )


def test_workflow_triggers_on_main_push(pages_workflow_text: str):
    assert re.search(r"branches:\s*\[main\]", pages_workflow_text), (
        "pages.yml should still trigger on push to main."
    )
