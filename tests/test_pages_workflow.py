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
# allow-list has to be checked in this direction too.
MUST_DEPLOY = [
    "index.html",
    "main/heat_risk_map.html",
    "userguide/index.html",
    "data/context/sandag_landuse.pmtiles",
    "data/context/overlays",
    "data/exposure/processed/landsat",
]


@pytest.fixture(scope="session")
def staging_block(pages_workflow_text: str) -> str:
    """The `Stage site` step body — the only place that decides what ships."""
    match = re.search(
        r"- name: Stage site.*?(?=\n      - )", pages_workflow_text, re.DOTALL
    )
    assert match, "No `Stage site` step found in pages.yml."
    return match.group(0)


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
