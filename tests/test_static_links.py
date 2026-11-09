"""Verify relative links from the landing page and README still resolve.

Catches the classic "renamed an asset, forgot to update the link" regression.
External URLs (http/https/mailto/data/#anchor) are not checked.
"""

import re
from pathlib import Path

import pytest


_HREF_OR_SRC = re.compile(r'(?:href|src)\s*=\s*"([^"]+)"', re.IGNORECASE)
_MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# A bare `#` — not `#section-id`, which is a real in-page anchor.
_DEAD_ANCHOR = re.compile(r'<a\b[^>]*\bhref="#"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)

# Every page the deploy allow-list publishes as a browsable HTML page.
_DEPLOYED_PAGES = ("index.html", "userguide/index.html")


def _local_targets(text: str, patterns) -> list[str]:
    targets = []
    for pat in patterns:
        for raw in pat.findall(text):
            url = raw.strip()
            if not url:
                continue
            if url.startswith(("http://", "https://", "mailto:", "data:", "#")):
                continue
            # Strip fragment / query for filesystem check.
            url = url.split("#", 1)[0].split("?", 1)[0]
            if url:
                targets.append(url)
    return targets


def _index_links(project_root: Path) -> list[tuple[Path, str]]:
    text = (project_root / "index.html").read_text(encoding="utf-8")
    return [
        (project_root / "index.html", t) for t in _local_targets(text, [_HREF_OR_SRC])
    ]


def _readme_links(project_root: Path) -> list[tuple[Path, str]]:
    text = (project_root / "README.md").read_text(encoding="utf-8")
    return [(project_root / "README.md", t) for t in _local_targets(text, [_MD_LINK])]


@pytest.mark.parametrize(
    "source_file,target",
    [pytest.param(s, t, id=f"{s.name}:{t}") for s, t in _index_links(Path(__file__).resolve().parent.parent)],
)
def test_index_local_link_resolves(source_file, target):
    resolved = (source_file.parent / target).resolve()
    assert resolved.exists(), (
        f"{source_file.name} links to '{target}' but {resolved} does not exist."
    )


@pytest.mark.parametrize("page", _DEPLOYED_PAGES)
def test_no_placeholder_anchors(project_root, page):
    """`href="#"` is a link to nowhere dressed up as a link to something.

    The landing page shipped a "Technical Guide" hero button pointing at `#`,
    sitting between "Launch the Tool" and "User Guide" on the public site. The
    resolver above cannot see it: it skips every target starting with `#` so
    that real fragment links (`#data`, `#feedback`) aren't checked against the
    filesystem, and a bare `#` rides along in that exemption.

    A placeholder for a page that doesn't exist yet belongs in an issue, not in
    the nav.
    """
    text = (project_root / page).read_text(encoding="utf-8")
    dead = [re.sub(r"\s+", " ", m).strip() for m in _DEAD_ANCHOR.findall(text)]
    assert not dead, (
        f"{page} has anchor(s) pointing at a bare '#': {dead}. "
        "Point them at a real target or remove them — on a deployed page this "
        "is a nav item that silently does nothing."
    )


@pytest.mark.parametrize("page", _DEPLOYED_PAGES + ("main/heat_risk_map.html",))
def test_license_links_open_on_github(project_root, page):
    """Pages serves LICENSE as application/octet-stream, so a relative link downloads it."""
    text = (project_root / page).read_text(encoding="utf-8")
    hrefs = re.findall(r'href="([^"]*(?:LICENSE|DATA_LICENSE\.md)[^"]*)"', text)
    raw = [h for h in hrefs if not h.startswith("https://github.com/cccia-sio/socal-heat-hub/blob/main/")]
    assert not raw, f"{page} links to the raw license file(s) {raw}; link to the file on GitHub."


def test_readme_local_links_resolve(project_root):
    """README is mostly external links; iterate inline so an all-external
    README isn't reported as a misleading "empty parametrize" skip.
    """
    links = _readme_links(project_root)
    if not links:
        pytest.skip("README has no local links to verify.")
    broken = [
        (source.name, target)
        for source, target in links
        if not (source.parent / target).resolve().exists()
    ]
    assert not broken, f"Broken README local links: {broken}"
