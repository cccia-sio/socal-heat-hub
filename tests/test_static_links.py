"""Verify relative links from the landing page and README still resolve.

Catches the classic "renamed an asset, forgot to update the link" regression.
External URLs (http/https/mailto/data/#anchor) are not checked.
"""

import re
from pathlib import Path

import pytest


_HREF_OR_SRC = re.compile(r'(?:href|src)\s*=\s*"([^"]+)"', re.IGNORECASE)
_MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


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
