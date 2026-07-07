"""Enforce the .gitignore allow-list.

CLAUDE.md: only specific build scripts and the landsat data subtree should be
tracked. A stray CSV or notebook tracked under `data/` or `scripts/` won't
deploy (workflow excludes them) but will pollute the repo and confuse
contributors. This test fails fast if the allow-list drifts.
"""

import subprocess
from pathlib import Path

import pytest


ALLOWED_SCRIPT_FILES = {
    "build_diy_pca_viewer.py",
    "build_lst_overlay.py",
    "build_data_overlays.py",
    "build_userguide_pdf.sh",
    "build_ucla_heat_tract_xwalk.py",
}

# Tracked data files. The landsat and context/overlays directories are wholly
# tracked; other entries are explicit allow-listed singletons (see .gitignore).
ALLOWED_DATA_PREFIXES = (
    "data/exposure/processed/landsat/",
    "data/context/overlays/",
)
ALLOWED_DATA_FILES = {
    "data/context/sandag_landuse.pmtiles",
}


def _ls_files(project_root: Path, prefix: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_only_allowed_scripts_tracked(project_root):
    tracked = _ls_files(project_root, "scripts/")
    unexpected = [
        path for path in tracked if Path(path).name not in ALLOWED_SCRIPT_FILES
    ]
    assert not unexpected, (
        f"Unexpected tracked files under scripts/: {unexpected}. "
        f"Allow-list (see .gitignore): {sorted(ALLOWED_SCRIPT_FILES)}"
    )


def test_only_allowed_data_tracked(project_root):
    tracked = _ls_files(project_root, "data/")
    unexpected = [
        path
        for path in tracked
        if path not in ALLOWED_DATA_FILES
        and not any(path.startswith(p) for p in ALLOWED_DATA_PREFIXES)
    ]
    assert not unexpected, (
        f"Unexpected tracked files under data/: {unexpected}. "
        "Update .gitignore allow-list and this test together if intentional."
    )


@pytest.mark.parametrize("script_name", sorted(ALLOWED_SCRIPT_FILES))
def test_allowed_script_actually_present(project_root, script_name):
    assert (project_root / "scripts" / script_name).exists(), (
        f"Allow-listed script scripts/{script_name} is missing from working tree."
    )
