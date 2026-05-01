from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def viewer_html_path(project_root: Path) -> Path:
    return project_root / "main" / "diy_pca.html"


@pytest.fixture(scope="session")
def viewer_html_text(viewer_html_path: Path) -> str:
    if not viewer_html_path.exists():
        pytest.skip(f"{viewer_html_path} not built — run scripts/build_diy_pca_viewer.py")
    return viewer_html_path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def build_script_text(project_root: Path) -> str:
    return (project_root / "scripts" / "build_diy_pca_viewer.py").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def pages_workflow_text(project_root: Path) -> str:
    return (project_root / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
