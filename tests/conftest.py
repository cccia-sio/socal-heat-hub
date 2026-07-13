import re
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def viewer_html_path(project_root: Path) -> Path:
    return project_root / "main" / "heat_risk_map.html"


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


# The viewer ships as one big inlined script, so most guards here work by
# slicing a function out of it and asserting on its source. Top-level functions
# in HTML_TEMPLATE always start at column 0 and nested/anonymous ones are
# indented, so "next line starting with `function`" is a reliable end marker —
# and it beats the hand-rolled `.index(...)` pairs these tests used to use,
# which broke whenever a neighbouring function was renamed or moved.
_TOP_LEVEL_FN = re.compile(r"^function \w+\(", re.MULTILINE)


@pytest.fixture(scope="session")
def js_fn(viewer_html_text: str):
    """Return `js_fn("name")` -> the source of that top-level JS function."""

    def _source(name: str) -> str:
        match = re.search(rf"^function {re.escape(name)}\(", viewer_html_text, re.MULTILINE)
        assert match, (
            f"Top-level JS function `{name}()` not found in the generated viewer. "
            "Was it renamed, inlined, or is main/heat_risk_map.html stale? "
            "Rebuild with: python3 scripts/build_diy_pca_viewer.py"
        )
        nxt = _TOP_LEVEL_FN.search(viewer_html_text, match.end())
        return viewer_html_text[match.start() : nxt.start() if nxt else len(viewer_html_text)]

    return _source


def strip_js_comments(src: str) -> str:
    """Drop `//` comments, leaving string literals intact.

    Needed for any "this bug is NOT present" assertion. The code that fixes a bug
    usually also *describes* it, so `assert "_rgbTriplet" not in src` happily trips
    over the comment `// _rgbTriplet only reads rgb()`. Both of the guards in
    test_tract_pdf and test_risk_orientation hit this.

    Can't just cut at the first `//` — the viewer is full of 'https://...' URLs —
    so quotes are tracked.
    """
    kept = []
    for line in src.splitlines():
        quote = None
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch == "/" and line[i + 1 : i + 2] == "/":
                line = line[:i]
                break
            i += 1
        kept.append(line)
    return "\n".join(kept)


@pytest.fixture(scope="session")
def js_code(js_fn):
    """`js_code("name")` -> that function's source with comments stripped.

    Use this for negative assertions; use `js_fn` when comments are harmless.
    """
    return lambda name: strip_js_comments(js_fn(name))
