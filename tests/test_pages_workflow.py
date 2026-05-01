"""Guard the GitHub Pages deploy workflow against the rsync-order gotcha.

CLAUDE.md spells this out: rsync is "first match wins," so every
`--include='data/...'` rule has to appear before the catch-all
`--exclude='data/**'`. If someone reorders these, deploys silently drop the
landsat tiles or pmtiles overlay and the viewer renders without basemaps.
"""

import re


def _rule_line_numbers(yml_text: str, predicate) -> list[int]:
    """Return 1-based line numbers where `predicate(stripped_line)` is True."""
    return [
        i
        for i, line in enumerate(yml_text.splitlines(), start=1)
        if predicate(line.strip())
    ]


def test_data_includes_precede_data_catchall_exclude(pages_workflow_text):
    include_lines = _rule_line_numbers(
        pages_workflow_text,
        lambda s: s.startswith("--include=") and "data/" in s,
    )
    exclude_lines = _rule_line_numbers(
        pages_workflow_text,
        # Tolerate trailing rsync line-continuation (`--exclude='data/**' \`).
        lambda s: s.startswith("--exclude='data/**'"),
    )

    assert include_lines, "No --include='data/...' rules found in pages.yml."
    assert exclude_lines, (
        "No --exclude='data/**' catch-all found. The workflow used to have "
        "one — if it's intentional, update this test."
    )

    last_include = max(include_lines)
    first_exclude = min(exclude_lines)
    assert last_include < first_exclude, (
        f"rsync rule order broken: a `data/` include is on line {last_include} "
        f"but the catch-all exclude is on line {first_exclude}. rsync is "
        "first-match-wins, so includes must come BEFORE the exclude. See "
        "CLAUDE.md for context."
    )


def test_build_tooling_excluded(pages_workflow_text):
    """scripts/, .github/, .git/ should never be deployed to Pages."""
    for path in (".git", ".github", "scripts"):
        pattern = rf"--exclude=['\"]{re.escape(path)}['\"]"
        assert re.search(pattern, pages_workflow_text), (
            f"Expected `--exclude='{path}'` in pages.yml — build tooling would "
            "otherwise leak into the published site."
        )


def test_workflow_triggers_on_main_push(pages_workflow_text):
    assert re.search(r"branches:\s*\[main\]", pages_workflow_text), (
        "pages.yml should still trigger on push to main."
    )
