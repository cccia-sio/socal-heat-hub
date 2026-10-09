"""Static checks on the viewer build script itself.

Full integration (running build_diy_pca_viewer.py end-to-end) requires the
gitignored CSV/GeoJSON inputs under data/, so we skip that here. What we CAN
check cheaply is that every placeholder token the script substitutes also
appears in its HTML_TEMPLATE — a typo there silently leaves an unreplaced
__FOO_PLACEHOLDER__ in the deployed viewer.
"""

import re


PLACEHOLDER_RE = re.compile(r"__[A-Z0-9_]+_PLACEHOLDER__")
REPLACE_RE = re.compile(
    r'\.replace\(\s*["\'](__[A-Z0-9_]+_PLACEHOLDER__)["\']'
)


def test_every_replace_call_has_matching_token_in_template(build_script_text):
    """If the script calls .replace("__FOO__", ...) but FOO never appears in
    HTML_TEMPLATE, the replace is a no-op and we ship broken HTML.
    """
    replaced = set(REPLACE_RE.findall(build_script_text))
    assert replaced, "No placeholder replace() calls found — has the script been refactored?"

    # Find HTML_TEMPLATE assignment and grab the rest of the file (template is
    # the long triple-quoted string at the bottom of the script).
    template_start = build_script_text.find("HTML_TEMPLATE")
    assert template_start != -1, "HTML_TEMPLATE not found in build script."
    template_region = build_script_text[template_start:]

    template_tokens = set(PLACEHOLDER_RE.findall(template_region))

    missing = sorted(replaced - template_tokens)
    assert not missing, (
        f"Placeholder(s) {missing} are .replace()d but never appear in "
        "HTML_TEMPLATE. Either fix the typo or remove the dead replace() call."
    )


def test_every_template_token_has_a_replace_call(build_script_text):
    """Inverse direction: a token in the template with no replace() call ships
    as literal `__FOO_PLACEHOLDER__` text in the viewer.
    """
    replaced = set(REPLACE_RE.findall(build_script_text))
    template_start = build_script_text.find("HTML_TEMPLATE")
    template_region = build_script_text[template_start:]
    template_tokens = set(PLACEHOLDER_RE.findall(template_region))

    orphans = sorted(template_tokens - replaced)
    assert not orphans, (
        f"Template tokens with no matching .replace() call: {orphans}. "
        "Add a replacement in main() or remove the token."
    )


def test_welcome_modal_pieces_in_template(build_script_text):
    """The welcome popup needs its coordinated pieces: modal markup, the
    dismiss button, and the closeWelcome handler. If any silently disappears,
    the popup either never shows or never dismisses. The popup is intentionally
    per-session (no localStorage gating) — it shows on every page load.
    """
    template_start = build_script_text.find("HTML_TEMPLATE")
    template = build_script_text[template_start:]

    assert 'id="welcome-modal"' in template, "welcome-modal markup missing from template."
    assert "closeWelcome" in template, "closeWelcome dismiss handler missing from template."
    assert 'id="wlc-got-it"' in template, "Got-it dismiss button missing from template."


def test_inlined_data_is_escaped_for_script_context(build_script_text):
    """Every data blob spliced into the viewer's <script> goes through _script_safe().

    Some strings come from outside services (Cool Zones names are from a County
    feature service); one containing `</script>` would end the tag and run what
    follows on the public site. The mailto is an href, not script, so it's exempt.
    """
    calls = re.findall(
        r'\.replace\(\s*"(__[A-Z0-9_]+_PLACEHOLDER__)",\s*(.+)\)$', build_script_text, re.M
    )
    assert calls, "No placeholder replace() calls found — has the script been refactored?"
    raw = [
        ph
        for ph, val in calls
        if ph != "__FEEDBACK_MAILTO_PLACEHOLDER__" and not val.startswith("_script_safe(")
    ]
    assert not raw, f"Inlined without _script_safe(): {raw}"
