"""Front-end security regressions for the generated viewer.

These assert against `main/heat_risk_map.html` (what actually ships), but every
fix belongs in `HTML_TEMPLATE` inside scripts/build_diy_pca_viewer.py — the
generated file is rebuilt from it and hand-edits are lost.
"""

import re

import pytest


_REMOTE_SCRIPT = re.compile(r'<script\b[^>]*\bsrc="https://[^"]+"[^>]*>', re.I)
_REMOTE_STYLESHEET = re.compile(
    r'<link\b(?=[^>]*\brel="stylesheet")[^>]*\bhref="https://[^"]+"[^>]*>', re.I
)
_BLANK_ANCHOR = re.compile(r'<a\b[^>]*target="_blank"[^>]*>', re.I)


def test_geocoder_result_escaped_before_innerhtml(js_code):
    """Nominatim `display_name` reaches innerHTML via _addrSetStatus.

    Nominatim is backed by OpenStreetMap, which anyone can edit — a POI named
    `<img src=x onerror=...>` is stored XSS against every user who searches a
    matching term. The 80-char truncation is not a mitigation; a working payload
    fits in 28. The autocomplete dropdown always escaped via _addrEsc(); this
    path (the confirmed-match status line) did not.
    """
    src = js_code("_addrApplyHit")
    assert "_addrEsc(dnShort)" in src, (
        "_addrApplyHit builds the status line from the geocoder's display_name "
        "without escaping it. That string is written to innerHTML — escape it "
        "with _addrEsc() as the suggestion dropdown already does."
    )
    assert "'+dnShort+'" not in src, (
        "Raw dnShort is still concatenated into an HTML string in _addrApplyHit."
    )


@pytest.mark.parametrize("pattern", [_REMOTE_SCRIPT, _REMOTE_STYLESHEET])
def test_cdn_resources_pinned_with_sri(viewer_html_text: str, pattern: re.Pattern):
    """A CDN compromise otherwise yields arbitrary JS with full page privilege."""
    unpinned = [
        tag for tag in pattern.findall(viewer_html_text) if "integrity=" not in tag
    ]
    assert not unpinned, (
        "Remote resources loaded without a Subresource Integrity hash:\n  "
        + "\n  ".join(unpinned)
        + "\nAdd integrity=\"sha384-...\" + crossorigin=\"anonymous\" in "
        "HTML_TEMPLATE. Compute with:\n"
        "  curl -sL <url> | openssl dgst -sha384 -binary | openssl base64 -A"
    )


def test_dynamic_script_loader_pins_integrity(js_code, viewer_html_text: str):
    """jsPDF is injected at PDF-export time; crossOrigin alone verifies nothing."""
    src = js_code("_loadScript")
    assert "s.integrity" in src, (
        "_loadScript sets crossOrigin but never .integrity — it will execute "
        "whatever bytes the CDN returns. Accept an integrity argument and set it."
    )
    assert re.search(r"_loadScript\(\s*'https://[^']+'\s*,\s*'sha\d{3}-", viewer_html_text), (
        "_loadScript() is called without an SRI hash argument."
    )


def test_blank_anchors_have_noopener(viewer_html_text: str):
    bad = [
        tag for tag in _BLANK_ANCHOR.findall(viewer_html_text) if "noopener" not in tag
    ]
    assert not bad, (
        "target=\"_blank\" anchors missing rel=\"noopener\" (reverse tabnabbing — "
        "the opened page gets a handle on window.opener):\n  " + "\n  ".join(bad)
    )


def test_no_analytics_or_telemetry(viewer_html_text: str):
    """README promises no tracking. Keep it true."""
    trackers = [
        "google-analytics", "googletagmanager", "gtag(", "plausible.io",
        "hotjar", "mixpanel", "segment.com", "fbq(", "matomo", "clarity.ms",
    ]
    found = [t for t in trackers if t in viewer_html_text.lower()]
    assert not found, f"Tracking/analytics found in the viewer: {found}"
