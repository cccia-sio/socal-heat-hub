"""Static guards for the viewer's phone layout.

Under the drawer breakpoint (MOBILE_MQ below) the sidebar is an off-canvas
drawer, and the hamburger, zoom buttons, legend and logos sit over the top of
the map. Two bugs lived there:

* The guided tour spotlighted sidebar controls without opening the drawer, so
  the ring sat off-screen (x < 0) beside "Pick a mode, then Next".
* A tract popup autopanned with Leaflet's 5 px default. Leaflet's controls stack
  above the popup pane, so they covered its title, composite pill and
  Percentile/Value toggle.

Both fixes apply only under the breakpoint; desktop keeps its behaviour.
"""

import re

MOBILE_MQ = "(max-width:768px),(max-height:500px) and (orientation:landscape)"


def test_js_breakpoint_matches_the_drawer_css(viewer_html_text, js_fn):
    assert f"@media {MOBILE_MQ}{{" in viewer_html_text, (
        "The drawer's CSS breakpoint changed. Update MOBILE_MQ here and the "
        "query in _isMobileLayout() to match."
    )
    assert MOBILE_MQ in js_fn("_isMobileLayout"), (
        "_isMobileLayout() tests a different query than the drawer CSS, so the "
        "tour and popup fixes would switch on at the wrong width."
    )


# ── Bug A: tour spotlights off-screen controls ──────────────────────────────


def test_tour_drawer_sync_is_phone_only_and_follows_the_target(js_code):
    src = js_code("_tourSyncDrawer")
    assert "_isMobileLayout()" in src, (
        "_tourSyncDrawer() must do nothing on desktop, where the sidebar is "
        "always visible."
    )
    assert "toggleSidebar(" in src, "_tourSyncDrawer() no longer opens/closes the drawer."
    assert "getElementById('sidebar')" in src and ".contains(" in src, (
        "_tourSyncDrawer() must open the drawer only when the step's target is "
        "inside #sidebar, and close it for map or centered steps."
    )


def test_every_tour_step_syncs_the_drawer(js_code):
    src = js_code("_showTourStep")
    assert "_tourSyncDrawer(el)" in src, (
        "_showTourStep() doesn't open the drawer for a sidebar target; on a "
        "phone the spotlight lands off-screen."
    )
    assert "_tourSyncDrawer(null)" in src, (
        "_showTourStep() doesn't close the drawer for a centered step; the "
        "drawer would cover the map the step talks about."
    )


def test_tour_measures_the_target_after_the_drawer_slides_in(js_code):
    """getBoundingClientRect mid-slide returns the drawer's transformed x."""
    assert "_afterDrawerSlide(" in js_code("_showTourStep"), (
        "_showTourStep() positions the spotlight while the drawer is still "
        "sliding in, so the ring lands left of the control."
    )
    src = js_code("_afterDrawerSlide")
    assert "transitionend" in src and "setTimeout(" in src, (
        "_afterDrawerSlide() must wait for the drawer's transitionend, with a "
        "timer for when none fires."
    )


def test_tour_waits_for_a_slide_started_by_the_previous_step(js_code):
    """Next tapped mid-slide: the drawer is already marked open, so the new step
    doesn't toggle it, but the drawer is still moving and must be waited for."""
    assert "getAnimations" in js_code("_tourSyncDrawer"), (
        "_tourSyncDrawer() only reports a slide it started itself; a step shown "
        "while the previous step's slide is running measures the drawer mid-slide."
    )


def test_tour_done_step_says_where_find_address_is_on_a_phone(viewer_html_text):
    """On a phone the bottom-left stack is reparented into the drawer's Map Tools."""
    m = re.search(r'\{key:"done",[^}]*?body:"([^"]*)"', viewer_html_text)
    assert m, "The main tour's done step was not found."
    assert "Find address" not in m.group(1) or "Map Tools" in m.group(1), (
        "The done step places Find address bottom-left only; on a phone it is "
        "under Map Tools in the drawer."
    )


def test_tour_skips_positioning_a_step_the_user_left(js_code):
    assert "_currentSteps()[_tourIdx]!==step" in js_code("_showTourStep"), (
        "A deferred spotlight for a step the user already left would overwrite "
        "the current step's spotlight."
    )


# ── Bug B: map controls cover the tract popup ───────────────────────────────


def test_tract_popup_is_fitted_before_it_opens(js_code):
    """Leaflet runs click listeners in registration order and autopans inside
    the open call, so the fit has to be in the listener registered before
    bindPopup's own."""
    src = js_code("drawTracts")
    fit, bind = src.find("_fitPopupToMobileMap("), src.find("lyr.bindPopup(")
    assert fit != -1, "drawTracts() no longer fits the tract popup to the phone map."
    assert fit < bind, (
        "_fitPopupToMobileMap() is registered after bindPopup(), so the popup "
        "opens and autopans before its padding is set."
    )


def test_tract_popup_binding_keeps_desktop_options(js_code):
    m = re.search(r"lyr\.bindPopup\(h,(\{[^}]*\})\)", js_code("drawTracts"))
    assert m and m.group(1) == "{maxWidth:360,minWidth:344}", (
        "The tract popup's bound options changed. Phone-only padding belongs in "
        "_fitPopupToMobileMap(); static options would change desktop too."
    )


def test_popup_fit_clears_the_top_and_bottom_controls(js_code):
    src = js_code("_fitPopupToMobileMap")
    for sel in ("#menu-toggle", "#logos", "#legend", ".leaflet-top", ".leaflet-bottom"):
        assert sel in src, f"_fitPopupToMobileMap() no longer measures {sel}."
    for opt in ("autoPanPaddingTopLeft", "autoPanPaddingBottomRight", "maxHeight"):
        assert opt in src, f"_fitPopupToMobileMap() no longer sets {opt}."


def test_popup_fit_is_phone_only(js_code):
    assert "_isMobileLayout()" in js_code("_fitPopupToMobileMap"), (
        "_fitPopupToMobileMap() must leave desktop popups on Leaflet's defaults."
    )
