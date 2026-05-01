"""Sanity-check the landsat overlay assets that ship with the viewer.

These are the only data files committed to the repo — if any of them goes
missing or its bounds JSON gets corrupted, the LST overlay breaks at runtime.
"""

import json


def test_landsat_files_present(project_root):
    landsat_dir = project_root / "data" / "exposure" / "processed" / "landsat"
    expected = ["annual_lst_4326.png", "annual_lst_4326_cb.png", "annual_lst_bounds.json"]
    missing = [name for name in expected if not (landsat_dir / name).exists()]
    assert not missing, f"Missing landsat assets: {missing}"


def test_landsat_bounds_json_is_valid(project_root):
    bounds_path = (
        project_root
        / "data"
        / "exposure"
        / "processed"
        / "landsat"
        / "annual_lst_bounds.json"
    )
    payload = json.loads(bounds_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), (
        "annual_lst_bounds.json should be a JSON object."
    )
    # Don't assert on a specific schema — the build script reads this and the
    # viewer consumes whatever it produces. Just ensure it's parseable.


def test_landsat_pngs_nonempty(project_root):
    landsat_dir = project_root / "data" / "exposure" / "processed" / "landsat"
    for name in ("annual_lst_4326.png", "annual_lst_4326_cb.png"):
        size = (landsat_dir / name).stat().st_size
        assert size > 1024, f"{name} is only {size} bytes — looks truncated."
