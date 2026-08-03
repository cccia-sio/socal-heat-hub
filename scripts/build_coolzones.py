"""Rebuild the cool-zone site list from the County's two published sources.

Replaces a hand-transcribed CSV whose only provenance record was a filename in
a code comment. Two County sources, neither sufficient alone:

  * HLTH_COOL_ZONES_OES (ArcGIS FeatureServer) -- 72 sites WITH coordinates.
    A view over SDEP2.SANGIS.HLTH_COOL_ZONES, owned by cosd.hhsa. This is the
    geometry authority: HHSA Office of Business Intelligence geocodes the
    program's tabular data into that feature class. The sibling view
    HLTH_COOL_ZONES_PUBLIC went token-gated in 2026 and is no longer usable.

  * CoolZoneSiteList.pdf -- 80 sites, addresses only, no coordinates. This is
    the coverage authority: as of 2026-07-23 it is a strict superset of the
    service, carrying 8 sites the service omits (several backcountry: Fallbrook,
    Lakeside, Santa Ysabel, Valley Center).

So: take geometry from the service, take coverage from the PDF, and geocode
whatever the PDF has that the service doesn't (US Census, Nominatim fallback).

Designated Cool Zones only. An earlier CSV carried 28 extra City of San Diego
libraries and recreation centers with no upstream feed and no recorded origin.
Those were never Cool Zones: the County program already designates 18 City
library branches by name, and the extras were the facilities it did *not*
designate (it lists Pacific Beach Branch Library; the extras added Pacific Beach
Recreation Center). Undesignated sites carry no commitment to air conditioning,
heat-event hours, or capacity, so the map must not present them as refuge.

pdfplumber is optional. Without it the script still runs and emits the 72
service sites, logging what was skipped. It ships in the `heat-risk-sd` env
(the default `python3`), NOT in `socal-heat` -- which is the env the viewer
build needs. Run this with `python3`, then rebuild the viewer separately.

Writes:  data/cool_zones/coolzone_sites.csv   (overwritten from upstream)
           columns: id, organization, address, city, state, zip, phone,
                    hours, pet_policy, lat, lon, geocode_source, source

`source` is dataset provenance, currently "county" for every row. (In the
previous CSV this column held the geocoder name, which ARCHITECTURE.md
mis-described as provenance.) `geocode_source` records where the coordinates
came from: the County's own geocoding, or ours for the PDF-only sites.

Rebuild with:  python3 scripts/build_coolzones.py
Then:          python3 scripts/build_diy_pca_viewer.py   (socal-heat env)
"""

import csv
import json
import logging
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "cool_zones" / "coolzone_sites.csv"
PDF_CACHE = ROOT / "data" / "cool_zones" / "CoolZoneSiteList.pdf"

OES_QUERY = ("https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/"
             "services/HLTH_COOL_ZONES_OES/FeatureServer/0/query")
PDF_URL = ("https://www.sandiegocounty.gov/content/dam/sdc/hhsa/programs/phs/"
           "ExtremeHeat/coolzones/CoolZoneSiteList.pdf")

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
UA = "socal-heat-hub/1.0 (+https://github.com/cccia-sio/socal-heat-hub)"

# San Diego County bounding box -- a geocoder that lands outside it is wrong.
SD_BBOX = (-117.7, 32.4, -116.0, 33.6)
DUPE_METRES = 250          # same street address within this range == same building

# Matches the service's own maxRecordCount. ArcGIS silently truncates at that
# ceiling, and the geojson response format -- unlike f=json -- carries no
# `exceededTransferLimit` flag, so a full page is the ONLY truncation signal
# available. 72 sites today, but a silent drop would quietly shrink the map.
PAGE_SIZE = 2000

# PDF table layout, column index per field. The County's table is wider than the
# fields we take (pet policy sits at 9), so a narrower row means the layout
# moved. Organization and Address are load-bearing -- without them a site can be
# neither identified nor geocoded -- and the rest are display-only.
PDF_COLS = {"organization": 0, "address": 1, "city": 2, "zip": 3,
            "phone": 4, "hours": 5, "pet_policy": 9}
PDF_MIN_COLS = max(PDF_COLS.values()) + 1

FIELDS = ["id", "organization", "address", "city", "state", "zip", "phone",
          "hours", "pet_policy", "lat", "lon", "geocode_source", "source"]


def _get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def clean(s):
    """Collapse PDF cell whitespace, including hyphen-split city names."""
    s = re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip()
    return re.sub(r"(?<=-) (?=\w)", "", s)          # "Cardiff-by- the-Sea"


def _cell(row, field):
    """PDF cell by field name, or '' if the row is too narrow to reach it."""
    i = PDF_COLS[field]
    return clean(row[i]) if i < len(row) else ""


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def norm_addr(s):
    """Street-address key: drop unit numbers, city suffixes, street-type words."""
    s = (s or "").lower()
    s = re.sub(r"#\s*\w+", " ", s)
    s = re.sub(r"\b(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln"
               r"|way|court|ct|place|pl|highway|hwy|parkway|pkwy)\b\.?", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def in_sd(lat, lon):
    w, s, e, n = SD_BBOX
    return s <= lat <= n and w <= lon <= e


def metres(a, b):
    """Great-circle distance between (lat, lon) pairs."""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def fetch_service():
    """County sites with authoritative County-geocoded coordinates."""
    url = OES_QUERY + "?" + urllib.parse.urlencode({
        "where": "1=1", "outFields": "*", "outSR": 4326,
        "f": "geojson", "resultRecordCount": PAGE_SIZE})
    feats = json.loads(_get(url))["features"]
    if len(feats) >= PAGE_SIZE:
        raise RuntimeError(
            f"the feature service returned a full page ({len(feats)} features), "
            "so the response is probably truncated and sites are missing. "
            "f=geojson reports no exceededTransferLimit flag, so this count is "
            "the only warning available. Page the query with resultOffset before "
            "trusting the output.")
    out = []
    for f in feats:
        g, p = f.get("geometry"), f["properties"]
        if not g:
            logger.warning(f"service site has no geometry: {p.get('Organization')}")
            continue
        lon, lat = g["coordinates"][:2]
        out.append({
            "organization": clean(p.get("Organization")),
            "address": clean(p.get("Address")),
            "city": clean(p.get("City")),
            "state": clean(p.get("State")) or "CA",
            "zip": str(p.get("ZipCode") or "").strip(),
            "phone": clean(p.get("Phone")),
            "hours": clean(p.get("OperationalHours")),
            "pet_policy": clean(p.get("PetPolicy")),
            "lat": lat, "lon": lon, "geocode_source": "oes_service",
        })
    logger.info(f"OES feature service: {len(out)} County sites with coordinates")
    return out


def fetch_pdf_sites():
    """County site list -- fuller coverage than the service, but no coordinates."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed in this interpreter -- skipping "
                       "the PDF. Output will cover only the feature service, "
                       "which omits ~8 sites. Run with the heat-risk-sd env.")
        return None

    PDF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PDF_CACHE.write_bytes(_get(PDF_URL))
    logger.info(f"downloaded {PDF_URL.rsplit('/', 1)[-1]} "
                f"({PDF_CACHE.stat().st_size // 1024} KB)")

    sites, data_rows, narrow = [], 0, 0
    with pdfplumber.open(PDF_CACHE) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            for row in table:
                org = _cell(row, "organization")
                if not org or org == "Organization":
                    continue                    # spillover row or repeated header
                data_rows += 1
                if len(row) < PDF_MIN_COLS:
                    narrow += 1
                # Address is not optional: it is what the geocoder resolves for
                # the sites the feature service omits, so a row without one
                # cannot reach the map at all. Say so rather than emitting a
                # site that silently fails to place later.
                address = _cell(row, "address")
                if not address:
                    logger.error(f"  DROPPED, no address to geocode: {org}")
                    continue
                sites.append({
                    "organization": org, "address": address,
                    "city": _cell(row, "city"), "state": "CA",
                    "zip": _cell(row, "zip"), "phone": _cell(row, "phone"),
                    "hours": _cell(row, "hours"),
                    "pet_policy": _cell(row, "pet_policy"),
                })
    if narrow:
        logger.warning(
            f"{narrow}/{data_rows} PDF rows have fewer than {PDF_MIN_COLS} "
            "columns -- the County's table layout has probably changed. Sites "
            "still load, but hours and pet policy may be blank; check PDF_COLS "
            "against the published PDF.")
    logger.info(f"CoolZoneSiteList.pdf: {len(sites)} County sites (no coordinates)")
    return sites


def geocode_census(site):
    url = CENSUS_URL + "?" + urllib.parse.urlencode({
        "address": f"{site['address']}, {site['city']}, CA {site['zip']}",
        "benchmark": "Public_AR_Current", "format": "json"})
    matches = json.loads(_get(url, timeout=60))["result"]["addressMatches"]
    if not matches:
        return None
    c = matches[0]["coordinates"]
    return c["y"], c["x"]


def geocode_nominatim(site):
    url = NOMINATIM_URL + "?" + urllib.parse.urlencode({
        "q": f"{site['address']}, {site['city']}, CA {site['zip']}",
        "format": "json", "limit": 1, "countrycodes": "us"})
    hits = json.loads(_get(url, timeout=60))
    time.sleep(1.1)                                  # Nominatim: 1 req/sec
    return (float(hits[0]["lat"]), float(hits[0]["lon"])) if hits else None


def geocode(site):
    """Census first, Nominatim as fallback; reject anything outside the county."""
    for name, fn in (("census", geocode_census), ("nominatim", geocode_nominatim)):
        try:
            hit = fn(site)
        except Exception as exc:
            logger.warning(f"  {name} failed for {site['organization']}: {exc}")
            continue
        if hit and in_sd(*hit):
            return hit[0], hit[1], name
        if hit:
            logger.warning(f"  {name} put {site['organization']} outside SD County "
                           f"({hit[0]:.4f}, {hit[1]:.4f}) -- rejected")
    return None


def build_county(service, pdf_sites):
    """Merge: PDF sets coverage, the service supplies coordinates where it can."""
    by_name = {norm_name(s["organization"]): s for s in service}
    by_addr = {norm_addr(s["address"]) + "|" + norm_name(s["city"]): s
               for s in service}

    if pdf_sites is None:                            # no pdfplumber -- service only
        return sorted(service, key=lambda s: s["organization"].lower())

    county, matched, to_geocode = [], set(), []
    for site in pdf_sites:
        hit = (by_name.get(norm_name(site["organization"]))
               or by_addr.get(norm_addr(site["address"]) + "|" + norm_name(site["city"])))
        if hit:
            matched.add(id(hit))
            site["lat"], site["lon"] = hit["lat"], hit["lon"]
            site["geocode_source"] = "oes_service"
            county.append(site)
        else:
            to_geocode.append(site)

    if to_geocode:
        logger.info(f"{len(to_geocode)} PDF sites absent from the service -- geocoding")
    for site in to_geocode:
        hit = geocode(site)
        if not hit:
            logger.error(f"  UNPLACED, dropped: {site['organization']} "
                         f"({site['address']}, {site['city']})")
            continue
        site["lat"], site["lon"], site["geocode_source"] = hit
        logger.info(f"  {hit[2]}: {site['organization']}")
        county.append(site)

    for s in service:                    # service-only sites: keep, but flag
        if id(s) not in matched:
            logger.warning(f"in service but not in the PDF: {s['organization']}")
            county.append(s)

    return sorted(county, key=lambda s: s["organization"].lower())


def main():
    service = fetch_service()
    pdf_sites = fetch_pdf_sites()
    county = build_county(service, pdf_sites)

    # The County lists the same building twice under different city names
    # (La Jolla/Riford as both "La Jolla" and "San Diego"), so the city can't be
    # part of the key. Street address alone collides across cities -- hence the
    # distance guard.
    rows, kept = [], []
    for s in county:
        dupe = next((k for k in kept
                     if norm_addr(k["address"]) == norm_addr(s["address"])
                     and metres((k["lat"], k["lon"]),
                                (s["lat"], s["lon"])) < DUPE_METRES), None)
        if dupe:
            logger.info(f"dropping duplicate site: {s['organization']} == "
                        f"{dupe['organization']}")
            continue
        kept.append(s)
    for i, s in enumerate(kept, 1):
        rows.append({**s, "id": f"sd{i:03d}", "source": "county"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    logger.info(f"wrote {len(rows)} designated Cool Zone sites to "
                f"{OUT_CSV.relative_to(ROOT)}")
    logger.info("rebuild the viewer to publish: python3 scripts/build_diy_pca_viewer.py")


if __name__ == "__main__":
    main()
