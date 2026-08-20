#!/usr/bin/env python3
"""
STEP 2 — retrieve addresses and assign them to blocks 1..26.

This version accepts:
    --calibration calibration.json
and intentionally fetches addresses from a larger area than the red boundary
to tolerate calibration error.

Usage:
    python3 saratoga_block_addresses.py data/saratoga.png \
        --calibration calibration.json

Optional:
    --margin-m 150

Outputs:
    addresses_by_block.csv
    block_debug.png

Dependencies:
    pip install opencv-python numpy pandas requests scikit-image shapely pyproj
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform, unary_union
from skimage.segmentation import watershed


BLOCK_SEEDS = {
     1: (192, 677),  2: (205, 900),  3: (387, 971),  4: (404, 808),
     5: (402, 634),  6: (678, 642),  7: (632, 817),  8: (578, 970),
     9: (773, 911), 10: (929, 778), 11: (636, 299), 12: (622, 431),
    13: (805, 312), 14: (864, 547), 15: (1015, 310), 16: (1017, 448),
    17: (1094, 631), 18: (1285, 639), 19: (1071, 845), 20: (976, 1015),
    21: (1487, 642), 22: (1697, 700), 23: (1274, 845), 24: (1500, 844),
    25: (1228, 1015), 26: (1441, 1015),
}

OUTER_POLYGON_PIXELS = np.array([
    (90, 545), (434, 545), (454, 509), (472, 478), (500, 451),
    (473, 405), (501, 374), (500, 200), (1015, 202), (1040, 177),
    (1074, 80), (1138, 103), (1138, 547), (1648, 548), (1652, 590),
    (1738, 624), (1786, 672), (1795, 704), (1760, 741), (1646, 786),
    (1646, 1078), (272, 1078), (272, 1008), (88, 956),
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Known boundary-control parcels
#
# These are used only for parcels whose visual assignment is known to be
# sensitive to a few pixels of georeferencing error. APN is preferable to
# address text because it is the stable county parcel identifier.
#
# 20090 Kilbride Dr = APN 39330022. It belongs to block 23 in the original
# hand-drawn map, even though the projected parcel can overlap block 24 slightly.
# ---------------------------------------------------------------------------
PARCEL_BLOCK_OVERRIDES = {
    "39330022": 23,
}




def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = data.get("points", [])
    if len(pts) < 4:
        raise ValueError("Need at least 4 calibration points")

    return [
        (
            float(p["pixel_x"]),
            float(p["pixel_y"]),
            float(p["longitude"]),
            float(p["latitude"]),
        )
        for p in pts
    ]


def detect_block_barriers(image):
    """
    Detect only the TRUE block boundaries: black outlines + red outer outline.

    IMPORTANT: green loops are deliberately ignored. They are highlights drawn
    inside some blocks, not block boundaries.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 90]))

    # Keep thick hand-drawn black lines while removing much of the map text.
    black = cv2.morphologyEx(
        black,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    barrier = cv2.bitwise_or(black, red)

    # Seal small pen gaps.
    barrier = cv2.morphologyEx(
        barrier,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    barrier = cv2.dilate(
        barrier,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    return barrier


def segment_blocks(image):
    """
    Segment blocks by connected free-space regions bounded by the black/red pen.

    Why this replaces watershed:
    The old watershed leaked through small gaps. In the supplied screenshot
    that made block 21 enormous (~289k pixels) and caused 30+ parcels to be
    assigned there.

    This method:
      1. treats black/red pen as walls,
      2. finds connected free-space components,
      3. maps each component to its numbered seed,
      4. if two seeds share a component because a line has a tiny gap,
         splits that component by nearest-seed Voronoi.

    On the original screenshot, block 21 becomes ~30k pixels instead of ~289k.
    """
    barrier = detect_block_barriers(image)

    outer_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(
        outer_mask,
        [OUTER_POLYGON_PIXELS.astype(np.int32)],
        1,
    )

    free = ((outer_mask > 0) & (barrier == 0)).astype(np.uint8)

    _, components = cv2.connectedComponents(free, connectivity=8)

    labels = np.zeros(image.shape[:2], dtype=np.int32)

    component_seeds = {}

    for block, (x, y) in BLOCK_SEEDS.items():
        component_id = int(components[y, x])

        if component_id == 0:
            # Extremely defensive fallback: search a small neighborhood.
            found = None
            for radius in range(1, 16):
                y0 = max(0, y - radius)
                y1 = min(components.shape[0], y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(components.shape[1], x + radius + 1)

                values = components[y0:y1, x0:x1]
                values = values[values > 0]

                if values.size:
                    counts = np.bincount(values)
                    found = int(np.argmax(counts))
                    break

            if found is None:
                raise RuntimeError(f"Could not place block seed {block}")

            component_id = found

        component_seeds.setdefault(component_id, []).append((block, x, y))

    for component_id, seeds in component_seeds.items():
        ys, xs = np.where(components == component_id)

        if len(seeds) == 1:
            labels[ys, xs] = seeds[0][0]
            continue

        # A tiny line gap joined multiple visual blocks. Split the connected
        # component by the nearest numbered seed.
        distances = np.stack(
            [
                (xs - seed_x) ** 2 + (ys - seed_y) ** 2
                for _, seed_x, seed_y in seeds
            ],
            axis=1,
        )

        nearest = np.argmin(distances, axis=1)

        for seed_index, (block, _, _) in enumerate(seeds):
            choose = nearest == seed_index
            labels[ys[choose], xs[choose]] = block

    return labels


def labels_to_block_geometry(labels):
    """
    Convert each block's pixel mask to a Shapely polygon/multipolygon.

    These geometries are later intersected with parcel polygons. This is more
    reliable than assigning a parcel from only one centroid point, especially
    for edge parcels such as 19969 Scotland Dr.
    """
    result = {}

    for block in range(1, 27):
        mask = (labels == block).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        polys = []

        for contour in contours:
            pts = contour.reshape(-1, 2)

            if len(pts) < 3:
                continue

            poly = Polygon([(float(x), float(y)) for x, y in pts])

            if not poly.is_valid:
                poly = poly.buffer(0)

            if not poly.is_empty and poly.area > 25:
                polys.append(poly)

        if polys:
            result[block] = unary_union(polys)

    return result



def fit_homography(control_points, forward=True):
    if forward:
        src = [[x, y] for x, y, lon, lat in control_points]
        dst = [[lon, lat] for x, y, lon, lat in control_points]
    else:
        src = [[lon, lat] for x, y, lon, lat in control_points]
        dst = [[x, y] for x, y, lon, lat in control_points]

    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float64),
        np.array(dst, dtype=np.float64),
        method=0,
    )

    if H is None:
        raise RuntimeError("Could not fit homography")

    return H


def transform_points(points, H):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def original_geo_polygon(pixel_to_geo):
    coords = transform_points(OUTER_POLYGON_PIXELS, pixel_to_geo)
    poly = Polygon(coords)
    return poly if poly.is_valid else poly.buffer(0)


def buffer_polygon_meters(poly, meters):
    c = poly.centroid
    zone = int((c.x + 180) / 6) + 1
    epsg = 32600 + zone if c.y >= 0 else 32700 + zone

    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(epsg)

    to_utm = Transformer.from_crs(wgs84, utm, always_xy=True).transform
    to_wgs = Transformer.from_crs(utm, wgs84, always_xy=True).transform

    poly_utm = shapely_transform(to_utm, poly)
    buffered_utm = poly_utm.buffer(meters)
    return shapely_transform(to_wgs, buffered_utm)


# ---------------------------------------------------------------------------
# Santa Clara County parcel GIS
#
# This is much better for this task than OpenStreetMap because the county
# parcel layer includes a "Situs_Address_Full" field for the property address.
# ---------------------------------------------------------------------------

PARCEL_QUERY_URL = (
    "https://services2.arcgis.com/tcv2cMrq63AgvbHF/"
    "arcgis/rest/services/Parcels_Public_View/FeatureServer/0/query"
)


def download_addresses(poly):
    """
    Query Santa Clara County parcels using ArcGIS JSON rather than GeoJSON.

    Important change:
    - We no longer rely on GeoJSON CRS behavior.
    - We ask ArcGIS for geometry in EPSG:4326.
    - If ArcGIS still returns California State Plane coordinates, the parser
      detects that and converts EPSG:2227 -> EPSG:4326.
    - We accept the whole expanded query envelope; you said extra addresses
      are OK and can be manually removed.
    """
    min_lon, min_lat, max_lon, max_lat = poly.bounds

    print(
        "Expanded geographic bbox: "
        f"lon {min_lon:.6f} .. {max_lon:.6f}, "
        f"lat {min_lat:.6f} .. {max_lat:.6f}"
    )

    params = {
        "f": "json",
        "where": "Situs_Address_Full IS NOT NULL",
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID_1,APN,Situs_Address_Full,AGOL_Situs_Search",
        "returnGeometry": "true",
        "outSR": "4326",
        "resultRecordCount": 2000,
    }

    print("Querying Santa Clara County parcel GIS...")

    r = requests.get(
        PARCEL_QUERY_URL,
        params=params,
        timeout=120,
        headers={"User-Agent": "saratoga-block-address-extractor/6.0"},
    )
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"County GIS error: {data['error']}")

    features = data.get("features", [])
    print(f"County GIS returned {len(features)} parcel features.")

    if not features:
        raise RuntimeError("County GIS returned no parcel features.")

    return data


def _convert_xy_to_lonlat(x, y):
    """
    ArcGIS should honor outSR=4326, but handle native EPSG:2227 coordinates
    defensively if it doesn't.
    """
    x = float(x)
    y = float(y)

    if -180 <= x <= 180 and -90 <= y <= 90:
        return x, y

    transformer = Transformer.from_crs(
        CRS.from_epsg(2227),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)


def geometry_to_lonlat_polygons(feature):
    """
    Convert ArcGIS parcel rings into usable Shapely polygons in lon/lat.

    County parcels in this neighborhood are simple polygons in practice.
    For multipart records we keep each valid ring and union them.
    """
    geom = feature.get("geometry") or {}
    rings = geom.get("rings") or []

    polys = []

    for ring in rings:
        coords = []

        for pair in ring:
            if len(pair) < 2:
                continue
            coords.append(_convert_xy_to_lonlat(pair[0], pair[1]))

        if len(coords) < 4:
            continue

        try:
            p = Polygon(coords)

            if not p.is_valid:
                p = p.buffer(0)

            if not p.is_empty and p.area > 0:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        return None

    return unary_union(polys)


def lonlat_geometry_to_pixel(geom, geo_to_pixel):
    """
    Project a Shapely parcel polygon from lon/lat to screenshot pixels.
    """
    def convert_coords(coords):
        arr = np.asarray(coords, dtype=np.float64)

        if arr.size == 0:
            return []

        xy = transform_points(arr[:, :2], geo_to_pixel)
        return [(float(x), float(y)) for x, y in xy]

    def one_polygon(poly):
        exterior = convert_coords(poly.exterior.coords)

        holes = [
            convert_coords(interior.coords)
            for interior in poly.interiors
        ]

        out = Polygon(exterior, holes)

        if not out.is_valid:
            out = out.buffer(0)

        return out

    if geom.geom_type == "Polygon":
        return one_polygon(geom)

    if geom.geom_type == "MultiPolygon":
        parts = [one_polygon(p) for p in geom.geoms]
        parts = [p for p in parts if not p.is_empty]
        return unary_union(parts) if parts else None

    return None


def normalize_county_address(raw):
    """
    Turn county strings such as:
        13284 BEAUMONT AV SARATOGA CA 95070-5050
    into:
        13284 Beaumont Ave, Saratoga, CA

    This matches the user's blocks.json format.
    """
    text = " ".join(str(raw or "").strip().split())

    if not text:
        return "", "", ""

    # Remove Saratoga/CA/ZIP tail while preserving the street portion.
    m = re.match(
        r"^\s*([0-9]+[A-Za-z0-9\-]*)\s+(.+?)\s+SARATOGA\s+CA(?:\s+\d{5}(?:-\d{4})?)?\s*$",
        text,
        flags=re.IGNORECASE,
    )

    if m:
        house = m.group(1)
        street_raw = m.group(2)
    else:
        # Fallback if county formatting changes.
        m = re.match(r"^\s*([0-9]+[A-Za-z0-9\-]*)\s+(.+)$", text)
        if not m:
            return text.title(), "", ""
        house = m.group(1)
        street_raw = m.group(2)
        street_raw = re.sub(
            r"\s+SARATOGA\s+CA(?:\s+\d{5}(?:-\d{4})?)?\s*$",
            "",
            street_raw,
            flags=re.IGNORECASE,
        )

    tokens = street_raw.upper().split()

    suffixes = {
        "AV": "Ave",
        "AVE": "Ave",
        "AVENUE": "Ave",
        "DR": "Dr",
        "DRIVE": "Dr",
        "CT": "Ct",
        "COURT": "Ct",
        "LN": "Ln",
        "LANE": "Ln",
        "WAY": "Way",
        "RD": "Rd",
        "ROAD": "Rd",
        "PL": "Pl",
        "PLACE": "Pl",
        "CIR": "Cir",
        "CIRCLE": "Cir",
    }

    pretty = []

    for token in tokens:
        if token in suffixes:
            pretty.append(suffixes[token])
        else:
            pretty.append(token.title())

    street = " ".join(pretty)
    address = f"{house} {street}, Saratoga, CA"

    return address, house, street


def assign_parcel_to_block(parcel_pixel, block_geometries):
    """
    Assign a parcel conservatively.

    v5 always compared against block_geom.buffer(6). That was useful for edge
    parcels, but it could pull a neighboring parcel across a boundary. Blocks
    3 and 15 are especially sensitive because several parcels sit very close
    to their drawn borders.

    New strategy:
      1. First use the UNBUFFERED block polygons.
      2. Prefer a block containing the parcel representative point.
      3. Otherwise choose the block with the largest real polygon overlap.
      4. Only use a small 4-pixel buffer as an edge-case fallback.
      5. The fallback must cover at least 20% of the parcel and must clearly
         beat the second-best candidate.

    This keeps true boundary parcels while rejecting "barely touching" extras.
    """
    if parcel_pixel is None or parcel_pixel.is_empty or parcel_pixel.area <= 0:
        return "", 0.0

    parcel_area = float(parcel_pixel.area)
    rep = parcel_pixel.representative_point()

    # ------------------------------------------------------------------
    # Pass 1: unbuffered geometry.
    # ------------------------------------------------------------------
    candidates = []

    for block, block_geom in block_geometries.items():
        try:
            overlap = float(parcel_pixel.intersection(block_geom).area)
            contains_rep = bool(block_geom.covers(rep))
        except Exception:
            continue

        if overlap > 0:
            candidates.append((block, overlap, contains_rep))

    # If the representative point is inside one or more candidate blocks,
    # choose the one with the greatest actual overlap.
    rep_candidates = [c for c in candidates if c[2]]

    if rep_candidates:
        rep_candidates.sort(key=lambda x: x[1], reverse=True)
        block, overlap, _ = rep_candidates[0]

        # Avoid a pathological sliver caused by imperfect geometry.
        if overlap / parcel_area >= 0.10:
            return int(block), overlap

    # Otherwise use actual polygon overlap, but require a meaningful fraction.
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)

        best_block, best_overlap, _ = candidates[0]
        second_overlap = candidates[1][1] if len(candidates) > 1 else 0.0

        best_fraction = best_overlap / parcel_area

        # A normal parcel should have a substantial portion inside its block.
        # Also require the winner to beat the runner-up when the parcel spans
        # a drawn boundary.
        clear_winner = (
            second_overlap <= 0
            or best_overlap >= second_overlap * 1.20
        )

        if best_fraction >= 0.18 and clear_winner:
            return int(best_block), best_overlap

    # ------------------------------------------------------------------
    # Pass 2: small buffer only for genuine edge parcels.
    # ------------------------------------------------------------------
    edge_candidates = []

    for block, block_geom in block_geometries.items():
        try:
            overlap = float(
                parcel_pixel.intersection(block_geom.buffer(4)).area
            )
        except Exception:
            continue

        if overlap > 0:
            edge_candidates.append((block, overlap))

    if not edge_candidates:
        return "", 0.0

    edge_candidates.sort(key=lambda x: x[1], reverse=True)

    best_block, best_overlap = edge_candidates[0]
    second_overlap = (
        edge_candidates[1][1]
        if len(edge_candidates) > 1
        else 0.0
    )

    best_fraction = best_overlap / parcel_area

    # Conservative fallback: at least 20% of the parcel must be captured,
    # and the winner must be distinctly better than the neighboring block.
    clear_winner = (
        second_overlap <= 0
        or best_overlap >= second_overlap * 1.35
    )

    if best_fraction >= 0.20 and clear_winner:
        return int(best_block), best_overlap

    return "", 0.0


def assign_inside_point_fallback(parcel_pixel, block_geometries):
    """
    Safety fallback used ONLY when the parcel's representative address point is
    inside the big red boundary but the normal conservative classifier returned
    no block.

    Unlike v7, this never treats "parcel polygon intersects red boundary" as
    enough. That was the bug that pulled the school and other outside parcels
    into blocks 21/22/3/11/12.

    Strategy:
      1. If the representative point lies inside a block polygon, use it.
      2. Else use maximum real (unbuffered) parcel/block overlap.
      3. Final fallback: nearest block seed to the representative point.

    No block buffering is used here, so outside parcels cannot be pulled across
    the big red boundary.
    """
    if parcel_pixel is None or parcel_pixel.is_empty:
        return "", 0.0, "none"

    rep = parcel_pixel.representative_point()

    # 1) Representative point containment.
    containing = []
    for block, geom in block_geometries.items():
        try:
            if geom.covers(rep):
                overlap = float(parcel_pixel.intersection(geom).area)
                containing.append((block, overlap))
        except Exception:
            continue

    if containing:
        containing.sort(key=lambda x: x[1], reverse=True)
        block, overlap = containing[0]
        return int(block), overlap, "fallback_rep_point"

    # 2) Maximum real overlap.
    best_block = ""
    best_overlap = 0.0

    for block, geom in block_geometries.items():
        try:
            overlap = float(parcel_pixel.intersection(geom).area)
        except Exception:
            continue

        if overlap > best_overlap:
            best_overlap = overlap
            best_block = block

    if best_block and best_overlap > 0:
        return int(best_block), best_overlap, "fallback_real_overlap"

    # 3) Nearest seed to parcel representative point.
    rx, ry = float(rep.x), float(rep.y)
    nearest_block = min(
        BLOCK_SEEDS,
        key=lambda b: (BLOCK_SEEDS[b][0] - rx) ** 2 + (BLOCK_SEEDS[b][1] - ry) ** 2,
    )

    return int(nearest_block), 0.0, "fallback_nearest_seed"



def _polygon_to_mask(poly, shape):
    """
    Rasterize a Shapely Polygon/MultiPolygon into an image-sized uint8 mask.
    This lets us compare a parcel directly against the final block label image,
    which is the closest possible representation of the original hand drawing.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    if poly is None or poly.is_empty:
        return mask

    def draw_polygon(p):
        exterior = np.asarray(p.exterior.coords, dtype=np.float64)

        if len(exterior) >= 3:
            pts = np.rint(exterior[:, :2]).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

        # Remove holes, if any.
        for interior in p.interiors:
            ring = np.asarray(interior.coords, dtype=np.float64)
            if len(ring) >= 3:
                pts = np.rint(ring[:, :2]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 0)

    if poly.geom_type == "Polygon":
        draw_polygon(poly)

    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            draw_polygon(p)

    return mask


def correct_block_23_24(parcel_pixel, labels, block):
    """
    Correct only parcels near the 23/24 shared boundary by counting ACTUAL
    parcel pixels inside the final segmented block masks.

    This replaces the old x-coordinate rule.

    Why:
      The bottom-right parcel of block 23 extends close to block 24, but the
      original drawing has most of that parcel inside block 23. A centroid or
      x-midpoint rule can still call it 24. Pixel-overlap directly answers the
      intended question: which drawn block contains more of the parcel?

    Rules:
      - Only run for parcels initially assigned to 23 or 24.
      - Rasterize the parcel polygon in screenshot coordinates.
      - Count parcel pixels labeled 23 vs 24.
      - Choose whichever block has more actual overlap.
      - Require a modest 5% advantage to avoid noisy flips on a true 50/50 edge.
        If neither side has that advantage, keep the original assignment.
    """
    if block not in (23, 24):
        return block, "normal", 0, 0

    if parcel_pixel is None or parcel_pixel.is_empty:
        return block, "normal", 0, 0

    parcel_mask = _polygon_to_mask(parcel_pixel, labels.shape)

    area23 = int(np.count_nonzero((parcel_mask > 0) & (labels == 23)))
    area24 = int(np.count_nonzero((parcel_mask > 0) & (labels == 24)))

    total = area23 + area24

    if total == 0:
        return block, "normal", area23, area24

    # Require only a small but real majority. The user's described parcel
    # should have a clear 23 majority.
    margin = max(3, int(round(total * 0.05)))

    if area23 >= area24 + margin:
        return 23, "corrected_23_24_pixel_overlap", area23, area24

    if area24 >= area23 + margin:
        return 24, "corrected_23_24_pixel_overlap", area23, area24

    return block, "normal", area23, area24



def apply_parcel_override(attrs, address, block, assignment_method):
    """
    Apply a stable APN-based override for known boundary-control parcels.

    Returns:
        (block, assignment_method)
    """
    apn = str(attrs.get("APN", "") or "").strip()

    if apn in PARCEL_BLOCK_OVERRIDES:
        forced_block = int(PARCEL_BLOCK_OVERRIDES[apn])
        return forced_block, f"apn_override_{apn}"

    return block, assignment_method



def collect(county_json, fetch_poly, original_poly, geo_to_pixel, labels):
    h, w = labels.shape
    rows = []

    block_geometries = labels_to_block_geometry(labels)

    bad_geometry = 0
    no_address = 0

    for feature in county_json.get("features", []):
        attrs = feature.get("attributes", {}) or {}

        raw_address = str(attrs.get("Situs_Address_Full") or "").strip()

        if not raw_address:
            no_address += 1
            continue

        parcel_geo = geometry_to_lonlat_polygons(feature)

        if parcel_geo is None or parcel_geo.is_empty:
            bad_geometry += 1
            continue

        rep = parcel_geo.representative_point()
        lon, lat = float(rep.x), float(rep.y)
        pt = Point(lon, lat)

        # IMPORTANT:
        # "Inside the big red boundary" means the parcel's representative
        # address point is inside the red polygon. A parcel merely touching or
        # crossing the red line does NOT count. This prevents the school and
        # other outside parcels from being pulled into blocks.
        inside_red_boundary = bool(original_poly.covers(pt))
        inside_expanded_boundary = bool(fetch_poly.covers(pt))

        parcel_pixel = lonlat_geometry_to_pixel(parcel_geo, geo_to_pixel)

        block = ""
        overlap_px2 = 0.0
        overlap23_px = 0
        overlap24_px = 0
        assignment_method = "outside_red_boundary"

        if inside_red_boundary:
            block, overlap_px2 = assign_parcel_to_block(
                parcel_pixel,
                block_geometries,
            )
            assignment_method = "normal"

            # If an address point is truly inside the red boundary, make sure
            # it is not lost just because the conservative classifier rejected
            # a borderline parcel.
            if not block:
                block, overlap_px2, assignment_method = assign_inside_point_fallback(
                    parcel_pixel,
                    block_geometries,
                )

            # Targeted cleanup for the nearly vertical 23/24 divider.
            corrected_block, correction_method, overlap23_px, overlap24_px = correct_block_23_24(
                parcel_pixel,
                labels,
                block,
            )
            if corrected_block != block:
                block = corrected_block
                assignment_method = correction_method

        # Nearby parcels outside the red polygon are retained in the CSV for
        # manual review, but they are intentionally left with block="" so they
        # can never appear in blocks.json.
        px, py = transform_points([(lon, lat)], geo_to_pixel)[0]
        x, y = int(round(px)), int(round(py))

        address, house, street = normalize_county_address(raw_address)

        # Final stable correction for known edge parcels. This is intentionally
        # applied after the geometric classifier because it handles the tiny
        # number of parcels where screenshot georeferencing error is larger than
        # the visual gap between two block outlines.
        block, assignment_method = apply_parcel_override(
            attrs,
            address,
            block,
            assignment_method,
        )

        rows.append({
            "block": block,
            "inside_original_boundary": inside_red_boundary,
            "inside_expanded_boundary": inside_expanded_boundary,
            "address": address,
            "house_number": house,
            "street": street,
            "city": "Saratoga",
            "state": "CA",
            "postcode": "",
            "latitude": lat,
            "longitude": lon,
            "pixel_x": x,
            "pixel_y": y,
            "block_overlap_px2": round(overlap_px2, 2),
            "parcel_area_px2": round(float(parcel_pixel.area), 2) if parcel_pixel is not None else 0,
            "block_overlap_fraction": (
                round(float(overlap_px2) / float(parcel_pixel.area), 4)
                if parcel_pixel is not None and parcel_pixel.area > 0
                else 0
            ),
            "assignment_method": assignment_method,
            "block23_overlap_pixels": overlap23_px,
            "block24_overlap_pixels": overlap24_px,
            "apn": attrs.get("APN", ""),
            "objectid": attrs.get("OBJECTID_1", ""),
            "source": "Santa Clara County Parcels_Public_View",
        })

    print(
        f"Parsed {len(rows)} parcel addresses; "
        f"{bad_geometry} parcels had unusable geometry; "
        f"{no_address} had no site address."
    )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=["apn", "address"],
        keep="first",
    )

    df["_house"] = df["house_number"].map(house_sort)
    df["_block"] = pd.to_numeric(df["block"], errors="coerce").fillna(999)

    return (
        df.sort_values(
            ["_block", "street", "_house", "house_number"],
            kind="stable",
        )
        .drop(columns=["_house", "_block"])
        .reset_index(drop=True)
    )


def house_sort(value):
    m = re.match(r"\s*(\d+)", str(value))
    return int(m.group(1)) if m else 10**9


def detect_block_captains(image, labels):
    """
    Determine whether each block has a block captain from the hand-drawn color.

    Source-map convention:
      GREEN marking -> no block captain
      BLACK marking -> block captain exists

    Returns:
        dict[int, bool]
        True  = block captain exists
        False = no block captain
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(
        hsv,
        np.array([35, 80, 60]),
        np.array([95, 255, 255]),
    )

    # The green marker is thick and often centered on the border, so expand it.
    green = cv2.dilate(
        green,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    result = {}

    print("\nBlock captain color detection:")
    print("--------------------------------")

    for block in range(1, 27):
        block_mask = (labels == block).astype(np.uint8)

        # Look slightly around the block boundary too.
        expanded = cv2.dilate(
            block_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        )

        green_pixels = int(
            np.count_nonzero((green > 0) & (expanded > 0))
        )

        # Thick green loops create hundreds/thousands of pixels. This avoids
        # treating tiny map artifacts as a green block.
        is_green = green_pixels >= 120

        has_captain = not is_green
        result[block] = has_captain

        status = "BLACK -> captain" if has_captain else "GREEN -> no captain"
        print(
            f"  block {block:2d}: {status} "
            f"(green pixels={green_pixels})"
        )

    return result



def write_blocks_json(df, output_path, block_captains):
    """
    Output:
        {
            "1": {
                "active": true,
                "block_captain": true,
                "addresses": [...]
            }
        }

    block_captain:
        true  -> black marking on source map
        false -> green marking on source map
    """
    result = {}

    for block in range(1, 27):
        addresses = []

        if df is not None and not df.empty:
            block_numeric = pd.to_numeric(df["block"], errors="coerce")
            subset = df[
                (block_numeric == block)
                & (df["inside_original_boundary"].astype(bool))
            ]

            seen = set()

            for value in subset["address"].tolist():
                address = str(value).strip()

                if address and address not in seen:
                    seen.add(address)
                    addresses.append(address)

        result[str(block)] = {
            "active": True,
            "block_captain": bool(block_captains.get(block, True)),
            "addresses": addresses,
        }

    Path(output_path).write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )



def save_debug(image, labels, df, output):
    canvas = image.copy()

    edges = np.zeros(labels.shape, dtype=np.uint8)
    edges[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).astype(np.uint8)
    edges[1:, :] |= (labels[1:, :] != labels[:-1, :]).astype(np.uint8)
    canvas[edges > 0] = (255, 0, 255)

    for block, (x, y) in BLOCK_SEEDS.items():
        cv2.circle(canvas, (x, y), 7, (0, 165, 255), -1)

    if df is not None and not df.empty:
        for _, row in df.iterrows():
            x, y = int(row["pixel_x"]), int(row["pixel_y"])
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                if bool(row["inside_original_boundary"]):
                    cv2.circle(canvas, (x, y), 4, (255, 255, 0), -1)
                else:
                    cv2.circle(canvas, (x, y), 5, (255, 255, 0), 1)

    cv2.imwrite(str(output), canvas)


def main():
    print("saratoga_block_addresses_county_v12: APN boundary override + pixel overlap + captain colors")
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--margin-m", type=float, default=150.0)
    ap.add_argument("--csv", type=Path, default=Path("addresses_by_block.csv"))
    ap.add_argument("--json", type=Path, default=Path("blocks.json"))
    ap.add_argument("--debug", type=Path, default=Path("block_debug.png"))
    args = ap.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot open {args.image}")

    calibration = load_calibration(args.calibration)
    labels = segment_blocks(image)
    block_captains = detect_block_captains(image, labels)

    block_areas = {
        block: int((labels == block).sum())
        for block in range(1, 27)
    }
    print(
        "Segmentation check: "
        f"block 21={block_areas[21]} px, "
        f"block 22={block_areas[22]} px"
    )

    pixel_to_geo = fit_homography(calibration, forward=True)
    geo_to_pixel = fit_homography(calibration, forward=False)

    original_poly = original_geo_polygon(pixel_to_geo)
    fetch_poly = buffer_polygon_meters(original_poly, args.margin_m)

    print(f"Fetching county parcel addresses with {args.margin_m:.0f} m safety margin...")
    osm = download_addresses(fetch_poly)

    df = collect(osm, fetch_poly, original_poly, geo_to_pixel, labels)
    df.to_csv(args.csv, index=False)
    write_blocks_json(df, args.json, block_captains)
    save_debug(image, labels, df, args.debug)

    print(f"\nWrote {len(df)} candidate addresses to {args.csv.resolve()}")
    print(f"Wrote block JSON to {args.json.resolve()}")
    print(f"Wrote debug image to {args.debug.resolve()}")

    if not df.empty:
        inside = int(df["inside_original_boundary"].sum())
        outside = len(df) - inside
        assigned = int((df["block"].astype(str) != "").sum())
        inside_assigned = int(
            (
                df["inside_original_boundary"].astype(bool)
                & (df["block"].astype(str) != "")
            ).sum()
        )
        inside_unassigned = int(
            (
                df["inside_original_boundary"].astype(bool)
                & (df["block"].astype(str) == "")
            ).sum()
        )
        outside_assigned = int(
            (
                ~df["inside_original_boundary"].astype(bool)
                & (df["block"].astype(str) != "")
            ).sum()
        )

        print(f"Inside original red boundary: {inside}")
        print(f"Only in expanded safety margin: {outside}")
        print(f"Assigned to blocks 1..26: {assigned}")
        print(f"Inside red boundary and assigned: {inside_assigned}")
        print(f"Inside red boundary but unassigned: {inside_unassigned}")
        print(f"Outside red boundary but assigned: {outside_assigned}")


if __name__ == "__main__":
    main()
