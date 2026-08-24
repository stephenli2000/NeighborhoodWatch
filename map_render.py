#!/usr/bin/env python3
"""
Hybrid neighborhood map:
- keeps the Google-style road map background so streets remain easy to find
- overlays controlled, high-contrast BUILDING SHAPES only inside the big red boundary
- overlays large bold STREET NAMES aligned to road direction

Main improvements relative to v14:
1) building footprints are clipped to the red boundary
2) building footprints are filled solid (light gray) with dark outlines
3) the underlying street map remains visible
4) custom street labels are aligned to street direction and use simple collision avoidance

Usage:
    python3 map_v15_hybrid.py \
        --blocks blocks.json \
        --shapefile parcels.shp \
        --output map.png
"""

import argparse
import json
import math
import os
import pickle
import re
import statistics
import sys
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import requests
import shapely.affinity as affinity
from PIL import Image, ImageDraw

from geopy.geocoders import ArcGIS
from shapely.geometry import Point, box, LineString
from shapely.ops import unary_union


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

COUNTY_BUILDING_GEOJSON_URL = (
    "https://data.sccgov.org/resource/e8m9-vdfi.geojson"
)



def build_outer_boundary(gdf_blocks):
    """
    Build one large outer red boundary around all blocks.

    This restores the helper that was accidentally removed in v27.
    """
    union_geom = unary_union(gdf_blocks.geometry.tolist())

    outer = union_geom.buffer(55, join_style=1).buffer(-50, join_style=1)

    if outer.is_empty:
        outer = union_geom.buffer(45, join_style=1)

    outer = outer.buffer(5, join_style=1)

    if outer.is_empty:
        outer = union_geom.convex_hull.buffer(20, join_style=1)

    return outer


def extent_3857_to_4326(xmin, ymin, xmax, ymax):
    """
    Convert a Web Mercator map extent to lon/lat bounds.

    This helper was also accidentally removed when the road-fetch code in v27
    was replaced.
    """
    gdf = gpd.GeoSeries(
        [box(xmin, ymin, xmax, ymax)],
        crs="EPSG:3857",
    ).to_crs(epsg=4326)

    return gdf.total_bounds


def fetch_osm_roads(xmin, ymin, xmax, ymax):
    """
    Fetch actual named road centerlines from OpenStreetMap.

    This replaces the Santa Clara County road-centerline endpoint, which has
    been returning zero features. OSM address coverage may be sparse, but road
    geometry in this neighborhood is much more complete and is exactly what we
    need for label placement/rotation.
    """
    min_lon, min_lat, max_lon, max_lat = extent_3857_to_4326(
        xmin, ymin, xmax, ymax
    )

    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
[out:json][timeout:90];
way
  ["highway"]
  ["name"]
  ({bbox});
out geom tags;
"""

    last_error = None

    print("Downloading OpenStreetMap road centerlines...")

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                timeout=120,
                headers={"User-Agent": "saratoga-neighborhood-map/27"},
            )
            r.raise_for_status()
            data = r.json()

            rows = []

            for element in data.get("elements", []):
                tags = element.get("tags", {}) or {}
                name = str(tags.get("name") or "").strip()

                if not name:
                    continue

                coords = [
                    (float(p["lon"]), float(p["lat"]))
                    for p in element.get("geometry", [])
                    if "lon" in p and "lat" in p
                ]

                if len(coords) < 2:
                    continue

                rows.append(
                    {
                        "name": name,
                        "highway": tags.get("highway", ""),
                        "geometry": LineString(coords),
                    }
                )

            if not rows:
                raise RuntimeError("No named OSM roads returned")

            gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(epsg=3857)

            print(f"  OSM roads: {len(gdf)} named segments")
            return gdf

        except Exception as exc:
            last_error = exc
            print(f"  OSM road endpoint failed: {endpoint}: {exc}")

    print(f"  OSM roads unavailable: {last_error}")
    return None



def fetch_county_buildings(xmin, ymin, xmax, ymax):
    min_lon, min_lat, max_lon, max_lat = extent_3857_to_4326(xmin, ymin, xmax, ymax)

    where = f"within_box(the_geom,{max_lat},{min_lon},{min_lat},{max_lon})"
    params = {"$where": where, "$limit": 8000}

    print("Downloading Santa Clara County building footprints...")

    try:
        r = requests.get(
            COUNTY_BUILDING_GEOJSON_URL,
            params=params,
            timeout=120,
            headers={"User-Agent": "saratoga-neighborhood-map/15"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"  County building-footprint download failed: {exc}")
        return None

    features = data.get("features", [])
    if not features:
        print("  No county building footprints returned.")
        return None

    buildings = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if buildings.empty:
        return None

    print(f"  County buildings: {len(buildings)} footprints")
    return buildings.to_crs(epsg=3857)


def road_label(row):
    for field in ("fulladdr_label_abbrv", "st_name", "lst_name"):
        value = row.get(field)
        if value is not None:
            value = str(value).strip()
            if value and value.lower() != "nan":
                if field == "st_name":
                    typ = str(row.get("st_postyp") or "").strip()
                    if typ and typ.lower() != "nan":
                        value = f"{value} {typ}"
                elif field == "lst_name":
                    typ = str(row.get("lst_type") or "").strip()
                    if typ and typ.lower() != "nan":
                        value = f"{value} {typ}"
                return value
    return ""


def normalize_street_name(name):
    return re.sub(r"\s+", " ", str(name).strip()).upper()


def choose_street_label_features(roads, area_of_interest):
    """
    One label per street, focused around the neighborhood.
    """
    if roads is None or roads.empty:
        return []

    # Keep roads that intersect the neighborhood or are very close to it.
    try:
        near = roads[roads.geometry.intersects(area_of_interest.buffer(120))].copy()
    except Exception:
        near = roads.copy()

    groups = {}

    for _, row in near.iterrows():
        name = road_label(row)
        if not name:
            continue

        key = normalize_street_name(name)
        groups.setdefault(key, {"name": name, "geoms": []})
        groups[key]["geoms"].append(row.geometry)

    result = []
    for item in groups.values():
        geom = unary_union(item["geoms"])

        if geom.geom_type == "MultiLineString":
            geom = max(geom.geoms, key=lambda g: g.length)

        # Ignore tiny segments that do not provide room for a clean label.
        if geom.length < 35:
            continue

        result.append((item["name"], geom))

    # Longer streets first gives better collision-avoidance results.
    result.sort(key=lambda x: x[1].length if x[1] is not None else 0, reverse=True)
    return result


def line_label_candidates(geom):
    """
    Candidate label positions along the line.
    """
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "MultiLineString":
        geom = max(geom.geoms, key=lambda g: g.length)

    if geom.geom_type != "LineString" or geom.length <= 0:
        return []

    candidates = []

    for frac in (0.50, 0.35, 0.65, 0.25, 0.75):
        mid_dist = geom.length * frac
        delta = min(max(geom.length * 0.08, 3.0), 25.0)

        p0 = geom.interpolate(max(0, mid_dist - delta))
        p1 = geom.interpolate(min(geom.length, mid_dist + delta))
        pm = geom.interpolate(mid_dist)

        angle = math.degrees(math.atan2(p1.y - p0.y, p1.x - p0.x))

        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        candidates.append((pm.x, pm.y, angle))

    return candidates


def too_close_to_existing(x, y, existing, min_dist=95):
    for ex, ey in existing:
        if (x - ex) ** 2 + (y - ey) ** 2 < min_dist ** 2:
            return True
    return False


def normalize_road_name_for_match(name):
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", str(name or "")).upper()
    s = re.sub(r"\s+", " ", s).strip()

    suffix_map = {
        " DRIVE": " DR",
        " DR.": " DR",
        " AVENUE": " AVE",
        " AVE.": " AVE",
        " COURT": " CT",
        " CT.": " CT",
        " LANE": " LN",
        " LN.": " LN",
        " ROAD": " RD",
        " RD.": " RD",
        " PLACE": " PL",
        " PL.": " PL",
        " CIRCLE": " CIR",
        " CIR.": " CIR",
        " WAY": " WY",
        " WY.": " WY",
        " BOULEVARD": " BLVD",
        " BLVD.": " BLVD",
    }

    for old, new in suffix_map.items():
        if s.endswith(old):
            s = s[:-len(old)] + new

    return s


def county_road_name_for_match(row):
    # OSM road layer uses "name". Keep older county fields as fallback so the
    # matching helper remains backward-compatible.
    for field in ("name", "st_name", "lst_name", "fulladdr_label_abbrv"):
        value = row.get(field)

        if value is None:
            continue

        value = str(value).strip()
        if not value or value.lower() == "nan":
            continue

        if field == "name":
            return value

        if field == "st_name":
            typ = str(row.get("st_postyp") or "").strip()
            if typ and typ.lower() != "nan":
                value = f"{value} {typ}"

        elif field == "lst_name":
            typ = str(row.get("lst_type") or "").strip()
            if typ and typ.lower() != "nan":
                value = f"{value} {typ}"

        return value

    return ""

def strip_suffix_for_match(name):
    """
    Compare street names even if one source omits/changes the street suffix.
    """
    s = normalize_road_name_for_match(name)
    parts = s.split()
    if parts and parts[-1] in {"DR", "AVE", "CT", "LN", "RD", "PL", "CIR", "WY", "BLVD"}:
        return " ".join(parts[:-1])
    return s


def road_names_match(a, b):
    """
    Match road names robustly across slightly different county/blocks formats.
    """
    na = normalize_road_name_for_match(a)
    nb = normalize_road_name_for_match(b)
    if na == nb:
        return True

    sa = strip_suffix_for_match(a)
    sb = strip_suffix_for_match(b)
    if sa == sb and sa:
        return True

    # Allow one name to be a clean prefix of the other, e.g. "GLASGOW" vs
    # "GLASGOW DR".
    if sa and sb and (sa == nb or sb == na):
        return True

    return False


def infer_angle_from_points(points_3857):
    """
    Fallback angle from the address-point cloud when county road matching fails.
    """
    if len(points_3857) < 2:
        return 0.0

    xs = [p.x for p in points_3857]
    ys = [p.y for p in points_3857]
    x0 = sum(xs) / len(xs)
    y0 = sum(ys) / len(ys)

    sxx = sum((x - x0) ** 2 for x in xs)
    syy = sum((y - y0) ** 2 for y in ys)
    sxy = sum((x - x0) * (y - y0) for x, y in zip(xs, ys))

    if abs(sxy) < 1e-9 and abs(sxx - syy) < 1e-9:
        return 0.0

    angle = 0.5 * math.degrees(math.atan2(2 * sxy, sxx - syy))

    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle




def cluster_street_points(points_lonlat, distance_threshold_m=170.0):
    """
    Split one street's addresses into local clusters so long streets or
    disconnected segments (like Glasgow Dr) get local labels instead of one
    global average that can drift far from the actual street segment.
    """
    pts_gdf = gpd.GeoDataFrame(
        geometry=points_lonlat,
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    pts = list(pts_gdf.geometry)
    n = len(pts)

    if n <= 1:
        return [pts]

    remaining = set(range(n))
    clusters = []

    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        queue = [seed]

        while queue:
            i = queue.pop()
            pi = pts[i]

            neighbors = []
            for j in list(remaining):
                pj = pts[j]
                if pi.distance(pj) <= distance_threshold_m:
                    neighbors.append(j)

            for j in neighbors:
                remaining.remove(j)
                queue.append(j)
                cluster.append(j)

        clusters.append([pts[i] for i in cluster])

    return clusters



def oriented_label_rect(center_x, center_y, angle_deg, length_m, height_m):
    """
    Approximate the visual footprint of a street label as an oriented rectangle.
    """
    rect = box(
        -length_m / 2.0,
        -height_m / 2.0,
        length_m / 2.0,
        height_m / 2.0,
    )
    rect = affinity.rotate(rect, angle_deg, origin=(0, 0), use_radians=False)
    rect = affinity.translate(rect, xoff=center_x, yoff=center_y)
    return rect


def road_corridor_score(center_x, center_y, angle_deg, label_radius, buildings_union, outer_boundary):
    """
    Score how 'street-like' a candidate center is.

    A good street-label location should:
      - have little/no building overlap at the label itself
      - have buildings somewhat nearby on the two sides of the street
      - remain within/near the neighborhood boundary

    Lower is better.
    """
    # Approximate label footprint.
    label_length = max(48.0, label_radius * 2.45)
    label_height = max(12.0, label_radius * 0.78)

    center_rect = oriented_label_rect(
        center_x,
        center_y,
        angle_deg,
        label_length,
        label_height,
    )

    # Side probes just above/below the text, used to detect whether the
    # candidate sits inside a road corridor flanked by houses/buildings.
    probe_offset = label_height * 1.25
    theta = math.radians(angle_deg)
    nx = -math.sin(theta)
    ny = math.cos(theta)

    left_probe = oriented_label_rect(
        center_x + nx * probe_offset,
        center_y + ny * probe_offset,
        angle_deg,
        label_length * 0.92,
        label_height * 0.85,
    )
    right_probe = oriented_label_rect(
        center_x - nx * probe_offset,
        center_y - ny * probe_offset,
        angle_deg,
        label_length * 0.92,
        label_height * 0.85,
    )

    score = 0.0

    if buildings_union is not None and not buildings_union.is_empty:
        try:
            center_overlap = center_rect.intersection(buildings_union).area
            left_overlap = left_probe.intersection(buildings_union).area
            right_overlap = right_probe.intersection(buildings_union).area

            # Very strong penalty for putting the label on top of buildings.
            score += center_overlap * 0.70

            # Reward having some buildings on each side of the corridor.
            if left_overlap > 1.0:
                score -= min(left_overlap, 220.0) * 0.10
            else:
                score += 35.0

            if right_overlap > 1.0:
                score -= min(right_overlap, 220.0) * 0.10
            else:
                score += 35.0

        except Exception:
            if center_rect.intersects(buildings_union):
                score += 10000.0

    if outer_boundary is not None and not outer_boundary.is_empty:
        try:
            if not outer_boundary.buffer(14).contains(center_rect):
                score += 260.0
        except Exception:
            pass

    return score


def choose_best_anchor_positions(cx, cy, angle_deg, span, label_radius, buildings_union, outer_boundary):
    """
    Search for the best ALONG-street anchor positions near a local street cluster.

    This is the key improvement requested by the user: find a straighter/opener
    segment of the street, then place the label there.
    """
    theta = math.radians(angle_deg)
    tx = math.cos(theta)
    ty = math.sin(theta)

    if span < 80.0:
        alongs = [0.0, -0.18 * span, 0.18 * span]
    else:
        alongs = [
            -0.42 * span,
            -0.30 * span,
            -0.18 * span,
            -0.08 * span,
            0.0,
            0.08 * span,
            0.18 * span,
            0.30 * span,
            0.42 * span,
        ]

    scored = []
    for along in alongs:
        base_x = cx + tx * along
        base_y = cy + ty * along

        perp = choose_open_corridor_offset(
            base_x,
            base_y,
            angle_deg,
            label_radius,
            buildings_union,
            outer_boundary,
        )

        nx = -math.sin(theta)
        ny = math.cos(theta)
        x = base_x + nx * perp
        y = base_y + ny * perp

        score = road_corridor_score(
            x,
            y,
            angle_deg,
            label_radius,
            buildings_union,
            outer_boundary,
        )

        # Prefer anchor points that are not too far from the middle, all else equal.
        score += abs(along) * 0.06

        scored.append((score, x, y, along, perp))

    scored.sort(key=lambda item: item[0])

    # Return a few strong anchors.
    return scored[:4]


def choose_open_corridor_offset(center_x, center_y, angle_deg, label_radius, buildings_union, outer_boundary):
    """
    Choose a small perpendicular offset that places the label in an open road
    corridor rather than on top of buildings/parcels.

    Important design choice:
      Prefer SMALL offsets. We do not want labels drifting 1-2 parcels away.
    """
    theta = math.radians(angle_deg)
    nx = -math.sin(theta)
    ny = math.cos(theta)

    offsets = [0.0, 8.0, -8.0, 14.0, -14.0, 20.0, -20.0, 28.0, -28.0, 38.0, -38.0]

    best = 0.0
    best_score = None

    for off in offsets:
        x = center_x + nx * off
        y = center_y + ny * off

        score = road_corridor_score(
            x,
            y,
            angle_deg,
            label_radius,
            buildings_union,
            outer_boundary,
        )

        # Strong bias toward staying near the centerline / road corridor.
        score += abs(off) * 1.2

        if best_score is None or score < best_score:
            best_score = score
            best = off

    return best


def infer_angle_and_center(points_3857):
    """
    PCA-style local street direction from clustered address points.
    """
    if not points_3857:
        return 0.0, 0.0, 0.0, 80.0

    xs = [p.x for p in points_3857]
    ys = [p.y for p in points_3857]

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    if len(points_3857) == 1:
        return cx, cy, 0.0, 80.0

    sxx = sum((x - cx) ** 2 for x in xs)
    syy = sum((y - cy) ** 2 for y in ys)
    sxy = sum((x - cx) * (y - cy) for x, y in zip(xs, ys))

    angle = 0.5 * math.degrees(math.atan2(2 * sxy, sxx - syy))

    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    theta = math.radians(angle)
    tx = math.cos(theta)
    ty = math.sin(theta)
    projected = [(p.x - cx) * tx + (p.y - cy) * ty for p in points_3857]
    span = max(max(projected) - min(projected), 80.0)

    return cx, cy, angle, span


def local_straight_segments(points_3857, max_segments=3):
    """
    Build at most 3 meaningful straight-ish segments for one street cluster.

    Updated policy:
      - strongly prefer the LONGEST sufficiently-straight segment
      - only fall back to shorter segments if the longer one is not usable
      - avoid many tiny/random segments
      - explicitly avoid dead-end influence by testing trimmed groups
      - split only at large gaps, never into many pieces

    Returns tuples:
        (cx, cy, angle, span, straightness)
    """
    if not points_3857:
        return []

    gcx, gcy, gangle, gspan = infer_angle_and_center(points_3857)

    if len(points_3857) < 4 or gspan < 70.0:
        return [(gcx, gcy, gangle, gspan, 0.0)]

    theta = math.radians(gangle)
    tx = math.cos(theta)
    ty = math.sin(theta)

    def along_value(p):
        return (p.x - gcx) * tx + (p.y - gcy) * ty

    ordered = sorted(points_3857, key=along_value)
    alongs = [along_value(p) for p in ordered]
    n = len(ordered)

    MIN_POINTS = 4
    MIN_SPAN = 60.0

    def evaluate_group(group):
        if len(group) < MIN_POINTS:
            return None

        cx, cy, angle, span = infer_angle_and_center(group)
        if span < MIN_SPAN:
            return None

        xs = [p.x for p in group]
        ys = [p.y for p in group]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)

        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        tr = sxx + syy
        det = sxx * syy - sxy * sxy
        disc = max(tr * tr - 4.0 * det, 0.0) ** 0.5
        lam1 = max((tr + disc) / 2.0, 1e-9)
        lam2 = max((tr - disc) / 2.0, 1e-9)

        linearity_ratio = lam1 / lam2

        a = math.radians(angle)
        nx = -math.sin(a)
        ny = math.cos(a)

        residuals = [
            abs((p.x - cx) * nx + (p.y - cy) * ny)
            for p in group
        ]
        residual = sum(residuals) / len(residuals)
        straightness = residual / max(span, 1.0)

        # Reject weak / round / dead-end-like groups.
        if linearity_ratio < 3.2 and straightness > 0.10:
            return None

        return (cx, cy, angle, span, straightness, linearity_ratio, len(group))

    candidates = []

    def add_candidate(group):
        e = evaluate_group(group)
        if e is not None:
            candidates.append(e)

    # Whole cluster.
    add_candidate(ordered)

    # Trimmed versions: helps ignore cul-de-sacs / dead-end tails.
    trims = [
        (1, 0), (0, 1),
        (1, 1),
        (2, 0), (0, 2),
        (2, 1), (1, 2),
    ]
    for left_trim, right_trim in trims:
        if n - left_trim - right_trim >= MIN_POINTS:
            add_candidate(ordered[left_trim:n - right_trim])

    # Split only at large gaps along the main axis; max 3 pieces.
    gaps = []
    for i in range(n - 1):
        gap = alongs[i + 1] - alongs[i]
        gaps.append((gap, i))

    total_span = max(alongs) - min(alongs)
    large_gap_threshold = max(35.0, 0.18 * total_span)

    big_gaps = [(g, i) for g, i in gaps if g >= large_gap_threshold]
    big_gaps.sort(reverse=True)

    if big_gaps:
        cut_indices = sorted(i for _, i in big_gaps[: max_segments - 1])

        boundaries = [0]
        for i in cut_indices:
            boundaries.append(i + 1)
        boundaries.append(n)

        for a, b in zip(boundaries[:-1], boundaries[1:]):
            piece = ordered[a:b]
            if len(piece) >= MIN_POINTS:
                add_candidate(piece)

    if not candidates:
        return [(gcx, gcy, gangle, gspan, 0.0)]

    # Keep only sufficiently-straight candidates, then strongly prefer the
    # longest one. If nothing passes the strict filter, fall back to the best
    # available candidates.
    strict = [
        item for item in candidates
        if item[4] <= 0.085 and item[5] >= 3.0
    ]
    pool = strict if strict else candidates

    pool.sort(
        key=lambda item: (
            -item[3],      # LONGEST segment first
            item[4],       # then straighter
            -item[5],      # then more linear
            -item[6],      # then more points
        )
    )

    chosen = []
    for cx, cy, angle, span, straightness, linearity_ratio, count in pool:
        duplicate = False
        for ccx, ccy, cangle, cspan, cstraight in chosen:
            angle_diff = abs(angle - cangle)
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            if math.hypot(cx - ccx, cy - ccy) < 55.0 and angle_diff < 12.0:
                duplicate = True
                break

        if duplicate:
            continue

        chosen.append((cx, cy, angle, span, straightness))
        if len(chosen) >= max_segments:
            break

    if not chosen:
        chosen = [(gcx, gcy, gangle, gspan, 0.0)]

    return chosen


def street_label_candidates_for_cluster(points_3857, street_name, roads, buildings_union, outer_boundary):
    """
    Generate candidates for one local cluster of a street.

    Priority:
      1) If road centerlines are available, use them.
      2) Otherwise, split the cluster into locally straight segments and place
         the label on one of those segments.
    """
    # If county roads are available, preserve the road-based behavior.
    if roads is not None and not roads.empty:
        pts_gdf = gpd.GeoDataFrame(geometry=points_3857, crs="EPSG:3857").to_crs(epsg=4326)
        pts_lonlat = list(pts_gdf.geometry)
        return street_label_candidates_from_addresses(
            street_name,
            pts_lonlat,
            roads,
        )

    candidates = []

    for sx, sy, angle, span, residual in local_straight_segments(
        points_3857,
        max_segments=3,
    ):
        theta = math.radians(angle)
        tx = math.cos(theta)
        ty = math.sin(theta)
        nx = -math.sin(theta)
        ny = math.cos(theta)

        label_radius = max(18.0, span * 0.08)

        anchors = choose_best_anchor_positions(
            sx,
            sy,
            angle,
            span,
            label_radius,
            buildings_union,
            outer_boundary,
        )

        # Around each good anchor, keep alternatives very close to the street.
        for anchor_score, ax, ay, _, _ in anchors:
            for perp in (0.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0):
                for along in (0.0, -16.0, 16.0, -28.0, 28.0, -42.0, 42.0):
                    x = ax + tx * along + nx * perp
                    y = ay + ty * along + ny * perp
                    candidates.append((x, y, angle))

    # Final minimal fallback.
    if not candidates:
        cx, cy, angle, span = infer_angle_and_center(points_3857)
        theta = math.radians(angle)
        tx = math.cos(theta)
        ty = math.sin(theta)
        nx = -math.sin(theta)
        ny = math.cos(theta)
        for perp in (0.0, 4.0, -4.0, 8.0, -8.0):
            for along in (0.0, -14.0, 14.0, -24.0, 24.0):
                candidates.append((cx + tx * along + nx * perp, cy + ty * along + ny * perp, angle))

    return candidates

def street_label_candidates_from_addresses(street_name, points_lonlat, roads):
    """
    Generate label candidates from TRUE road geometry when available.

    With OSM roads:
      - collect every road segment whose name matches
      - merge them
      - choose the connected piece nearest to the address cluster
      - sample several positions along that road
      - compute rotation from a local tangent, not the whole street

    This fixes cases like Glasgow Dr, Beaumont Ave and Shadow Mountain Dr where
    address-point PCA can be parallel to the road but shifted into parcels, or
    can get the direction wrong when address points are sparse.
    """
    pts_gdf = gpd.GeoDataFrame(
        geometry=points_lonlat,
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    pts = list(pts_gdf.geometry)
    avg_x = sum(p.x for p in pts) / len(pts)
    avg_y = sum(p.y for p in pts) / len(pts)
    anchor = Point(avg_x, avg_y)

    matching_geoms = []

    if roads is not None and not roads.empty:
        for _, row in roads.iterrows():
            road_name = county_road_name_for_match(row)

            if not road_names_match(road_name, street_name):
                continue

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "MultiLineString":
                matching_geoms.extend(
                    g for g in geom.geoms
                    if not g.is_empty and g.length > 0
                )
            elif geom.length > 0:
                matching_geoms.append(geom)

    if matching_geoms:
        merged = unary_union(matching_geoms)

        if merged.geom_type == "LineString":
            pieces = [merged]
        elif merged.geom_type == "MultiLineString":
            pieces = list(merged.geoms)
        else:
            pieces = matching_geoms

        # Prefer the connected road piece closest to the address cluster, but
        # do not use extremely short dead-end stubs when a longer piece exists.
        pieces = [g for g in pieces if g.length >= 35.0] or pieces

        geom = min(
            pieces,
            key=lambda g: (
                anchor.distance(g),
                -g.length,
            ),
        )

        # Anchor near the median address projection onto the REAL road.
        projections = sorted(geom.project(p) for p in pts)
        median_d = projections[len(projections) // 2]

        # Candidate positions only ON the road centerline. We slide ALONG the
        # street to avoid collisions; no large perpendicular drift.
        distances = [median_d]

        for off in (-45, 45, -90, 90, -140, 140, -200, 200):
            distances.append(
                max(0.0, min(float(geom.length), median_d + off))
            )

        for frac in (0.18, 0.30, 0.42, 0.55, 0.68, 0.80):
            distances.append(float(geom.length) * frac)

        dedup = []
        for d in distances:
            if not any(abs(d - existing) < 12.0 for existing in dedup):
                dedup.append(d)

        candidates = []

        for d in dedup:
            pm = geom.interpolate(d)

            # Local tangent: short window around the candidate so curved
            # streets get the angle of that local straight-ish segment.
            delta = min(max(geom.length * 0.035, 8.0), 28.0)
            p0 = geom.interpolate(max(0.0, d - delta))
            p1 = geom.interpolate(min(float(geom.length), d + delta))

            angle = math.degrees(
                math.atan2(
                    p1.y - p0.y,
                    p1.x - p0.x,
                )
            )

            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # EXACTLY on road first; only very tiny visual offsets as fallback.
            theta = math.radians(angle)
            nx = -math.sin(theta)
            ny = math.cos(theta)

            for perp in (0.0, 3.0, -3.0, 6.0, -6.0):
                candidates.append(
                    (
                        pm.x + nx * perp,
                        pm.y + ny * perp,
                        angle,
                    )
                )

        return candidates

    # Fallback if road geometry cannot be fetched.
    angle = infer_angle_from_points(pts)
    theta = math.radians(angle)
    tx = math.cos(theta)
    ty = math.sin(theta)

    projected = [
        (p.x - avg_x) * tx + (p.y - avg_y) * ty
        for p in pts
    ]
    span = (
        max(max(projected) - min(projected), 80.0)
        if projected
        else 80.0
    )

    candidates = []

    for along in (0.0, -0.15 * span, 0.15 * span, -0.30 * span, 0.30 * span):
        candidates.append(
            (
                avg_x + tx * along,
                avg_y + ty * along,
                angle,
            )
        )

    return candidates


def estimate_label_radius_data(ax, text, font_size):
    """
    Approximate a text label's footprint radius in map data units.
    This is good enough for collision avoidance.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    fig = ax.figure
    width_in = fig.get_size_inches()[0] * ax.get_position().width
    height_in = fig.get_size_inches()[1] * ax.get_position().height

    if width_in <= 0 or height_in <= 0:
        return 50.0

    data_per_in_x = abs(x1 - x0) / width_in
    data_per_in_y = abs(y1 - y0) / height_in
    data_per_in = max(data_per_in_x, data_per_in_y)

    # Rough typographic approximation.
    text_width_in = max(1, len(text)) * font_size * 0.62 / 72.0
    text_height_in = font_size * 1.15 / 72.0

    width_data = text_width_in * data_per_in
    height_data = text_height_in * data_per_in

    return 0.55 * ((width_data ** 2 + height_data ** 2) ** 0.5)


def collides_with_occupied(x, y, radius, occupied):
    """
    Check if a candidate street label collides with previously placed items.
    occupied is a list of (x, y, radius).
    """
    for ox, oy, orad in occupied:
        if (x - ox) ** 2 + (y - oy) ** 2 < (radius + orad) ** 2:
            return True
    return False



def place_block_number_position(geom, centroid, occupied, radius):
    """
    Choose a block-number position after street labels are placed.

    Street names get priority. Block numbers are then nudged within the block to
    avoid sitting on top of those street labels.
    """
    try:
        rep = geom.representative_point()
    except Exception:
        rep = centroid

    candidates = [
        (centroid.x, centroid.y),
        (rep.x, rep.y),
    ]

    for dist in (12, 24, 36, 48, 60):
        for dx, dy in [
            (dist, 0), (-dist, 0), (0, dist), (0, -dist),
            (dist, dist), (dist, -dist), (-dist, dist), (-dist, -dist),
        ]:
            candidates.append((centroid.x + dx, centroid.y + dy))
            candidates.append((rep.x + dx, rep.y + dy))

    best = (rep.x, rep.y)
    best_penalty = None

    for x, y in candidates:
        pt = Point(x, y)
        penalty = 0.0

        try:
            if not geom.buffer(-1.5).contains(pt):
                penalty += 500.0
        except Exception:
            if not geom.contains(pt):
                penalty += 500.0

        for ox, oy, orad in occupied:
            dist = math.hypot(x - ox, y - oy)
            required = radius + orad
            if dist < required:
                penalty += (required - dist) * 25.0

        if best_penalty is None or penalty < best_penalty:
            best_penalty = penalty
            best = (x, y)

        if penalty == 0:
            break

    return best
def candidate_score(
    x,
    y,
    radius,
    occupied,
    buildings_union,
    outer_boundary,
    priority_index,
):
    """
    Score a street-label candidate.

    Lower is better.

    Goals:
      - stay close to the preferred road-centered candidates
      - avoid overlapping previous street labels and block-number circles
      - prefer visible road corridors rather than parcel interiors
      - stay within/near the neighborhood boundary
    """
    score = 0.0

    # Respect candidate priority from the generator (best anchors first).
    score += priority_index * 1.6

    # Penalize collisions with previously placed labels / block circles.
    for ox, oy, orad in occupied:
        dist = math.hypot(x - ox, y - oy)
        required = radius + orad

        if dist < required:
            score += (required - dist) * 22.0
        else:
            score += max(0.0, (required * 1.08 - dist)) * 0.15

    # Prefer points that behave like road corridors.
    score += road_corridor_score(
        x,
        y,
        0.0,  # overwritten below if unavailable; see note
        radius,
        buildings_union,
        outer_boundary,
    ) * 0.0

    # The oriented corridor score needs the label angle, but the current
    # interface does not pass it. So here we preserve compatibility and rely on
    # center/building penalties below. The angle-aware road-corridor selection
    # already happens earlier during anchor generation.
    label_geom = Point(x, y).buffer(radius * 0.50)

    if buildings_union is not None and not buildings_union.is_empty:
        try:
            overlap_area = label_geom.intersection(buildings_union).area
            score += overlap_area * 0.10
        except Exception:
            if label_geom.intersects(buildings_union):
                score += 5000.0

    if outer_boundary is not None and not outer_boundary.is_empty:
        try:
            if not outer_boundary.buffer(20).contains(label_geom):
                score += 180.0
        except Exception:
            pass

    return score



def geometry_to_pixel_mask(geom, width, height, extent, origin="upper"):
    """
    Rasterize a shapely geometry into a PIL mask aligned to an image extent.
    """
    x0, x1, y0, y1 = extent

    def to_px(x, y):
        px = (x - x0) / (x1 - x0) * (width - 1)
        if origin == "upper":
            py = (y1 - y) / (y1 - y0) * (height - 1)
        else:
            py = (y - y0) / (y1 - y0) * (height - 1)
        return (px, py)

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    def draw_polygon(poly, fill):
        ext = [to_px(x, y) for x, y in poly.exterior.coords]
        if len(ext) >= 3:
            draw.polygon(ext, fill=fill)
        for ring in poly.interiors:
            pts = [to_px(x, y) for x, y in ring.coords]
            if len(pts) >= 3:
                draw.polygon(pts, fill=0)

    if geom.geom_type == "Polygon":
        draw_polygon(geom, 255)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            draw_polygon(poly, 255)

    return np.asarray(img, dtype=float) / 255.0


def grayscale_basemap_outside_boundary(ax, boundary_geom, extent=None):
    """
    Directly rewrite basemap RGB pixels outside the big red boundary.

    Important: the Contextily basemap image usually extends beyond the axes
    view to whole map tiles. Therefore the pixel mask must use the basemap
    artist's ACTUAL extent, not ax.get_xlim()/view_bounds. Otherwise the mask
    is shifted and strips near the top/left/right can remain colored.
    """
    if not ax.images:
        return

    base_artist = ax.images[-1]
    arr = np.asarray(base_artist.get_array())

    if arr.ndim != 3 or arr.shape[2] < 3:
        return

    original_dtype = arr.dtype
    work = arr.astype(float, copy=True)

    # Pull grayscale a little underneath the thick red line so there is no
    # colored fringe at the vector/raster boundary.
    gray_boundary = boundary_geom.buffer(-3.0)
    if gray_boundary.is_empty:
        gray_boundary = boundary_geom

    # Use the exact image extent returned by Contextily.
    image_extent = tuple(float(v) for v in base_artist.get_extent())
    x0, x1, y0, y1 = image_extent
    image_box = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    outside_geom = image_box.difference(gray_boundary)
    if outside_geom.is_empty:
        return

    h, w = work.shape[:2]
    outside_pixels = geometry_to_pixel_mask(
        outside_geom,
        width=w,
        height=h,
        extent=image_extent,
        origin=base_artist.origin,
    ) > 0.5

    rgb = work[..., :3]
    luminance = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )

    rgb[outside_pixels, 0] = luminance[outside_pixels]
    rgb[outside_pixels, 1] = luminance[outside_pixels]
    rgb[outside_pixels, 2] = luminance[outside_pixels]

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        work = np.clip(work, info.min, info.max).astype(original_dtype)
    else:
        work = work.astype(original_dtype)

    base_artist.set_data(work)


def main():
    parser = argparse.ArgumentParser(
        description="Render neighborhood map quickly from process_blocks.py cache."
    )
    parser.add_argument(
        "cache",
        nargs="?",
        default="map_cache.pkl",
        help="Intermediate cache produced by process_blocks.py",
    )
    parser.add_argument("-o", "--output", default="map.png")
    parser.add_argument(
        "--roads-cache",
        default=None,
        help="Optional persistent OSM road cache (default: roads_cache.pkl beside map cache)",
    )
    parser.add_argument("--street-font-size", type=float, default=9.5)
    parser.add_argument("--house-number-font-size", type=float, default=3.0)
    parser.add_argument("--block-number-font-size", type=float, default=8.0)
    parser.add_argument(
        "--outer-boundary-linewidth",
        type=float,
        default=4.0,
        help="Line width for the outer red boundary",
    )
    parser.add_argument(
        "--no-house-numbers",
        action="store_true",
        help="Do not render house numbers on parcels",
    )
    parser.add_argument(
        "--gray-outside-boundary",
        action="store_true",
        help="Gray out the map area outside the big red boundary",
    )
    parser.add_argument("--building-fill", default="#CFCFCF")
    parser.add_argument("--building-edge", default="#4A4A4A")
    parser.add_argument("--building-linewidth", type=float, default=0.9)
    parser.add_argument(
        "--refresh-roads",
        action="store_true",
        help="Try downloading OSM road centerlines again for this render",
    )
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        raise SystemExit(
            f"Missing cache: {args.cache}\n"
            "Run process_blocks.py first."
        )

    print(f"Loading intermediate cache: {args.cache}")
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)

    gdf_blocks = cache["blocks"]
    outer_boundary = cache["outer_boundary"]
    local_parcels = cache["local_parcels"]
    buildings = cache.get("buildings")
    roads = cache.get("roads")

    cache_path = Path(args.cache)
    roads_cache_path = (
        Path(args.roads_cache)
        if args.roads_cache
        else cache_path.with_name("roads_cache.pkl")
    )

    if (roads is None or roads.empty) and roads_cache_path.exists():
        try:
            with roads_cache_path.open("rb") as f:
                persistent_roads = pickle.load(f)

            if persistent_roads is not None and not persistent_roads.empty:
                roads = persistent_roads
                cache["roads"] = roads
                print(
                    f"Loaded {len(roads)} road segments from persistent road cache."
                )
        except Exception as exc:
            print(f"Could not load persistent road cache: {exc}")

    parcel_number_labels = cache["parcel_number_labels"]
    street_label_points = cache["street_label_points"]
    view_xmin, view_ymin, view_xmax, view_ymax = cache["view_bounds"]

    if args.refresh_roads:
        refreshed = fetch_osm_roads(view_xmin, view_ymin, view_xmax, view_ymax)

        if refreshed is not None and not refreshed.empty:
            roads = refreshed
            cache["roads"] = refreshed

            with open(args.cache, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

            try:
                with roads_cache_path.open("wb") as f:
                    pickle.dump(refreshed, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                print(f"Could not update persistent road cache: {exc}")

            print(
                f"Updated road cache with {len(refreshed)} OSM road segments."
            )
        elif roads is not None and not roads.empty:
            print(
                "OSM refresh failed — keeping "
                f"{len(roads)} existing cached road segments."
            )
        else:
            print(
                "OSM refresh failed — no existing road cache is available; "
                "using address-point fallback."
            )

    if roads is None or roads.empty:
        print("Street-label mode: address-point fallback")
    else:
        print(f"Street-label mode: cached road centerlines ({len(roads)} segments)")

    buildings_union = None
    if buildings is not None and not buildings.empty:
        try:
            buildings_union = unary_union(list(buildings.geometry))
        except Exception:
            buildings_union = None

    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300)
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)

    google_url = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
    cx.add_basemap(
        ax,
        source=google_url,
        zoom=18,
        interpolation="bilinear",
        zorder=0,
    )

    if buildings is not None and not buildings.empty:
        buildings.plot(
            ax=ax,
            facecolor=args.building_fill,
            edgecolor=args.building_edge,
            linewidth=args.building_linewidth,
            alpha=0.95,
            zorder=1.8,
        )

    if local_parcels is not None and not local_parcels.empty:
        local_parcels.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#9A9A9A",
            linewidth=0.15,
            alpha=0.10,
            zorder=1.9,
        )

    if args.gray_outside_boundary:
        grayscale_basemap_outside_boundary(
            ax=ax,
            boundary_geom=outer_boundary,
            extent=(view_xmin, view_xmax, view_ymin, view_ymax),
        )

    gpd.GeoSeries([outer_boundary], crs="EPSG:3857").plot(
        ax=ax,
        facecolor="none",
        edgecolor="#D50000",
        linewidth=args.outer_boundary_linewidth,
        zorder=3,
    )

    # Block outlines first. Street labels get placement priority over block numbers.
    for _, row in gdf_blocks.iterrows():
        has_captain = bool(row["block_captain"])
        edge_color = "#111111" if has_captain else "#00C853"
        fill_color = "#FFFFFF" if has_captain else "#00E676"

        gpd.GeoSeries([row["geometry"]], crs="EPSG:3857").plot(
            ax=ax,
            facecolor=fill_color,
            edgecolor="none",
            alpha=0.08,
            zorder=4,
        )
        gpd.GeoSeries([row["geometry"]], crs="EPSG:3857").plot(
            ax=ax,
            facecolor="none",
            edgecolor=edge_color,
            linewidth=2.25,
            zorder=5,
        )

    if not args.no_house_numbers:
        seen = set()
        for label in parcel_number_labels:
            key = (label["house_number"], round(label["x"], 1), round(label["y"], 1))
            if key in seen:
                continue
            seen.add(key)
            ax.text(
                label["x"], label["y"], label["house_number"],
                fontsize=args.house_number_font_size,
                color="#111111",
                ha="center", va="center",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.04",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.55,
                ),
            )

    # Street labels first; block numbers are placed afterward.
    occupied_labels = []
    street_items = sorted(
        street_label_points.items(),
        key=lambda kv: (len(kv[1]), len(kv[0])),
        reverse=True,
    )

    for street_name, points_lonlat in street_items:
        clusters = cluster_street_points(points_lonlat, distance_threshold_m=190.0)
        all_candidates = []
        for cluster_pts_3857 in clusters:
            all_candidates.extend(
                street_label_candidates_for_cluster(
                    cluster_pts_3857,
                    street_name,
                    roads,
                    buildings_union,
                    outer_boundary,
                )
            )

        label_radius = estimate_label_radius_data(ax, street_name, args.street_font_size)
        best = None

        # Strongly prefer the earliest candidates, which come from the longest
        # sufficiently-straight segment first. Only fall back if that segment
        # actually collides with an already-placed street label.
        for x, y, angle in all_candidates:
            if not collides_with_occupied(x, y, label_radius, occupied_labels):
                best = (x, y, angle)
                break

        # If every candidate collides, then use the old scoring fallback.
        if best is None:
            best_score = None
            for idx, (x, y, angle) in enumerate(all_candidates):
                score = candidate_score(
                    x=x,
                    y=y,
                    radius=label_radius,
                    occupied=occupied_labels,
                    buildings_union=buildings_union,
                    outer_boundary=outer_boundary,
                    priority_index=idx,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best = (x, y, angle)

        if best is None:
            continue

        x, y, angle = best
        print(f"Street label: {street_name:<24} angle={angle:6.1f} deg")

        txt = ax.text(
            x, y, street_name,
            fontsize=args.street_font_size,
            fontweight="bold",
            color="#111111",
            ha="center", va="center",
            rotation=angle,
            rotation_mode="anchor",
            zorder=20,
            clip_on=True,
        )
        txt.set_path_effects([
            pe.Stroke(
                linewidth=max(3.5, args.street_font_size * 0.34),
                foreground="white",
            ),
            pe.Normal(),
        ])
        occupied_labels.append((x, y, label_radius))

    # Block numbers after street labels.
    for _, row in gdf_blocks.iterrows():
        block_id = str(row["block_id"])
        centroid = Point(float(row["centroid_x"]), float(row["centroid_y"]))
        block_radius = estimate_label_radius_data(
            ax, block_id, args.block_number_font_size
        ) * 0.70
        x, y = place_block_number_position(
            row["geometry"], centroid, occupied_labels, block_radius
        )
        ax.text(
            x, y, block_id,
            fontsize=args.block_number_font_size,
            fontweight="bold",
            color="black",
            ha="center", va="center",
            zorder=21,
            bbox=dict(
                boxstyle="circle,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.90,
            ),
        )
        occupied_labels.append((x, y, block_radius))

    ax.set_title(
        "Saratoga Neighborhood Watch Map",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_axis_off()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Success! Map saved as '{args.output}'.")


if __name__ == "__main__":
    main()
