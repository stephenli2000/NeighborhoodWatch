#!/usr/bin/env python3
"""
Fully automatic calibration for the Saratoga screenshot.

NO GUI.
NO clicking.
NO input().
NO manual latitude/longitude entry.

Run:
    python3 make_calibration_auto.py saratoga.png

Outputs:
    calibration.json
    calibration_debug.png

Dependencies:
    pip install opencv-python requests shapely
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import requests
from shapely.geometry import LineString
from shapely.ops import nearest_points, unary_union


EXPECTED_WIDTH = 1903
EXPECTED_HEIGHT = 1142

CENTER_LAT = 37.2670
CENTER_LON = -122.0180
SEARCH_RADIUS_M = 5000

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

USER_AGENT = "saratoga-auto-calibration/3.0"

# Screenshot-specific road-intersection pixels.
LANDMARKS = [
    {"name": "Ljepava / Shadow Mountain", "pixel": (1129, 289),
     "a": "ljepava", "b": "shadow_mountain"},
    {"name": "Miljevich / Shadow Mountain", "pixel": (1128, 458),
     "a": "miljevich", "b": "shadow_mountain"},
    {"name": "Glasgow / Beaumont", "pixel": (1583, 700),
     "a": "glasgow", "b": "beaumont"},
    {"name": "Kilbride / Beaumont", "pixel": (1581, 819),
     "a": "kilbride", "b": "beaumont"},
    {"name": "Edinburgh / Beaumont", "pixel": (1579, 992),
     "a": "edinburgh", "b": "beaumont"},
    {"name": "Ljepava / Regan", "pixel": (527, 458),
     "a": "ljepava", "b": "regan"},
]

ROAD_REGEX = (
    r"^(Ljepava|Miljevich|Glasgow|Kilbride|Edinburgh|"
    r"Shadow Mountain|Beaumont|Regan) "
    r"(Drive|Dr|Avenue|Ave|Lane|Ln)$"
)


def normalize_road_name(name):
    if not name:
        return None

    s = " ".join(name.lower().replace(".", "").split())

    prefixes = {
        "ljepava": "ljepava",
        "miljevich": "miljevich",
        "glasgow": "glasgow",
        "kilbride": "kilbride",
        "edinburgh": "edinburgh",
        "shadow mountain": "shadow_mountain",
        "beaumont": "beaumont",
        "regan": "regan",
    }

    for prefix, key in prefixes.items():
        if s.startswith(prefix + " "):
            return key

    return None


def build_query():
    return f"""
[out:json][timeout:90];
way
  ["highway"]
  ["name"~"{ROAD_REGEX}",i]
  (around:{SEARCH_RADIUS_M},{CENTER_LAT},{CENTER_LON});
out geom;
"""


def fetch_osm_roads():
    last_error = None

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"Querying OpenStreetMap: {endpoint}")
            response = requests.post(
                endpoint,
                data={"data": build_query()},
                headers={"User-Agent": USER_AGENT},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("elements"):
                return data

            last_error = RuntimeError("No OSM road geometry returned")
        except Exception as exc:
            last_error = exc
            print(f"  failed: {exc}")

    raise RuntimeError(f"All Overpass endpoints failed: {last_error}")


def parse_roads(data):
    pieces = {}

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        key = normalize_road_name(tags.get("name"))

        if not key:
            continue

        coords = [
            (float(p["lon"]), float(p["lat"]))
            for p in element.get("geometry", [])
            if "lon" in p and "lat" in p
        ]

        if len(coords) >= 2:
            pieces.setdefault(key, []).append(LineString(coords))

    return {
        key: unary_union(lines)
        for key, lines in pieces.items()
    }


def road_intersection(roads, a, b):
    if a not in roads:
        raise KeyError(f"OSM road not found: {a}")
    if b not in roads:
        raise KeyError(f"OSM road not found: {b}")

    ga = roads[a]
    gb = roads[b]
    inter = ga.intersection(gb)

    if not inter.is_empty:
        if inter.geom_type == "Point":
            p = inter
        elif inter.geom_type == "MultiPoint":
            p = list(inter.geoms)[0]
        else:
            p = inter.representative_point()

        return float(p.x), float(p.y), 0.0

    pa, pb = nearest_points(ga, gb)

    lon = (pa.x + pb.x) / 2.0
    lat = (pa.y + pb.y) / 2.0

    mean_lat = math.radians(lat)
    dx = (pa.x - pb.x) * 111320.0 * math.cos(mean_lat)
    dy = (pa.y - pb.y) * 110540.0
    gap_m = math.hypot(dx, dy)

    return float(lon), float(lat), gap_m


def scale_pixel(pixel, width, height):
    x, y = pixel
    return (
        int(round(x * width / EXPECTED_WIDTH)),
        int(round(y * height / EXPECTED_HEIGHT)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("calibration.json"))
    parser.add_argument("--debug", type=Path, default=Path("calibration_debug.png"))
    parser.add_argument("--max-road-gap", type=float, default=30.0)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot open image: {args.image}")

    height, width = image.shape[:2]

    osm = fetch_osm_roads()
    roads = parse_roads(osm)

    points = []
    debug = image.copy()

    print("\nAutomatic calibration:")
    print("----------------------")

    for landmark in LANDMARKS:
        try:
            lon, lat, gap_m = road_intersection(
                roads, landmark["a"], landmark["b"]
            )
        except Exception as exc:
            print(f"SKIP {landmark['name']}: {exc}")
            continue

        if gap_m > args.max_road_gap:
            print(
                f"SKIP {landmark['name']}: road gap {gap_m:.1f} m "
                f"> {args.max_road_gap:.1f} m"
            )
            continue

        x, y = scale_pixel(landmark["pixel"], width, height)

        point = {
            "pixel_x": x,
            "pixel_y": y,
            "longitude": lon,
            "latitude": lat,
        }
        points.append(point)

        print(
            f"{len(points)}. {landmark['name']}: "
            f"pixel=({x},{y}), lat/lon=({lat:.7f},{lon:.7f})"
        )

        cv2.circle(debug, (x, y), 9, (0, 0, 255), -1)
        cv2.putText(
            debug,
            str(len(points)),
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if len(points) < 4:
        raise SystemExit(
            "\nAutomatic calibration failed: fewer than four usable landmarks."
        )

    result = {
        "image": str(args.image),
        "image_width": width,
        "image_height": height,
        "method": "automatic_osm_intersections",
        "points": points,
    }

    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cv2.imwrite(str(args.debug), debug)

    print()
    print(f"Saved {len(points)} calibration points.")
    print(f"Calibration file: {args.output.resolve()}")
    print(f"Debug image:      {args.debug.resolve()}")


if __name__ == "__main__":
    main()
