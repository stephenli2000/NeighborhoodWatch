#!/usr/bin/env python3
"""
STEP 1: Slow preprocessing / caching.

This script does the expensive work once:
  - loads the county parcel shapefile
  - geocodes every address in blocks.json
  - matches addresses to parcels
  - builds the 26 block polygons and outer red boundary
  - prepares house-number and street-name source points
  - downloads/clips county building footprints
  - optionally downloads OSM road centerlines if available

It writes one intermediate pickle cache. The renderer can then be iterated
without repeating the slow block/address processing.

Usage:
    python3 process_blocks.py \
      -b blocks.json \
      -s data/parcels.shp \
      -o map_cache.pkl

The pickle file is intended to be loaded only by render_map.py in the same
trusted project/environment.
"""

import argparse
import json
import os
import pickle
import re
import statistics
from pathlib import Path

import geopandas as gpd
import requests
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
    union_geom = unary_union(gdf_blocks.geometry.tolist())
    outer = union_geom.buffer(55, join_style=1).buffer(-50, join_style=1)
    if outer.is_empty:
        outer = union_geom.buffer(45, join_style=1)
    outer = outer.buffer(5, join_style=1)
    if outer.is_empty:
        outer = union_geom.convex_hull.buffer(20, join_style=1)
    return outer


def extent_3857_to_4326(xmin, ymin, xmax, ymax):
    gdf = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs="EPSG:3857").to_crs(epsg=4326)
    return gdf.total_bounds


def fetch_osm_roads(xmin, ymin, xmax, ymax):
    min_lon, min_lat, max_lon, max_lat = extent_3857_to_4326(xmin, ymin, xmax, ymax)
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
    print("Downloading OpenStreetMap road centerlines (optional cache)...")

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                timeout=120,
                headers={"User-Agent": "saratoga-neighborhood-map-cache/1"},
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
                if len(coords) >= 2:
                    rows.append({
                        "name": name,
                        "highway": tags.get("highway", ""),
                        "geometry": LineString(coords),
                    })

            if not rows:
                raise RuntimeError("No named OSM roads returned")

            roads = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(epsg=3857)
            print(f"  Cached {len(roads)} OSM road segments")
            return roads
        except Exception as exc:
            last_error = exc
            print(f"  OSM road endpoint failed: {endpoint}: {exc}")

    print(f"  OSM roads not cached: {last_error}")
    return None


def fetch_county_buildings(xmin, ymin, xmax, ymax):
    min_lon, min_lat, max_lon, max_lat = extent_3857_to_4326(xmin, ymin, xmax, ymax)
    where = f"within_box(the_geom,{max_lat},{min_lon},{min_lat},{max_lon})"
    params = {"$where": where, "$limit": 8000}

    print("Downloading Santa Clara County building footprints...")
    r = requests.get(
        COUNTY_BUILDING_GEOJSON_URL,
        params=params,
        timeout=120,
        headers={"User-Agent": "saratoga-neighborhood-map-cache/1"},
    )
    r.raise_for_status()
    data = r.json()

    buildings = gpd.GeoDataFrame.from_features(data.get("features", []), crs="EPSG:4326")
    if buildings.empty:
        return None
    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings = buildings[
        buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    if buildings.empty:
        return None
    buildings = buildings.to_crs(epsg=3857)
    print(f"  County buildings: {len(buildings)} footprints")
    return buildings


def main():
    ap = argparse.ArgumentParser(description="Preprocess neighborhood map data into a fast rendering cache.")
    ap.add_argument("-b", "--blocks", default="blocks.json", help="Path to blocks.json")
    ap.add_argument("-s", "--shapefile", required=True, help="County parcel shapefile")
    ap.add_argument("-o", "--output", default="map_cache.pkl", help="Intermediate cache file")
    ap.add_argument(
        "--skip-roads",
        action="store_true",
        help="Do not attempt OSM road download while building the cache",
    )
    args = ap.parse_args()

    if not os.path.exists(args.blocks):
        raise SystemExit(f"Missing block file: {args.blocks}")
    if not os.path.exists(args.shapefile):
        raise SystemExit(f"Missing parcel shapefile: {args.shapefile}")

    with open(args.blocks, "r", encoding="utf-8") as f:
        block_data = json.load(f)

    print(f"Loading county parcels from {args.shapefile}...")
    parcels = gpd.read_file(args.shapefile).to_crs(epsg=3857)

    geolocator = ArcGIS(user_agent="neighborhood_watch_cache_builder")
    block_polygons = []
    parcel_number_labels = []
    matched_parcel_geometries = []
    street_label_points = {}

    for block_id, info in block_data.items():
        print(f"\nProcessing Block {block_id}...")
        coords = []
        house_numbers = []

        for addr in info.get("addresses", []):
            loc = geolocator.geocode(addr)
            if not loc or loc.raw.get("score", 0) < 95:
                print(f"  - Geocode skipped/low confidence: {addr}")
                continue

            point = Point(loc.longitude, loc.latitude)
            coords.append(point)

            m = re.match(r"^\s*([0-9]+[A-Za-z0-9\-]*)\b", str(addr))
            house_numbers.append(m.group(1) if m else "")

            street_match = re.match(
                r"^\s*[0-9]+[A-Za-z0-9\-]*\s+(.+?),\s*Saratoga",
                str(addr),
                flags=re.IGNORECASE,
            )
            if street_match:
                street_name = street_match.group(1).strip()
                street_label_points.setdefault(street_name, []).append(point)

            print(f"  + Added: {addr}")

        if not coords:
            continue

        gdf_pts = gpd.GeoDataFrame(
            {"house_number": house_numbers},
            geometry=coords,
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        pts_buf = gdf_pts.copy()
        pts_buf.geometry = pts_buf.geometry.buffer(3)

        matched = gpd.sjoin(
            parcels,
            pts_buf[["house_number", "geometry"]],
            how="inner",
            predicate="intersects",
        )

        for _, parcel_row in matched.iterrows():
            number = str(parcel_row.get("house_number", "") or "").strip()
            if number:
                rp = parcel_row.geometry.representative_point()
                parcel_number_labels.append({
                    "house_number": number,
                    "x": float(rp.x),
                    "y": float(rp.y),
                })

        matched = matched.drop_duplicates(subset=[parcels.geometry.name])
        if matched.empty:
            continue

        matched_parcel_geometries.extend(list(matched.geometry))
        merged_geom = unary_union(list(matched.geometry))

        smooth_boundary = merged_geom.buffer(22, join_style=1).buffer(-25, join_style=1)
        if smooth_boundary.is_empty:
            smooth_boundary = merged_geom.buffer(20, join_style=1).buffer(-21, join_style=1)
        if smooth_boundary.is_empty:
            smooth_boundary = merged_geom.convex_hull

        block_captain = info.get("block_captain")
        if block_captain is None:
            block_captain = info.get("active", True)

        block_polygons.append({
            "block_id": str(block_id),
            "geometry": smooth_boundary,
            "block_captain": bool(block_captain),
            "centroid_x": float(smooth_boundary.centroid.x),
            "centroid_y": float(smooth_boundary.centroid.y),
        })

    if not block_polygons:
        raise SystemExit("No blocks were successfully processed.")

    gdf_blocks = gpd.GeoDataFrame(block_polygons, geometry="geometry", crs="EPSG:3857")
    outer_boundary = build_outer_boundary(gdf_blocks)

    xmin, ymin, xmax, ymax = gdf_blocks.total_bounds

    if matched_parcel_geometries:
        parcel_bounds = [g.bounds for g in matched_parcel_geometries]
        median_w = statistics.median(b[2] - b[0] for b in parcel_bounds)
        median_h = statistics.median(b[3] - b[1] for b in parcel_bounds)
        pad_x = median_w * 1.5
        pad_y = median_h * 1.5
    else:
        pad_x = (xmax - xmin) * 0.08
        pad_y = (ymax - ymin) * 0.08

    view_bounds = (
        float(xmin - pad_x),
        float(ymin - pad_y),
        float(xmax + pad_x),
        float(ymax + pad_y),
    )

    view_xmin, view_ymin, view_xmax, view_ymax = view_bounds
    local_parcels = parcels.cx[view_xmin:view_xmax, view_ymin:view_ymax].copy()

    buildings = None
    try:
        buildings = fetch_county_buildings(*view_bounds)
        if buildings is not None and not buildings.empty:
            boundary_gdf = gpd.GeoDataFrame(geometry=[outer_boundary], crs="EPSG:3857")
            buildings = gpd.clip(buildings, boundary_gdf)
            buildings = buildings[buildings.geometry.notnull()].copy()
            buildings = buildings[~buildings.geometry.is_empty].copy()
    except Exception as exc:
        print(f"Building cache failed: {exc}")
        buildings = None

    roads = None
    if not args.skip_roads:
        roads = fetch_osm_roads(*view_bounds)

    cache = {
        "version": 1,
        "crs": "EPSG:3857",
        "source_blocks": os.path.abspath(args.blocks),
        "source_shapefile": os.path.abspath(args.shapefile),
        "blocks": gdf_blocks,
        "outer_boundary": outer_boundary,
        "local_parcels": local_parcels,
        "buildings": buildings,
        "roads": roads,
        "parcel_number_labels": parcel_number_labels,
        "street_label_points": street_label_points,
        "view_bounds": view_bounds,
    }

    output = Path(args.output)
    with output.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nPreprocessing complete.")
    print(f"Cache: {output.resolve()}")
    print(f"Blocks: {len(gdf_blocks)}")
    print(f"Local parcels cached: {len(local_parcels)}")
    print(f"Buildings cached: {0 if buildings is None else len(buildings)}")
    print(f"Road segments cached: {0 if roads is None else len(roads)}")
    print("\nYou can now iterate with render_map.py without repeating geocoding/parcels.")


if __name__ == "__main__":
    main()
