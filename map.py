import argparse
import contextily as cx
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import os
import sys
from geopy.geocoders import ArcGIS
from shapely.geometry import Point

def main():
    # -------------------------------------------------------------------------
    # 1. SET UP COMMAND LINE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Generate a Neighborhood Watch Map.")
    parser.add_argument('-b', '--blocks', default='blocks.json', help="Path to the JSON file containing block addresses (default: blocks.json).")
    parser.add_argument('-s', '--shapefile', required=True, help="Path to the county parcel shapefile (.shp).")
    
    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Load the block data from the provided (or default) JSON file
    if not os.path.exists(args.blocks):
        print(f"Error: Block data file '{args.blocks}' not found.")
        sys.exit(1)
        
    with open(args.blocks, 'r') as f:
        try:
            BLOCK_DATA = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            sys.exit(1)

    shapefile_name = args.shapefile
    if not os.path.exists(shapefile_name):
        print(f"Error: Missing parcel file: {shapefile_name}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. GEOCODE ADDRESSES AND MAP TO PARCELS
    # -------------------------------------------------------------------------
    print(f"Loading county parcel data from {shapefile_name}...")
    parcels = gpd.read_file(shapefile_name).to_crs(epsg=3857)

    geolocator = ArcGIS(user_agent="neighborhood_watch_master_mapper")
    block_polygons = []

    for block_id, info in BLOCK_DATA.items():
        print(f"\nProcessing Block {block_id}...")
        coords = []
        
        for addr in info["addresses"]:
            loc = geolocator.geocode(addr)
            if loc and loc.raw.get('score', 0) >= 95:
                coords.append(Point(loc.longitude, loc.latitude))
                print(f"  + Added: {addr}")
                
        if not coords:
            print(f"  - No valid coordinates found for Block {block_id}. Skipping.")
            continue

        # Convert to Web Mercator and buffer to guarantee parcel intersection
        gdf_pts = gpd.GeoDataFrame(geometry=coords, crs="EPSG:4326").to_crs(epsg=3857)
        gdf_pts_buf = gdf_pts.copy()
        gdf_pts_buf.geometry = gdf_pts.geometry.buffer(3)
        
        # Spatial Join to pull the exact legal boundary polygons
        matched_parcels = gpd.sjoin(parcels, gdf_pts_buf, how="inner", predicate="intersects")
        matched_parcels = matched_parcels.drop_duplicates(subset=[parcels.geometry.name])
        
        if not matched_parcels.empty:
            merged_geom = matched_parcels.geometry.unary_union
            
            # Morphological Close: Buffer out to cross streets, buffer in to shrink inside property lines
            smooth_boundary = merged_geom.buffer(22, join_style=1).buffer(-25, join_style=1)
            
            if smooth_boundary.is_empty:
                 smooth_boundary = merged_geom.buffer(20, join_style=1).buffer(-21, join_style=1)
                 
            block_polygons.append({
                "block_id": block_id,
                "geometry": smooth_boundary,
                "active": info.get("active", False),
                "centroid": smooth_boundary.centroid
            })

    if not block_polygons:
        print("\nNo blocks were successfully processed. Exiting.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. BUILD GEODATAFRAMES
    # -------------------------------------------------------------------------
    gdf_blocks = gpd.GeoDataFrame(block_polygons, crs="EPSG:3857")

    # -------------------------------------------------------------------------
    # 4. RENDER MULTI-BLOCK MAP
    # -------------------------------------------------------------------------
    print("\nRendering map...")
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300)

    xmin, ymin, xmax, ymax = gdf_blocks.total_bounds
    padding = max(xmax - xmin, ymax - ymin) * 0.35
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)

    # Layer 1: Google Maps Background
    google_url = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
    cx.add_basemap(ax, source=google_url, zoom=18, interpolation='bilinear', zorder=1)

    # Layer 2: Parcel boundaries (faint grey lines)
    local_parcels = parcels.cx[xmin - padding:xmax + padding, ymin - padding:ymax + padding]
    local_parcels.plot(ax=ax, facecolor='none', edgecolor='#b0b0b0', linewidth=0.5, alpha=0.6, zorder=2)

    # Layer 3: Individual Block Bubbles
    for _, row in gdf_blocks.iterrows():
        color = '#00E676' if row['active'] else '#212121' 
        fill_color = '#00E676' if row['active'] else '#ffffff'
        
        # Translucent Fill
        gpd.GeoSeries([row['geometry']]).plot(
            ax=ax, facecolor=fill_color, edgecolor='none', alpha=0.25, zorder=3
        )
        # Crisp Outer Border
        gpd.GeoSeries([row['geometry']]).plot(
            ax=ax, facecolor='none', edgecolor=color, linewidth=4.5, zorder=4
        )

        # Layer 4: Block Numbers
        c = row['centroid']
        ax.text(
            c.x, c.y, str(row['block_id']),
            fontsize=22, fontweight='bold', color='black',
            ha='center', va='center', zorder=5,
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='none', alpha=0.85)
        )

    ax.set_title("Saratoga Neighborhood Watch Map", fontsize=18, fontweight='bold', pad=20)
    ax.set_axis_off()

    output_file = "master_neighborhood_map.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Map saved as '{output_file}'.")

if __name__ == "__main__":
    main()