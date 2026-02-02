#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: split_shp
# @Time    : 2026/1/24 15:32
# @Author  : Kevin
# @Describe: Split dams by US states and reorganize corresponding DEM files

import os
import shutil
import geopandas as gpd
import pandas as pd

# US state name to abbreviation mapping (UPPERCASE)
state_to_abbr = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    "american samoa": "AS", "district of columbia": "DC", "guam": "GU",
    "puerto rico": "PR", "commonwealth of the northern mariana islands": "MP",
    "united states virgin islands": "VI"
}


def split_dams_by_states(all_dams_shp_path, states_shp_path, output_base_dir):
    """
    Split dams SHP by states using spatial join.
    Each state's dams are saved in its own subdirectory (e.g., AK/AK.shp)
    
    Args:
        all_dams_shp_path (str): Path to complete dams SHP file (e.g., GeoDAR_v11_dams.shp)
        states_shp_path (str): Path to US states boundary SHP file
        output_base_dir (str): Base output directory (e.g., USA_DAM_ByStates)

    Returns:
        dict: Dictionary mapping state_abbr -> output_shp_path
    """
    print(f"\nReading dams SHP: {all_dams_shp_path}")
    dams_gdf = gpd.read_file(all_dams_shp_path)
    print(f"Total dams: {len(dams_gdf)}")
    
    print(f"\nReading states SHP: {states_shp_path}")
    states_gdf = gpd.read_file(states_shp_path)
    
    # Find state name column
    state_name_col = None
    for col in ['NAME', 'STATE_NAME', 'name', 'state_name', 'State']:
        if col in states_gdf.columns:
            state_name_col = col
            break
    
    if not state_name_col:
        raise ValueError(f"Cannot find state name column. Available: {list(states_gdf.columns)}")
    
    print(f"Found {len(states_gdf)} states/territories")
    
    # Ensure CRS match
    if not dams_gdf.crs.equals(states_gdf.crs):
        print(f"Converting states CRS to match dams CRS...")
        states_gdf = states_gdf.to_crs(dams_gdf.crs)
    
    # Rename dams NAME column if exists to avoid conflict
    if 'NAME' in dams_gdf.columns:
        dams_gdf = dams_gdf.rename(columns={'NAME': 'DAM_NAME'})
    
    # Spatial join: assign each dam to a state
    print("\nPerforming spatial join...")
    joined = gpd.sjoin(
        dams_gdf,
        states_gdf[[state_name_col, 'geometry']],
        how='left',
        predicate='intersects'
    )
    
    # Rename state column
    if state_name_col != 'STATE_NAME':
        joined = joined.rename(columns={state_name_col: 'STATE_NAME'})
    
    # Remove index_right if exists
    if 'index_right' in joined.columns:
        joined = joined.drop(columns=['index_right'])
    
    # Get unique states
    unique_states = joined['STATE_NAME'].dropna().unique()
    print(f"Dams distributed across {len(unique_states)} states/territories")
    
    os.makedirs(output_base_dir, exist_ok=True)
    output_files = {}
    
    for state_name in sorted(unique_states):
        # Get dams for this state
        state_dams = joined[joined['STATE_NAME'] == state_name].copy()
        
        if len(state_dams) == 0:
            continue
        
        # Get state abbreviation
        state_abbr = state_to_abbr.get(str(state_name).lower())
        if not state_abbr:
            state_key = str(state_name).replace(' ', '_').lower()
            state_abbr = state_to_abbr.get(state_key)
        if not state_abbr:
            state_abbr = str(state_name).replace(' ', '_').upper()
        
        # Create state subdirectory
        state_dir = os.path.join(output_base_dir, state_abbr)
        os.makedirs(state_dir, exist_ok=True)
        
        # Reset index to get sequential IDs (0, 1, 2, ...)
        state_dams = state_dams.reset_index(drop=True)
        
        # Save state's dams SHP
        output_path = os.path.join(state_dir, f"{state_abbr}.shp")
        state_dams.to_file(output_path)
        output_files[state_abbr] = output_path
        print(f"  {state_abbr}: {len(state_dams)} dams -> {output_path}")
    
    print(f"\nTotal {len(output_files)} state dam SHP files created in {output_base_dir}")
    return output_files


def reorganize_dems_by_state(
    dams_by_region_dir,
    dem_base_dirs,
    output_dem_base_dir
):
    """
    Reorganize DEM files by state based on dam SHP files.
    
    For each dam in SHP (with ID and Name fields):
    - Find corresponding TIF in {dem_base_dir}/{group_name}/{id}.tif
    - Copy to {output_dem_base_dir}/{dem_type}/{state_abbr}/{id}.tif
    - Also save state-specific SHP files
    
    Args:
        dams_by_region_dir: Path to USA_DAM_ByRegion directory with group folders
        dem_base_dirs: Dict of {dem_type: dem_base_path}
        output_shp_dir: Output directory for state SHP files
        output_dem_base_dir: Output directory for reorganized DEMs
    """
    print("=" * 60)
    print("Reorganizing DEMs by state")
    print("=" * 60)
    
    # Stats
    stats = {
        'groups_processed': 0,
        'dams_processed': 0,
        'dems_copied': {dem_type: 0 for dem_type in dem_base_dirs.keys()},
        'errors': []
    }
    
    # Create output directory
    os.makedirs(output_dem_base_dir, exist_ok=True)
    
    # Dictionary to track next ID for each state
    state_next_id = {}
    
    # Get all group directories
    group_dirs = [d for d in os.listdir(dams_by_region_dir)
                  if d.startswith('group_') and os.path.isdir(os.path.join(dams_by_region_dir, d))]
    
    print(f"Found {len(group_dirs)} group directories")
    
    for group_dir in sorted(group_dirs):
        group_path = os.path.join(dams_by_region_dir, group_dir)
        shp_files = [f for f in os.listdir(group_path) if f.endswith('.shp')]
        
        for shp_file in shp_files:
            shp_path = os.path.join(group_path, shp_file)
            
            # Extract group name from filename (e.g., GeoDAR_v11_dams_of_USA_group1.shp -> GeoDAR_v11_dams_of_USA_group1)
            group_name = shp_file.replace('.shp', '')
            
            print(f"\nProcessing: {group_name}")
            
            # Read SHP
            gdf = gpd.read_file(shp_path)
            stats['groups_processed'] += 1
            
            # Find state name column
            state_col = None
            for col in ['Name', 'NAME', 'name', 'State', 'STATE']:
                if col in gdf.columns:
                    state_col = col
                    break
            
            if not state_col:
                print(f"  Warning: No state name column found in {shp_file}, skipping")
                continue
            
            print(f"  Dams in this group: {len(gdf)}")
            
            # Process each dam (idx is the old_id in group SHP)
            for idx, row in gdf.iterrows():
                old_id = idx  # Original ID in group SHP (matches TIF filename)
                state_name = row[state_col]
                
                if pd.isna(state_name):
                    continue
                
                # Get state abbreviation
                state_abbr = state_to_abbr.get(str(state_name).lower())
                if not state_abbr:
                    state_abbr = str(state_name).replace(' ', '_').upper()
                
                # Get new sequential ID for this state (starting from 0)
                if state_abbr not in state_next_id:
                    state_next_id[state_abbr] = 0
                new_id = state_next_id[state_abbr]
                state_next_id[state_abbr] += 1
                
                # Copy DEM files for this dam, rename to new_id
                for dem_type, dem_base_dir in dem_base_dirs.items():
                    # Source: {dem_base_dir}/{group_name}/{old_id}.tif
                    src_path = os.path.join(dem_base_dir, group_name, f"{old_id}.tif")
                    
                    # Handle CopernicusDEM _paired suffix
                    if dem_type == 'CopernicusDEM' and not os.path.exists(src_path):
                        src_path = os.path.join(dem_base_dir, group_name + '_paired', f"{old_id}.tif")
                    
                    if os.path.exists(src_path):
                        # Destination: {output_dem_base_dir}/{dem_type}/{state_abbr}/{new_id}.tif
                        state_dem_dir = os.path.join(output_dem_base_dir, dem_type, state_abbr)
                        os.makedirs(state_dem_dir, exist_ok=True)
                        
                        dst_path = os.path.join(state_dem_dir, f"{new_id}.tif")
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            stats['dems_copied'][dem_type] += 1
                        except Exception as e:
                            stats['errors'].append(f"Error copying {src_path}: {e}")
                
                stats['dams_processed'] += 1
            
            print(f"  Processed {len(gdf)} dams")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Groups processed: {stats['groups_processed']}")
    print(f"Dams processed: {stats['dams_processed']}")
    print(f"States with dams: {len(state_next_id)}")
    print("DEM files copied:")
    for dem_type, count in stats['dems_copied'].items():
        print(f"  {dem_type}: {count}")
    if stats['errors']:
        print(f"Errors: {len(stats['errors'])}")
        for err in stats['errors'][:5]:
            print(f"  - {err}")
    
    return stats


if __name__ == '__main__':
    print("=" * 60)
    print("Splitting dams by states and reorganizing DEMs...")
    print("=" * 60)
    
    # Configuration
    all_dams_shp = r"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\GeoDAR_v11_dams.shp"
    states_shp = r"C:\Users\Kevin\Documents\ResearchData\RangeOfUSA\States.shp"
    dams_by_states_dir = r"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\USA_DAM_ByStates"
    
    dams_by_region_dir = r"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\USA_DAM_ByRegion"
    dem_output_dir = r'D:\研究文件\ResearchData\USA_ByState'
    
    dem_base_dirs = {
        'USGSDEM': r'D:\研究文件\ResearchData\USA\USGSDEM',
        'GoogleRemoteSensing': r'D:\研究文件\ResearchData\USA\GoogleRemoteSensing',
        'CopernicusDEM': r'D:\研究文件\ResearchData\USA\CopernicusDEM'
    }
    
    # Step 1: Split dams SHP by states using spatial join
    # Output: USA_DAM_ByStates/AL/AL.shp (contains dams in Alabama), etc.
    print("\n" + "=" * 60)
    print("Step 1: Splitting dams by states...")
    print("=" * 60)
    state_shp_files = split_dams_by_states(all_dams_shp, states_shp, dams_by_states_dir)
    
    # Step 2: Reorganize DEM files by state
    # Uses USA_DAM_ByRegion/group_* to find DEMs and copy to state folders
    print("\n" + "=" * 60)
    print("Step 2: Reorganizing DEM files...")
    print("=" * 60)
    stats = reorganize_dems_by_state(
        dams_by_region_dir,
        dem_base_dirs,
        dem_output_dir
    )
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)
