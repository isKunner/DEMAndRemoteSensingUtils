<div id="top"></div>

<!-- PROJECT LOGO -->
<br />

<div>
  <h3 align="center">OperationOfArcGIS</h3>

  <p align="center">
    Various utility modules used in processing Digital Elevation Models (DEMs) and Remote Sensing
    <br />
    <a href="https://github.com/isKunner/DEMAndRemoteSensingUtilse">Kevin Chen</a>
  </p>
</div>


## 1. analysis.py

### 1.1. `analyze_two_dems`

*   **Function**: Compares the differences between two DEMs using statistical methods like the P-test.
*   **Primary Use Case**: Comparing a super-resolved DEM with its original counterpart.

<p align="right">(<a href="#top">back to top</a>)</p>

## 2. check_dam_volume.py

### 2.1. `process_checkdam_capacity`

*   **Function**: Calculates check dam capacity (storage volume) and updates DEM output. Only processes 2slope+road/2slope groupings.
*   **Core Workflow**:
  1. Read DEM and shapefile data
  2. Filter target groupings (group_type = 12, 2)
  3. Calculate control area and storage volume using BFS algorithm
  4. Update shapefile with control area and capacity fields
  5. Update DEM with siltation elevation values
*   **Key Features**:
  - Supports automatic CRS transformation
  - Uses breadth-first search (BFS) to calculate upstream control range
  - Optimized memory usage (uint8 data types)
  - Handles multiple check dams in batch

### 2.2. `bfs_upstream_elevation`

*   **Function**: Performs breadth-first search from upstream slope to calculate upstream control range, area, and storage volume.
*   **Algorithm**: 8-direction BFS with elevation constraints (only processes pixels below siltation surface)
*   **Parameters**:
  - `limit_area`: Maximum siltation area limit (default 5,000,000 m²)
  - `slope`: Height increment slope (0.0021 m elevation increase per meter distance)

### 2.3. `update_dem_precise`

*   **Function**: Precisely updates DEM data (100-1000x speedup) using vectorized operations.
*   **Optimization**: 
  - Collects all valid pixels first
  - Deduplicates pixels covered by multiple dams (keeps highest elevation)
  - Batch updates using vectorized assignment

<p align="right">(<a href="#top">back to top</a>)</p>

## 3. clean_nodata.py

### 3.1. `remove_nodata_rows_cols`

*   **Function**: Removes excess rows and columns containing only NoData values from the edges of a DEM.
*   **Current Use Case**: Works with DEMs in a geographic coordinate system (non-projected) where the target data area is surrounded by NoData padding.
*   **Example (Works)**:
    ```
    | None | None | None | None |
    |------|------|------|------|
    | 1    | 1    | 1    | None |
    | 1    | 1    | 1    | None |
    ```
*   **Example (Fails/May Produce Incorrect Results)**:
    ```
    | None | None | None | None |
    |------|------|------|------|
    | 1    | 1    | 1    | None |
    | None | None | None | None |
    | 1    | 1    | 1    | None |
    ```

### 3.2. Related Knowledge: Affine Transform

*   **Description**: An affine transformation matrix defines how pixel coordinates map to geographic/projected coordinates.
*   **Example (Wangmaogou DEM - Projected Coordinate System)**:
    ```python
    Affine(a=2.0, b=0.0, c=440820.09892957023,
           d=0.0, e=-2, f=4165146.2840191615,
           g=0.0, h=0.0, i=1.0)
    ```
*   **Parameter Meanings**:

    | Param | Meaning         | Description                                               |
    | :---- | :-------------- | :-------------------------------------------------------- |
    | a     | X Scale         | Column-wise change rate for longitude/X projected coord.  |
    | b     | X-Y Rotation    | Row-wise change rate for longitude/X projected coord.     |
    | c     | X Translation   | Top-left corner X coordinate.                             |
    | d     | Y-X Rotation    | Column-wise change rate for latitude/Y projected coord.   |
    | e     | Y Scale         | Row-wise change rate for latitude/Y projected coord. (Negative means image goes down) |
    | f     | Y Translation   | Top-left corner Y coordinate.                             |
    | g     | Perspective X   | Fixed at 0.                                               |
    | h     | Perspective Y   | Fixed at 0.                                               |
    | i     | Scaling Factor  | Fixed at 1.                                               |

*   **Coordinate Transformation Equations**:
    *   Pixel -> Geo:
        ```
        x_geo = a * col + b * row + c
        y_geo = d * col + e * row + f
        ```
    *   Matrix Form:
        $$
        \begin{pmatrix}
        a & b & c \\
        d & e & f \\
        g & h & i
        \end{pmatrix}
        \times
        \begin{pmatrix}
        x_{\text{pixel}} \\
        y_{\text{pixel}} \\
        1
        \end{pmatrix}
        =
        \begin{pmatrix}
        x_{\text{geo}} \\
        y_{\text{geo}} \\
        1
        \end{pmatrix}
        $$

<p align="right">(<a href="#top">back to top</a>)</p>

## 4. conver_coordinate.py

### 4.1. `convert_dem_egm2008_to_navd88_foot`

*   **Function**: Converts DEM from EGM2008 vertical datum (meters) to NAVD88 vertical datum (US Survey Foot), with horizontal CRS synchronized to NAD83 HARN Ohio South.
*   **Coordinate Systems**:
  - Source: WGS84 (horizontal) + EGM2008 (vertical, meters) - EPSG:4326+3855
  - Target: NAD83 HARN Ohio South (horizontal, feet) + NAVD88 (vertical, feet) - EPSG:3754+5703
*   **Conversion Process**:
  1. Horizontal coordinate transformation (WGS84 → NAD83 HARN Ohio South)
  2. Vertical datum transformation (EGM2008 → NAVD88)
  3. Unit conversion (meters → US Survey Foot)
*   **Note**: Supports two conversion modes: center coordinate approximation and pixel-by-pixel precision conversion.

<p align="right">(<a href="#top">back to top</a>)</p>

## 5. convert_shp_format.py

### 5.1. `points_shp_to_merged_multipolygon`

*   **Function**: Converts point SHP files to polygons, then merges them into a single MultiPolygon.
*   **Process**:
  1. Reads point SHP file
  2. Creates square buffer zones for each point (default 100x100 meters)
  3. Merges all polygons into a single MultiPolygon using cascaded_union
*   **Use Case**: Converting point features (e.g., dam locations) to area features for spatial analysis.

### 5.2. `shp_points_to_polygons`

*   **Function**: Converts point SHP files to buffered polygon GeoJSON (EPSG:4326).
*   **Features**:
  - Automatically handles CRS transformation
  - Creates square buffer zones (default 10x10 meters)
  - Outputs GeoJSON format, convenient for web visualization
*   **Coordinate System Handling**: Temporarily converts to projected coordinate system (EPSG:5070) for precise buffer calculation, then converts back to WGS84.

<p align="right">(<a href="#top">back to top</a>)</p>

## 6. coordinate_system.py

### 6.1. `create_coordinate_transformer`

*   **Function**: Creates a robust coordinate transformation object to convert a point's location from Coordinate System A to Coordinate System B.
*   **Core Idea**: Source Location -> Get WGS84 Lat/Lon -> Get Location in Target Coordinate System.
*   **Support Matrix**:

    | Source CRS | Target CRS | Support Status | Handling Method                           |
    |:-----------|:-----------|:---------------|:------------------------------------------|
    | Geographic | WGS84      | ✅ Direct       | Direct Transformation                     |
    | Projected  | WGS84      | ✅ Direct       | Direct Transformation                     |
    | Geographic | Projected  | ✅ Indirect     | Via WGS84 Intermediate Step               |
    | Projected  | Geographic | ✅ Indirect     | Via WGS84 Intermediate Step               |
    | Geographic | Geographic | ✅ Direct       | Direct Transformation                     |
    | Projected  | Projected  | ✅ Conditional  | Direct if same datum, otherwise via WGS84 |

Additional Features:
- Automatically detects CRS type (geographic vs projected) and datum.
- Validates coordinate systems using OSR authority information.
- Performs sanity checks on test points to ensure transformation validity.
- Falls back to a two-step WGS84 intermediate conversion when direct transformation fails or datums differ.
- Issues warnings for potential accuracy loss (e.g., geographic CRS with different datums).
- Returns both the target CRS object and a unified transformation function: `transform_func(x, y, z=0) -> (tx, ty)`.

### 6.2. `get_shp_wgs84_bounds`

*  **Function**: Reads a Shapefile and returns its bounding box in WGS84 (EPSG:4326) longitude/latitude coordinates.
*  **Input**: Path to a valid Shapefile (`.shp`).
Output: `(lon_min, lat_min, lon_max, lat_max)` as floats.
Behavior:
- Raises `FileNotFoundError` if the Shapefile does not exist.
- Raises `ValueError` if the file is empty or lacks CRS information.
- Automatically reprojects to EPSG:4326 if the original CRS is not WGS84.
- Ensures output bounds are always in geographic coordinates suitable for functions expecting latitude/longitude (e.g., `crop_tif_by_bounds`).

### 6.3. `transform_coordinates`

*   **Function**: Executes coordinate transformation using a transformer function returned by `create_coordinate_transformer`.
*   **Purpose**: Provides a safe, error-handled wrapper for point-wise conversion.
Usage:
    ```python
    target_crs, transformer = create_coordinate_transformer(src_srs, target_srs)
    tx, ty = transform_coordinates(transformer, x, y)
    ```

### 6.4. `batch_set_coordinate_system`

*   **Function**: Assign a coordinate system to TIF files.

### 6.5. `set_coordinate_system_for_tif`

*   **Function**: Assign a coordinate system to the sigle TIF file.

### 6.6. `reproject_raster_file`

*   **Function**: Reprojects a raster file (e.g., GeoTIFF) to a new coordinate reference system (CRS).
*   **Parameters**:
  - `input_path`: Path to input raster file
  - `output_path`: Path where reprojected file will be saved
  - `target_crs`: Target CRS in EPSG code or WKT string
*   **Features**: Handles multi-band rasters with appropriate resampling methods.

### 6.7. `add_vertical_datum_with_backup`

*   **Function**: Safely adds vertical datum to GeoTIFF DEM with automatic backup creation.
*   **Supported Datums**: EGM2008, EGM96
*   **Process**: Creates backup file, modifies CRS metadata without changing pixel values, uses gdal_edit.py internally.

<p align="right">(<a href="#top">back to top</a>)</p>

## 7. crop_dem_from_cordinate.py

### 7.1. `add_buffer_to_bounds`

*   **Function**: Adds a buffer zone around a given bounding box defined by latitude/longitude coordinates.
*   **Use Case Example**: Super-resolution often degrades edge pixels; cropping a larger area with a buffer allows trimming poor-quality edges after processing.
*   **⚠️ Important Caveat**:
    *   Buffers added to multiple tiles may result in inconsistent real-world areas due to varying ground distances per degree of longitude/latitude.
        *   **Geographic CRS**: Tile spans might be equal in degrees but differ in actual meters.
        *   **Projected CRS**: Calculations based on input lat/lon can lead to inconsistent cropped areas; further refinement needed.

### 7.2. `crop_tif_by_bounds`

*   **Function**: Crops a TIFF file based on a specified rectangular bounding box.
*   **Example Workflow**:
    ```
    Input Parameters:
    lon_min=110.347, lat_min=37.595
    lon_max=110.348, lat_max=37.596

    1. Create WGS84 Geometry:
       Polygon([(110.347,37.595), (110.348,37.595), ...])

    2. Read TIF File:
       CRS: EPSG:4527 (Projected)
       Affine: a=2.0, e=-2.0, c=440000, f=4166000

    3. Automatic Coordinate Transformation:
       (110.347,37.595) → (440820.1, 4165146.3)
       (110.348,37.596) → (440822.1, 4165148.3)

    4. Calculate Pixel Coordinates:
       Columns: 410 ~ 411
       Rows: 425 ~ 426

    5. Extract Pixel Data:
       out_image.shape = (1, 1, 1)  // Extracts 1x1 pixel(s)

    6. Generate New Transform Matrix:
       New Top-Left Corner: (440820.0, 4165148.0)

    7. Save File:
       Outputs the cropped small TIF file
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

## 8. crop_dem_from_dem.py

### 8.1. `crop_source_to_reference`

*   **Function**: Resamples and crops a target file based on the extent and properties of an already cropped reference TIFF file (ensuring a square output).
*   **Feature**: Automatically handles coordinate system transformations; the coordinate system of the reference file dictates the output area/crs.
*   **Input/Output Flexibility**:
  - Single reference file + single output path
  - Single reference file + output directory
  - Multiple reference files list + output directory
  - Multiple reference files list + output paths list
  - Input directory + output directory

<p align="right">(<a href="#top">back to top</a>)</p>

## 9. crop_shp_from_shp.py

### 9.1. `clip_points_by_boundary`

*   **Function**: Clips point data by boundary polygon.
*   **Use Case**: Extracting dam points within US state boundaries.
*   **Process**:
  1. Reads point data (e.g., global dam data)
  2. Reads boundary data (e.g., US state boundaries)
  3. Creates boundary polygon union for efficiency
  4. Filters points within boundary

### 9.2. `clip_points_by_boundary_with_state_info`

*   **Function**: Clips point data by boundary polygon and retains state information.
*   **Additional Features**:
  - Performs spatial join to associate points with state polygons
  - Adds state attributes (NAME, REGION, DIVISION) to output
  - Provides statistics of dam counts by state

<p align="right">(<a href="#top">back to top</a>)</p>

## 10. generated_shp_from_tif.py

### 10.1. `create_boundary_shp_from_dem`

*   **Function**: Generates rectangular boundary SHP file corresponding to TIF file, with rectangle boundaries matching TIF boundaries.
*   **Output**: Creates single polygon feature representing TIF extent with original CRS.

### 10.2. `batch_create_boundary_shp_from_dir`

*   **Function**: Batch processes TIF files in directory to generate corresponding boundary SHP files.
*   **Options**: Can save SHP files in same directory as TIF or specify output directory.

<p align="right">(<a href="#top">back to top</a>)</p>

## 11. get_flow_accumulation.py

### 11.1. `calculate_flow_accumulation`

*   **Function**: Calculates flow accumulation for a DEM using WhiteboxTools.
*   **Installation Note**: Whitebox actually calls an exe file to execute. After installing the library using pip and executing it, exe will be automatically downloaded, but installing it using conda will not result in errors. After downloading and decompressing using pip, the path may also be incorrect. Selected $0 is enough. You need to manually adjust the location of the directory or the directory where whitebox_tools.exe is defined and displayed. 
    ``wbt.exe_path = r"C:\Users\Kevin\anaconda3\envs\geo-torch\Lib\site-packages\whitebox\WBT"``
*   **Workflow (Based on Whitebox GAT concepts)**:
    1.  Original DEM
    2.  First Pass Depression Filling (`fill_depressions`)
    3.  Second Pass Depression Filling (`fill_depressions`)
    4.  First Pass Depression Breaching (`breach_depressions`)
    5.  Second Pass Depression Breaching (`breach_depressions`)
    6.  Flow Direction Calculation (`d8_pointer`)
    7.  Flow Accumulation Calculation (`d8_flow_accumulation`)
    8.  Output Result
*   **Related Resource**:
    *   [Whitebox Geospatial Analysis Tools (Whitebox GAT)](https://www.whiteboxgeo.com/manual/wbt_book/intro.html)
    *   **Common Terrain Tools**: `slope`, `aspect`, `curvature`, `ruggedness`, `tpi`
    *   **Common Hydrological Tools**: `fill_depressions`, `breach_depressions`, `d8_pointer`, `d8_flow_accumulation`, `watershed`
    *   **Common Image Processing Tools**: `resample`, `clip_raster_to_polygon`, `mosaic`, `change_vector_analysis`

<p align="right">(<a href="#top">back to top</a>)</p>

## 12. get_information.py

### 12.1. `get_pixel_size_accurate`

*   **Function**: Calculates the accurate physical size of a raster pixel in meters.
*   **Logic**:
    *   **Case 1 (Projected CRS)**: Read directly from transform parameters.
    *   **Case 2 (Geographic CRS)**: Transform to UTM (or similar) and calculate the actual ground distance.

### 12.2. `get_tif_latlon_bounds`

*   **Function**: Retrieves the bounding box of a TIFF file expressed in WGS84 Latitude/Longitude coordinates.
*   **Logic**:
    *   **Case 1 (Geographic CRS)**: Read directly.
    *   **Case 2 (Projected CRS)**: Transform extents to WGS84.

### 12.3. `get_crs_transformer`

*   **Function**: Creates a transformer object for converting coordinates between two specific Coordinate Reference Systems.
*   **Example**:
    ```python
    transformer = get_crs_transformer("EPSG:4326", "EPSG:32650")
    lon, lat = 110.0, 35.0
    x, y = transformer.transform(lon, lat)
    ```

### 12.4. `geo_to_pixel`

*   **Function**: Converts geographic coordinates (lat/lon) to image pixel coordinates (row, col).

### 12.5. `pixel_to_geo`

*   **Function**: Converts image pixel coordinates (row, col) to geographic coordinates (lat/lon).

### 12.6. `pixel_to_pixel`

*   **Function**: Converts pixel coordinates from source raster to pixel coordinates in target raster (handles different CRS).

### 12.7. `get_tif_info`

*   **Function**: Gets basic attribute information of TIFF raster data.
*   **Output**: Dictionary containing dimensions, data type, CRS, bounds, transform matrix, statistics, and vertical datum information.

### 12.8. `get_degree_per_meter`

*   **Function**: Calculates the degree change per meter distance at given latitude/longitude.
*   **Use Case**: Useful for buffer calculations in geographic coordinate systems.

<p align="right">(<a href="#top">back to top</a>)</p>

## 13. modify_dem.py

### 13.1. `batch_modify_tifs_vectorized`

*   **Function**: Batch modifies multiple TIF file values using vectorized + parallel processing.
*   **Core Algorithm**: 
  1. Parallel reading of TIF metadata and data
  2. Calculates robust min/max ranges using modified Z-score method
  3. Vectorized transformation: `output = matrix * (max - min) + min`
  4. Parallel writing of output files
*   **Advantages**: 
  - Handles outliers using modified Z-score (threshold 3.5)
  - Full vectorization and parallelization
  - Supports multi-band rasters

### 13.2. `calculate_robust_min_max_modified_zscore_vectorized`

*   **Function**: Calculates robust min/max using modified Z-score method for 2D/3D arrays.
*   **Formula**: Modified Z-Score = 0.6745 * (x - median) / MAD
*   **MAD**: Median Absolute Deviation

<p align="right">(<a href="#top">back to top</a>)</p>

## 14. modify_from_shp.py

### 14.1. `check_dam_info_extract`

*   **Function**: Complete check dam information extraction workflow (5-step process).
*   **Steps**:
  1. Crop DEM by shapefile bounds (with buffer)
  2. Edge extension of dam slope features
  3. Extract DEM minimum elevation + calculate flow accumulation
  4. Calculate relative height of dam slopes
  5. Calculate final elevation values
*   **Scenarios Supported**:
  - 2 slopes + 1 road
  - 1 slope + 1 road
  - 2 slopes (no road)
  - 1 slope (no road)

### 14.2. `extract_elevation_from_dem`

*   **Function**: Extracts min/max elevation values from DEM for geometries (optimized batch version).
*   **Optimization**: Uses batch rasterization instead of per-feature mask for 10-100x speedup.

### 14.3. `update_dem_with_elevation_values`

*   **Function**: Generates new DEM based on geometries and 'elev' field (optimized version).
*   **Features**: Batch rasterization, automatic filtering of invalid features, vectorized update.

<p align="right">(<a href="#top">back to top</a>)</p>

## 15. modify_shp.py

### 15.1. `filter_shp`

*   **Function**: Reads SHP file and filters or removes rows by type field.
*   **Modes**: 
  - "include": Keep rows with specified type
  - "exclude": Remove rows with specified type
*   **Use Case**: Extracting specific feature types (e.g., only 'upstream' or 'control_area' features).

<p align="right">(<a href="#top">back to top</a>)</p>

## 16. resize_tif.py

### 16.1. `unify_dem`

*   **Function**: Unifies input DEM to match resolution and coordinate system of target DEM (supports multi-band). Can also add buffer.
*   **Potential Issues**:
    *   Floating-point precision errors during coordinate calculations might cause slight variations in column/row counts.
    *   Resampling algorithms (e.g., bilinear) might introduce minor deviations at boundaries.
    *   Pixel alignment issues between old and new coordinate systems.
*   **Note**: Minor irregularities might exist but are generally considered negligible for overall calculations.

### 16.2. `resample_to_target_resolution`

*   **Function**: Simple resampling function suitable for projected coordinate systems. Takes desired resolution in meters as input.

### 16.3. `resample_geography_to_target_resolution`

*   **Function**: Resamples a DEM, potentially transforming from a geographic coordinate system to a projected one before resampling.
*   **⚠️ Warning**: Currently experiencing misalignment issues, possibly due to source data characteristics.

### 16.4. `upsample_geography_data`

*   **Function**: Upsamples (increases resolution) of data in geographic coordinate system, supporting multi-band.
*   **Method**: Uses bilinear interpolation to increase resolution by specified factor.

### 16.5. `downsample_tif`

*   **Function**: Downsamples (decreases resolution) input TIFF file, supporting multi-band.
*   **Method**: Uses average pooling (default) or other resampling methods.

### 16.6. `downsample_tif_advanced`

*   **Function**: Advanced downsampling with feature enhancement.
*   **Steps**:
  1. Laplacian sharpening (enhances edges)
  2. Gaussian smoothing (reduces noise)
  3. Downsample using specified resampling method
*   **Use Case**: Suitable for DEMs where feature preservation is important during downsampling.

### 16.7. `downsample_directory` / `downsample_directory_advanced`

*   **Function**: Batch processes all TIFF files in directory for downsampling.
*   **Features**: Supports both standard and advanced downsampling modes.

<p align="right">(<a href="#top">back to top</a>)</p>

## 17. splicing_dem.py

### 17.1. `merge_georeferenced_tifs`

*   **Function**: Mosaics multiple GeoTIFF files into a single large GeoTIFF, handling overlapping regions. Requires identical coordinate systems, NoData values, etc.
*   **Implementation Approach**:
    1.  Determine the overall extent of all files to be merged.
    2.  Create a large empty array.
    3.  Populate the array with data from each file, keeping count of overlaps for potential averaging.
*   **Overlap Strategies**:
  - 'mean': Average of overlapping pixels
  - 'first': First valid pixel
  - 'last': Last valid pixel
  - 'max': Maximum value
  - 'min': Minimum value
*   **Key Considerations**:
    *   **NoData Handling**: Attempts to manage source NoData values to prevent them from affecting calculations (especially in 'mean' strategy).
    *   **Global Bounds & Transform**: Correctly calculates the final mosaic's geographic extent, dimensions, and affine transform matrix.
    *   **⚠️ Dimension Calculation**: Uses `int(round(...))`. While usually fine, floating-point precision mismatches might cause slight dimension discrepancies.
    *   **⚠️ Memory Usage**: Current code uses `ds.read()` to load entire source files at once, which can consume significant memory for large files.

<p align="right">(<a href="#top">back to top</a>)</p>

## 18. split_dem.py

### 18.1. `split_tif`

*   **Function**: Splits a large DEM TIFF file into smaller tiles.
*   **Implementation Details**:
    *   Implements a sliding window logic based on `step = tile_size - overlap` to create overlaps.
    *   **NoData Handling (Initial)**: Attempts to skip tiles where all pixels are NoData or NaN, optimizing by avoiding creation of meaningless files.
*   **Limitation**: Currently supports only single-band rasters.

<p align="right">(<a href="#top">back to top</a>)</p>

## 19. split_shp.py

### 19.1. `split_shp_by_us_regions`

*   **Function**: Splits SHP file by US state regions (for platform upload size limits).
*   **Grouping Strategy**:
  1. Groups by US Census Bureau regions and divisions
  2. Creates 16 groups (most with 3 states, last 2 with 4 states)
  3. Group 13 split into two files due to large size
*   **Output Structure**: Creates separate directories for each group with subset SHP files.

<p align="right">(<a href="#top">back to top</a>)</p>

## 20. utils.py

### 20.1. `read_tif`

*   **Function**: Successfully reads common raster attributes: pixel values, affine transform, coordinate reference system, and NoData value.

### 20.2. `write_tif`

*   **Function**: Writes data to a GeoTIFF file using provided core information (data, transform, crs, nodata, etc.).

### 20.3. `pixel_to_geo_coords`

*   **Function**: This function converts pixel coordinates to geographic coordinates using GDAL's geotransform parameters.`.

### 20.4. `get_geotransform_and_crs`

*   **Function**: Get geotransform and CRS information from a TIFF file.

### 20.5. `calculate_meters_per_degree_precise`

*   **Function**: Precisely calculates meters per degree at given longitude/latitude using UTM projection.
*   **Method**: Uses small offset (delta=0.00001) to calculate actual ground distance.

<p align="right">(<a href="#top">back to top</a>)</p>

## Installation and Usage

### Requirements

```bash
pip install rasterio geopandas numpy scipy matplotlib whitebox pyproj gdal shapely
```

### Importing the Package

```python
from DEMAndRemoteSensingUtils import (
    crop_source_to_reference,
    read_tif, 
    write_tif,
    crop_tif_by_bounds,
    pixel_to_geo_coords,
    get_geotransform_and_crs,
    pixel_to_pixel,
    calculate_meters_per_degree_precise,
    check_dam_info_extract,
    filter_shp,
    process_checkdam_capacity,
    batch_modify_tifs_vectorized,
    merge_geo_referenced_tifs
)
```

### Example Usage

```python
# Example 1: Crop DEM by coordinates
from DEMAndRemoteSensingUtils import crop_tif_by_bounds

crop_tif_by_bounds(
    input_tif_path="input.tif",
    output_tif_path="output.tif",
    lon_min=110.347, lat_min=37.595,
    lon_max=110.348, lat_max=37.596,
    buffer_distance_km=2
)

# Example 2: Calculate flow accumulation
from DEMAndRemoteSensingUtils import calculate_flow_accumulation

calculate_flow_accumulation(
    dem_path="dem.tif",
    flow_accum_path="flow_accum.tif"
)

# Example 3: Merge multiple DEMs
from DEMAndRemoteSensingUtils import merge_geo_referenced_tifs

merge_geo_referenced_tifs(
    input_dir="tiles/",
    output_path="merged.tif",
    overlap_strategy='mean'
)
```

<p align="right">(<a href="#top">back to top</a>)</p>

## License

This project is licensed under the MIT License.

<p align="right">(<a href="#top">back to top</a>)</p>
