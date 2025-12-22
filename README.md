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

## 2. clean_nodata.py

### 2.1. `remove_nodata_rows_cols`

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

### 2.2. Related Knowledge: Affine Transform

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

## 3. coordinate_system

### 3.1. `create_coordinate_transformer`

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

### 3.2. `get_shp_wgs84_bounds`
Function: Reads a Shapefile and returns its bounding box in WGS84 (EPSG:4326) longitude/latitude coordinates.
Input: Path to a valid Shapefile (`.shp`).
Output: `(lon_min, lat_min, lon_max, lat_max)` as floats.
Behavior:
- Raises `FileNotFoundError` if the Shapefile does not exist.
- Raises `ValueError` if the file is empty or lacks CRS information.
- Automatically reprojects to EPSG:4326 if the original CRS is not WGS84.
- Ensures output bounds are always in geographic coordinates suitable for functions expecting latitude/longitude (e.g., `crop_tif_by_bounds`).

### 3.3. `transform_coordinates`
*   **Function**: Executes coordinate transformation using a transformer function returned by `create_coordinate_transformer`.
*   **Purpose**: : Provides a safe, error-handled wrapper for point-wise conversion.
Usage:
    ```python
    target_crs, transformer = create_coordinate_transformer(src_srs, target_srs)
    tx, ty = transform_coordinates(transformer, x, y)
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

## 4. crop_dem_from_cordinate.py

### 4.1. `add_buffer_to_bounds`

*   **Function**: Adds a buffer zone around a given bounding box defined by latitude/longitude coordinates.
*   **Use Case Example**: Super-resolution often degrades edge pixels; cropping a larger area with a buffer allows trimming poor-quality edges after processing.
*   **⚠️ Important Caveat**:
    *   Buffers added to multiple tiles may result in inconsistent real-world areas due to varying ground distances per degree of longitude/latitude.
        *   **Geographic CRS**: Tile spans might be equal in degrees but differ in actual meters.
        *   **Projected CRS**: Calculations based on input lat/lon can lead to inconsistent cropped areas; further refinement needed.

### 4.2. `crop_tif_by_bounds`

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

## 5. crop_dem_from_dem.py

### 5.1. `extract_matching_files`

*   **Function**: Resamples and crops a target file based on the extent and properties of an already cropped reference TIFF file (ensuring a square output).
*   **Feature**: Automatically handles coordinate system transformations; the coordinate system of the reference file dictates the output area/crs.

<p align="right">(<a href="#top">back to top</a>)</p>

## 6. get_flow_accumulation.py

### 6.1. `calculate_flow_accumulation`

*   **Function**: Calculates flow accumulation for a DEM.
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

## 7. get_information.py

### 7.1. `get_pixel_size_accurate`

*   **Function**: Calculates the accurate physical size of a raster pixel in meters.
*   **Logic**:
    *   **Case 1 (Projected CRS)**: Read directly from transform parameters.
    *   **Case 2 (Geographic CRS)**: Transform to UTM (or similar) and calculate the actual ground distance.

### 7.2. `get_tif_latlon_bounds`

*   **Function**: Retrieves the bounding box of a TIFF file expressed in WGS84 Latitude/Longitude coordinates.
*   **Logic**:
    *   **Case 1 (Geographic CRS)**: Read directly.
    *   **Case 2 (Projected CRS)**: Transform extents to WGS84.

### 7.3. `get_crs_transformer`

*   **Function**: Creates a transformer object for converting coordinates between two specific Coordinate Reference Systems.
*   **Example**:
    ```python
    transformer = get_crs_transformer("EPSG:4326", "EPSG:32650")
    lon, lat = 110.0, 35.0
    x, y = transformer.transform(lon, lat)
    ```

### 7.4. `geo_to_pixel`

*   **Function**: Converts geographic coordinates (lat/lon) to image pixel coordinates (row, col).

### 7.5. `pixel_to_geo`

*   **Function**: Converts image pixel coordinates (row, col) to geographic coordinates (lat/lon).

<p align="right">(<a href="#top">back to top</a>)</p>

## 8. resize_dem.py

### 8.1. `unify_dem`

*   **Function**: Unifies an input DEM to match the resolution and coordinate system of a target DEM (supports multi-band). Can also add a buffer.
*   **Potential Issues**:
    *   Floating-point precision errors during coordinate calculations might cause slight variations in column/row counts.
    *   Resampling algorithms (e.g., bilinear) might introduce minor deviations at boundaries.
    *   Pixel alignment issues between old and new coordinate systems.
*   **Note**: Minor irregularities might exist but are generally considered negligible for overall calculations.

### 8.2. `resample_to_target_resolution`

*   **Function**: Simple resampling function suitable for projected coordinate systems. Takes desired resolution in meters as input.

### 8.3. `resample_geography_to_target_resolution`

*   **Function**: Resamples a DEM, potentially transforming from a geographic coordinate system to a projected one before resampling.
*   **⚠️ Warning**: Currently experiencing misalignment issues, possibly due to source data characteristics.

<p align="right">(<a href="#top">back to top</a>)</p>

## 9. splicing_dem.py

### 9.1. `merge_georeferenced_tifs`

*   **Function**: Mosaics multiple GeoTIFF files into a single large GeoTIFF, handling overlapping regions. Requires identical coordinate systems, NoData values, etc.
*   **Implementation Approach**:
    1.  Determine the overall extent of all files to be merged.
    2.  Create a large empty array.
    3.  Populate the array with data from each file, keeping count of overlaps for potential averaging.
*   **Key Considerations**:
    *   **NoData Handling**: Attempts to manage source NoData values to prevent them from affecting calculations (especially in 'mean' strategy).
    *   **Global Bounds & Transform**: Correctly calculates the final mosaic's geographic extent, dimensions, and affine transform matrix.
    *   **⚠️ Dimension Calculation**: Uses `int(round(...))`. While usually fine, floating-point precision mismatches might cause slight dimension discrepancies.
    *   **⚠️ Memory Usage**: Current code uses `ds.read()` to load entire source files at once, which can consume significant memory for large files.

<p align="right">(<a href="#top">back to top</a>)</p>

## 10. split_dem.py

### 10.1. `split_tif`

*   **Function**: Splits a large DEM TIFF file into smaller tiles.
*   **Implementation Details**:
    *   Implements a sliding window logic based on `step = tile_size - overlap` to create overlaps.
    *   **NoData Handling (Initial)**: Attempts to skip tiles where all pixels are NoData or NaN, optimizing by avoiding creation of meaningless files.
*   **Limitation**: Currently supports only single-band rasters.

<p align="right">(<a href="#top">back to top</a>)</p>

## 11. utils.py

### 11.1. `read_tif`

*   **Function**: Successfully reads common raster attributes: pixel values, affine transform, coordinate reference system, and NoData value.

### 11.2. `write_tif`

*   **Function**: Writes data to a GeoTIFF file using provided core information (data, transform, crs, nodata, etc.).

### 11.3. `pixel_to_geo_coords`

*   **Function**: This function converts pixel coordinates to geographic coordinates using GDAL's geotransform parameters.`.

### 11.4. `get_geotransform_and_crs`

*   **Function**: Get geotransform and CRS information from a TIFF file.

<p align="right">(<a href="#top">back to top</a>)</p>