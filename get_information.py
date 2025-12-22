#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: get_information
# @Time    : 2025/7/26 22:27
# @Author  : Kevin
# @Describe: 获取Tif栅格数据的基本属性信息

import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import pyproj
from Logger import LoggerManager

def get_pixel_size_accurate(raster_path, is_print=False):
    logger = LoggerManager.get_logger()

    """
    计算栅格的大小，以m为单位
    :param raster_path:
    :return:
    """

    with rasterio.open(raster_path) as src:
        # 1. 确认数据使用地理坐标系（经纬度）
        if not src.crs.is_geographic:
            pixel_size_m_x = abs(src.transform[0])  # X方向像元大小（米）
            pixel_size_m_y = abs(src.transform[4])
            return pixel_size_m_x, pixel_size_m_y

        # 2. 计算栅格中心经纬度（作为转换基准点）
        center_lon = (src.bounds.left + src.bounds.right) / 2
        center_lat = (src.bounds.top + src.bounds.bottom) / 2

        # 3. 创建从地理坐标系到UTM的转换器（使用中心经纬度确定UTM带）
        utm_zone = int((center_lon + 180) / 6) + 1
        # 判断南北半球
        if center_lat > 0:
            utm_crs = f"EPSG:326{utm_zone}"  # 北半球
        else:
            utm_crs = f"EPSG:327{utm_zone}"  # 南半球

        transformer = Transformer.from_crs(
            src.crs.to_string(),  # 源坐标系（地理坐标系）
            utm_crs,  # 目标坐标系（UTM投影）
            always_xy=True  # 确保经纬度顺序为x=经度，y=纬度
        )

        # 4. 计算像元大小（度）
        pixel_size_deg_x = src.transform[0]  # X方向像元大小（度）
        pixel_size_deg_y = abs(src.transform[4])  # Y方向像元大小（度，取绝对值）

        # 5. 将中心经纬度转换为UTM坐标
        x_center, y_center = transformer.transform(center_lon, center_lat)

        # 6. 计算经纬度方向各偏移1个像元后的UTM坐标
        lon_east = center_lon + pixel_size_deg_x  # 向东偏移1个像元
        lat_north = center_lat + pixel_size_deg_y  # 向北偏移1个像元
        x_east, y_east = transformer.transform(lon_east, center_lat)
        x_north, y_north = transformer.transform(center_lon, lat_north)

        # 7. 计算实际距离（米）
        pixel_size_m_x = abs(x_east - x_center)  # X方向像元大小（米）
        pixel_size_m_y = abs(y_north - y_center)  # Y方向像元大小（米）

        # 计算分辨率比例（检查是否为正方形像素）
        ratio = pixel_size_m_x / pixel_size_m_y

        if is_print:

            logger.info(f"数据中心经纬度：{center_lon:.6f}°, {center_lat:.6f}°")
            logger.info(f"对应UTM带：{'北半球' if center_lat > 0 else '南半球'} Zone {utm_zone}")
            logger.info(f"像元大小（度）：X={pixel_size_deg_x:.10f}°, Y={pixel_size_deg_y:.10f}°")
            logger.info(f"像元大小（米）：X={pixel_size_m_x:.2f}m, Y={pixel_size_m_y:.2f}m")
            logger.info(f"X/Y分辨率比例：{ratio:.4f}（理想值为1，表示正方形像素）")

        return pixel_size_m_x, pixel_size_m_y

def get_tif_latlon_bounds(tif_path):

    """获取TIFF文件的经纬度坐标范围（无论原坐标系如何）"""
    logger = LoggerManager.get_logger()
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # 原始坐标范围，格式为(left, bottom, right, top)
        src_crs = src.crs  # 原始坐标系

        # 判断是否已经是地理坐标系（经纬度），EPSG:4326是WGS84经纬度
        if src_crs.is_geographic or src_crs.to_epsg() == 4326:
            latlon_bounds = bounds
        else:
            # 创建转换器：原始坐标系 → WGS84经纬度
            transformer = Transformer.from_crs(
                src_crs, "EPSG:4326", always_xy=True
            )
            # 转换四个角点的坐标
            left, bottom = transformer.transform(bounds.left, bounds.bottom)
            right, top = transformer.transform(bounds.right, bounds.top)
            latlon_bounds = (left, bottom, right, top)

        logger.info(f"经纬度范围 (lon_min, lat_min, lon_max, lat_max): {latlon_bounds}")
        return latlon_bounds

def get_crs_transformer(src_crs, dst_crs="EPSG:4326"):

    """
    Create a coordinate reference system (CRS) transformer for converting coordinates between two CRS.

    This function creates a pyproj.Transformer object that can transform coordinates from
    source coordinate system to destination coordinate system. It handles both string-based
    EPSG codes and pyproj.CRS objects as inputs.

    # Example 1: Create transformer from WGS84 to UTM Zone 49N
    transformer = get_crs_transformer("EPSG:4326", "EPSG:32649")

    # Input coordinates (Beijing, China in WGS84)
    lon, lat = 116.3975, 39.9085

    # Transform to UTM coordinates
    if transformer:
        x, y = transformer.transform(lon, lat)
        print(f"Input (WGS84): {lon}, {lat}")
        print(f"Output (UTM Zone 49N): {x:.2f}, {y:.2f}")
        # Output: Input (WGS84): 116.3975, 39.9085
        #         Output (UTM Zone 49N): 437000.00, 4410000.00

    # Example 2: Create transformer from UTM to WGS84
    transformer2 = get_crs_transformer("EPSG:32649", "EPSG:4326")

    # Input UTM coordinates
    utm_x, utm_y = 437000.00, 4410000.00

    # Transform back to geographic coordinates
    if transformer2:
        lon2, lat2 = transformer2.transform(utm_x, utm_y)
        print(f"Input (UTM Zone 49N): {utm_x}, {utm_y}")
        print(f"Output (WGS84): {lon2:.4f}, {lat2:.4f}")
        # Output: Input (UTM Zone 49N): 437000.00, 4410000.00
        #         Output (WGS84): 116.3975, 39.9085

    # Example 3: Same CRS - returns None
    transformer3 = get_crs_transformer("EPSG:4326", "EPSG:4326")
    print(transformer3)  # Output: None

    :param src_crs: Source coordinate reference system (can be EPSG code string or pyproj.CRS object)
    :param dst_crs: Destination coordinate reference system (default: "EPSG:4326" for WGS84 geographic)
    :return: pyproj.Transformer object if CRS systems are different, None if they are identical
    """

    if src_crs == dst_crs:
        return None

    if isinstance(src_crs, str):
        src_crs = pyproj.CRS(src_crs)
    if isinstance(dst_crs, str):
        dst_crs = pyproj.CRS(dst_crs)

    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)

def geo_to_pixel(target_src, lon, lat, is_cv: bool=False, input_src: str="EPSG:4326"):

    """
    Convert geographic coordinates to pixel coordinates in the source raster
    Handles both geographic and projected coordinate systems automatically.

    :param target_src: rasterio dataset object
    :param lon: longitude in WGS84 coordinate system
    :param lat: latitude in WGS84 coordinate system  
    :param is_cv: if True, returns (col, row) for OpenCV; if False, returns (row, col) for numpy/PIL
        When is_cv=False (default): Returns coordinates in (row, col) format, which is used by numpy arrays and PIL images
        When is_cv=True: Returns coordinates in (col, row) format, which is used by OpenCV
    :param input_src: source coordinate reference system of input coordinates (default: "EPSG:4326" for WGS84 geographic) Can be EPSG code string or pyproj.CRS object
    :return: pixel coordinates (row, col) or (col, row) depending on is_cv flag
    """

    logger = LoggerManager.get_logger()

    # Get the coordinate reference system of the source raster
    target_crs = target_src.crs

    # Transform WGS84 coordinates to the source raster's coordinate system
    if target_crs.to_string() != input_src:
        transformer = get_crs_transformer(input_src, target_crs)
        if transformer:
            target_x, target_y = transformer.transform(lon, lat)
        else:
            target_x, target_y = lon, lat
    else:
        target_x, target_y = lon, lat

    # Convert coordinates to pixel positions
    row, col = rowcol(target_src.transform, target_x, target_y)

    logger.info(f"Source CRS: {target_crs.to_string()} - Input coordinates: ({lon}, {lat}) → Target coordinates: ({target_x}, {target_y}) → Pixel coordinates: ({col}, {row})")

    if is_cv:
        return col, row

    return row, col

def pixel_to_geo(input_src, row, col, target_src: str= "EPSG:4326"):
    logger = LoggerManager.get_logger()
    """
    Convert pixel coordinates to geographic coordinates (longitude, latitude)
    
    Handles both geographic and projected source coordinate systems automatically.
    
    :param src: rasterio dataset object
    :param row: pixel row coordinate
    :param col: pixel column coordinate
    :param target_src: target coordinate reference system (default: "EPSG:4326" for WGS84 geographic)
    :return: tuple of (longitude, latitude) in target coordinate system
    """
    # Get the source coordinate reference system
    src_crs = input_src.crs

    # Calculate coordinates in the source's coordinate system (could be geographic or projected)
    coord_x, coord_y = input_src.transform * (col, row)

    # Transform to target geographic coordinates
    if src_crs.to_string() == target_src:
        lon, lat = coord_x, coord_y  # Already in target coordinates
    else:
        transformer = get_crs_transformer(src_crs, target_src)
        if transformer:
            lon, lat = transformer.transform(coord_x, coord_y)  # Transform from source to target
        else:
            lon, lat = coord_x, coord_y  # Fallback handling

    logger.info(f"Source CRS: {src_crs.to_string()} - Target CRS: {target_src} - Pixel coordinates: ({col}, {row}) → Source coordinates: ({coord_x}, {coord_y}) → Target coordinates: ({lon}, {lat})")

    return lon, lat  # Always return coordinates in target system

def pixel_to_pixel(src, row, col, target_src, is_cv: bool = False):
    """
    Convert pixel coordinates from source raster to pixel coordinates in target raster

    This function combines pixel_to_geo and geo_to_pixel to enable direct pixel-to-pixel
    coordinate transformation between two rasters with different coordinate systems.

    :param src: Source rasterio dataset object (for input pixel coordinates)
    :param row: Source pixel row coordinate
    :param col: Source pixel column coordinate
    :param target_src: Target rasterio dataset object (for output pixel coordinates)
    :param is_cv: if True, returns (col, row) for OpenCV; if False, returns (row, col) for numpy/PIL
    :return: pixel coordinates (row, col) or (col, row) depending on is_cv flag in target raster
    """

    # Step 1: Convert source pixel coordinates to geographic coordinates
    lon, lat = pixel_to_geo(src, row, col)

    # Step 2: Convert geographic coordinates to target raster pixel coordinates
    target_coords = geo_to_pixel(target_src, lon, lat, is_cv=is_cv)

    return target_coords
