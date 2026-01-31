#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: get_information
# @Time    : 2025/7/26 22:27
# @Author  : Kevin
# @Describe: 获取Tif栅格数据的基本属性信息

import numpy as np

from rasterio.transform import rowcol
import rasterio
from rasterio.enums import Resampling
from pyproj import Transformer
import pyproj
from rasterio.warp import transform_bounds


def decimal_to_dms(decimal_degree, is_latitude=True):
    """
    将十进制度数转换为度分秒(DMS)格式
    :param decimal_degree: 十进制度数值
    :param is_latitude: 是否为纬度(True=纬度N/S, False=经度E/W)
    :return: 度分秒字符串，如 39°54'36.50"N
    """
    direction = ''
    if decimal_degree is None:
        return "N/A"

    if is_latitude:
        direction = 'N' if decimal_degree >= 0 else 'S'
    else:
        direction = 'E' if decimal_degree >= 0 else 'W'

    decimal_degree = abs(float(decimal_degree))
    degrees = int(decimal_degree)
    minutes_float = (decimal_degree - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60

    return f"{degrees}°{minutes:02d}'{seconds:05.2f}\"{direction}"


def get_tif_info(tif_path):
    """
    获取Tif栅格数据的基本属性信息（包含经纬度范围）
    :param tif_path: TIF文件路径
    :return: 包含TIF基本信息的字典
    """

    with rasterio.open(tif_path) as src:
        info = {}

        # 原有基本信息获取代码...
        info['width'] = src.width
        info['height'] = src.height
        info['count'] = src.count
        info['dtype'] = src.dtypes[0]
        info['nodata'] = src.nodata
        info['crs'] = src.crs.to_string() if src.crs else None
        info['bounds'] = {
            'left': src.bounds.left,
            'right': src.bounds.right,
            'top': src.bounds.top,
            'bottom': src.bounds.bottom
        }

        # ========== 新增：经纬度范围计算 ==========
        if src.crs:
            try:
                # 转换为 WGS84 (EPSG:4326) 地理坐标
                if not src.crs.is_geographic:
                    # 如果是投影坐标，先转换
                    west, south, east, north = transform_bounds(
                        src.crs, 'EPSG:4326',
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top
                    )
                else:
                    # 已经是地理坐标
                    west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

                # 存储十进制经纬度
                info['lon_min'] = west  # 最小经度 (西)
                info['lon_max'] = east  # 最大经度 (东)
                info['lat_min'] = south  # 最小纬度 (南)
                info['lat_max'] = north  # 最大纬度 (北)

                # 转换为度分秒格式
                info['lon_min_dms'] = decimal_to_dms(west, is_latitude=False)
                info['lon_max_dms'] = decimal_to_dms(east, is_latitude=False)
                info['lat_min_dms'] = decimal_to_dms(south, is_latitude=True)
                info['lat_max_dms'] = decimal_to_dms(north, is_latitude=True)

            except Exception as e:
                info['lon_min'] = info['lon_max'] = info['lat_min'] = info['lat_max'] = None
                info['coordinate_error'] = str(e)

        # 原有统计信息计算...
        stats = {}
        for i in range(1, src.count + 1):
            band_data = src.read(i, out_shape=(max(1, int(src.height / 10)), max(1, int(src.width / 10))),
                                 resampling=Resampling.bilinear)

            if src.nodata is not None:
                valid_data = band_data[band_data != src.nodata]
            else:
                valid_data = band_data[np.isfinite(band_data)]

            if len(valid_data) > 0:
                stats[f'band_{i}'] = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                }
        info['statistics'] = stats

        # ========== 修改后的输出部分 ==========
        print(f"\n{'=' * 50}")
        print(f"TIF文件: {tif_path}")
        print(f"{'=' * 50}")
        print(f"尺寸: {info['width']} x {info['height']} 像素")
        print(f"数据类型: {info['dtype']} | 波段数: {info['count']}")
        print(f"坐标系: {info['crs']}")

        # 新增：经纬度范围输出
        if 'lon_min' in info and info['lon_min'] is not None:
            print(f"\n【经纬度范围】")
            print(f"  经度 (Longitude):")
            print(f"    最小: {info['lon_min']:.6f}°  ({info['lon_min_dms']})")
            print(f"    最大: {info['lon_max']:.6f}°  ({info['lon_max_dms']})")
            print(f"  纬度 (Latitude):")
            print(f"    最小: {info['lat_min']:.6f}°  ({info['lat_min_dms']})")
            print(f"    最大: {info['lat_max']:.6f}°  ({info['lat_max_dms']})")

        if 'band_1' in stats:
            print(f"\n波段1统计: 最小={stats['band_1']['min']:.2f}, "
                  f"最大={stats['band_1']['max']:.2f}, "
                  f"均值={stats['band_1']['mean']:.2f}")
        print(f"{'=' * 50}\n")

        return info



def get_pixel_size_accurate(raster_path, is_print=False):

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

            print(f"数据中心经纬度：{center_lon:.6f}°, {center_lat:.6f}°")
            print(f"对应UTM带：{'北半球' if center_lat > 0 else '南半球'} Zone {utm_zone}")
            print(f"像元大小（度）：X={pixel_size_deg_x:.10f}°, Y={pixel_size_deg_y:.10f}°")
            print(f"像元大小（米）：X={pixel_size_m_x:.2f}m, Y={pixel_size_m_y:.2f}m")
            print(f"X/Y分辨率比例：{ratio:.4f}（理想值为1，表示正方形像素）")

        return pixel_size_m_x, pixel_size_m_y


def get_degree_per_meter(center_lon, center_lat, is_print=False):
    """
    计算1米距离对应的经纬度变化量（度）
    :param center_lon: 中心点经度
    :param center_lat: 中心点纬度
    :param is_print: 是否打印详细信息
    :return: (lon_degree_per_meter, lat_degree_per_meter) - 每米对应的经度和纬度变化量（度）
    """

    # 1. 创建从地理坐标系到UTM的转换器（使用中心经纬度确定UTM带）
    utm_zone = int((center_lon + 180) / 6) + 1

    # 判断南北半球
    if center_lat > 0:
        utm_crs = f"EPSG:326{utm_zone}"  # 北半球
    else:
        utm_crs = f"EPSG:327{utm_zone}"  # 南半球

    # 从UTM转换回地理坐标的逆向转换器
    transformer_to_geo = Transformer.from_crs(
        utm_crs,  # 源坐标系（UTM投影）
        "EPSG:4326",  # 目标坐标系（地理坐标系）
        always_xy=True  # 确保经纬度顺序为x=经度，y=纬度
    )

    # 从地理坐标系到UTM的转换器
    transformer_to_utm = Transformer.from_crs(
        "EPSG:4326",  # 源坐标系（地理坐标系）
        utm_crs,  # 目标坐标系（UTM投影）
        always_xy=True
    )

    # 2. 将中心经纬度转换为UTM坐标
    x_center, y_center = transformer_to_utm.transform(center_lon, center_lat)

    # 3. 在UTM坐标系中移动1米
    x_east = x_center + 1  # 向东移动1米
    y_north = y_center + 1  # 向北移动1米

    # 4. 将移动后的UTM坐标转换回地理坐标
    lon_east, lat_center = transformer_to_geo.transform(x_east, y_center)  # 东西方向变化
    lon_center, lat_north = transformer_to_geo.transform(x_center, y_north)  # 南北方向变化

    # 5. 计算每米对应的经纬度变化量（度）
    lon_degree_per_meter = abs(lon_east - center_lon)  # 经度方向每米变化量
    lat_degree_per_meter = abs(lat_north - center_lat)  # 纬度方向每米变化量

    if is_print:
        print(f"数据中心经纬度：{center_lon:.6f}°, {center_lat:.6f}°")
        print(f"对应UTM带：{'北半球' if center_lat > 0 else '南半球'} Zone {utm_zone}")
        print(f"1米距离对应的经纬度变化：")
        print(f"  经度变化：{lon_degree_per_meter:.10f}°/m")
        print(f"  纬度变化：{lat_degree_per_meter:.10f}°/m")
        print(f"  东西方向：1° ≈ {1/lon_degree_per_meter:.2f} 米")
        print(f"  南北方向：1° ≈ {1/lat_degree_per_meter:.2f} 米")

    return lon_degree_per_meter, lat_degree_per_meter


def get_tif_latlon_bounds(tif_path):

    """获取TIFF文件的经纬度坐标范围（无论原坐标系如何）"""
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

        print(f"经纬度范围 (lon_min, lat_min, lon_max, lat_max): {latlon_bounds}")
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

    print(f"Source CRS: {target_crs.to_string()} - Input coordinates: ({lon}, {lat}) → Target coordinates: ({target_x}, {target_y}) → Pixel coordinates: ({col}, {row})")

    if is_cv:
        return col, row

    return row, col

def pixel_to_geo(input_src, row, col, target_src: str= "EPSG:4326"):
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

    print(f"Source CRS: {src_crs.to_string()} - Target CRS: {target_src} - Pixel coordinates: ({col}, {row}) → Source coordinates: ({coord_x}, {coord_y}) → Target coordinates: ({lon}, {lat})")

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

if __name__ == '__main__':
    # 获取TIF信息
    get_tif_info(r'D:\研究文件\ResearchData\USA\GoogleRemoteSensing\GeoDAR_v11_dams_of_USA_group1\33.tif')
    get_tif_info(r'D:\研究文件\ResearchData\USA\GoogleRemoteSensing\GeoDAR_v11_dams_of_USA_group1\130.tif')
    get_tif_info(r'D:\研究文件\ResearchData\USA\GoogleRemoteSensing\GeoDAR_v11_dams_of_USA_group1\191.tif')
    get_tif_info(r'D:\研究文件\ResearchData\USA\GoogleRemoteSensing\GeoDAR_v11_dams_of_USA_group1\221.tif')
    get_tif_info(r'D:\研究文件\ResearchData\USA\GoogleRemoteSensing\GeoDAR_v11_dams_of_USA_group1\245.tif')