#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: utils
# @Time    : 2025/8/16 09:03
# @Author  : Kevin
# @Describe:
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from typing import List, Tuple, Optional

def pixel_to_geo_coords(x_pixel: float, y_pixel: float, geotransform: List[float]) -> Tuple[float, float]:

    """
    This function converts pixel coordinates to geographic coordinates using GDAL's geotransform parameters.
    Transforms image pixel coordinates (x_pixel, y_pixel) into real-world geographic coordinates (geo_x, geo_y).
    Uses the affine transformation model based on GDAL's geotransform matrix.
    Essential for converting labeled image annotations to georeferenced spatial data.
    The pixel_to_geo_coords function is suitable for both geographic coordinate systems and projected coordinate systems.
    :param x_pixel: The column index (x-coordinate) in pixel space
    :param y_pixel: The row index (y-coordinate) in pixel space
    :param geotransform:  A 6-element list containing GDAL geotransform coefficients
    :return: Returns a tuple (geo_x, geo_y) representing the corresponding geographic coordinates
    """

    geo_x = geotransform[0] + x_pixel * geotransform[1] + y_pixel * geotransform[2]
    geo_y = geotransform[3] + x_pixel * geotransform[4] + y_pixel * geotransform[5]
    return geo_x, geo_y

def get_geotransform_and_crs(tif_path: str) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Get geotransform and CRS information from a TIFF file
    :param tif_path: LocalPath to the TIFF file
    :return: Tuple of (geotransform, crs_wkt) or (None, None) if failed
    """
    try:
        with rasterio.open(tif_path) as src:
            # Get geotransform as tuple and convert to list
            geotransform = list(src.transform.to_gdal())
            # Get CRS as WKT string
            crs_wkt = src.crs.to_wkt() if src.crs else None
            return geotransform, crs_wkt
    except Exception:
        return None, None

def calculate_meters_per_degree_precise(lon, lat, delta=0.00001):
    """
    使用UTM投影精确计算给定经纬度处1度对应的米数

    Args:
        lat: 纬度值（度）
        lon: 经度值（度）
        delta: 微小角度偏移量

    Returns:
        tuple: (经度1度对应米数, 纬度1度对应米数)
    """

    # 根据经度判断UTM分区
    if lon < 108:
        utm_zone = 48  # Zone 48N
    else:
        utm_zone = 49  # Zone 49N

    # 构造UTM投影坐标系 (WGS84 UTM)
    utm_crs = CRS.from_string(f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs")
    wgs84_crs = CRS.from_epsg(4326)  # WGS84地理坐标系

    # 创建坐标转换器
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    # 计算经度方向的尺度因子
    x1, y1 = transformer.transform(lon - delta/2, lat)
    x2, y2 = transformer.transform(lon + delta/2, lat)
    dist_x = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    scale_lon = dist_x / delta  # 米/度

    # 计算纬度方向的尺度因子
    x1, y1 = transformer.transform(lon, lat - delta/2)
    x2, y2 = transformer.transform(lon, lat + delta/2)
    dist_y = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    scale_lat = dist_y / delta  # 米/度

    return scale_lon, scale_lat



def read_tif(path):
    """读取TIFF数据及地理信息（替代gdal实现）"""
    with rasterio.open(path) as src:
        # 读取数据数组
        data = src.read(1)  # 读取第一个波段
        # 获取地理变换信息（仿射矩阵）
        geotrans = src.transform
        # 获取投影信息
        proj = src.crs.to_wkt()  # 转换为WKT格式，与gdal输出格式一致
        # 获取无数据值
        nodata = src.nodata if src.nodata is not None else -9999.0
    return data, geotrans, proj, nodata

def write_tif(save_path, data, geotrans, proj, nodata_value=-9999.0):
    """保存TIFF文件（替代gdal实现）"""
    # 获取数据形状
    rows, cols = data.shape

    # 设置TIFF文件的元数据
    profile = {
        'driver': 'GTiff',
        'width': cols,
        'height': rows,
        'count': 1,  # 波段数
        'dtype': data.dtype,
        'crs': proj,
        'transform': geotrans,
        'nodata': nodata_value,
        'compress': 'lzw'  # 启用压缩，可根据需要修改
    }

    # 写入数据
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(data, 1)  # 写入第一个波段