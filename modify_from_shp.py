#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: modify_from_shp
# @Time    : 2025/12/16 21:43
# @Author  : Kevin
# @Describe:

import math
import os
import cv2
import argparse
import logging
import numpy as np
import pandas as pd

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Polygon, box

from coordinate_system import get_shp_bounds
from crop_dem_from_cordinate import crop_tif_by_bounds
from utils import calculate_meters_per_degree_precise


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("crop_dem_by_shp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def dict_from_row(row):
    return row.to_dict()


def update_gdf_from_dict(gdf, idx, data_dict):
    for k, v in data_dict.items():
        gdf.at[idx, k] = v


def extend_dimension(obj, dimension_type, extend_m, scale_lon, scale_lat):
    """通用扩展函数：扩展 width 或 height（米+度）"""
    angle_deg = obj['angle_deg']
    if dimension_type == 'width':
        angle_rad = math.radians(angle_deg)
    else:  # height
        angle_rad = math.radians(angle_deg + 90)

    delta_lon_m = extend_m * math.cos(angle_rad)
    delta_lat_m = extend_m * math.sin(angle_rad)
    delta_lon_deg = delta_lon_m / scale_lon
    delta_lat_deg = delta_lat_m / scale_lat

    obj[f'{dimension_type}_m'] += extend_m
    obj[f'{dimension_type}_deg'] += math.sqrt(delta_lon_deg**2 + delta_lat_deg**2)


def adjust_slope_dimension(slope, ref_obj, dimension_type, min_distance, scale_lon, scale_lat):
    current_value = slope[f'{dimension_type}_m']
    if current_value >= min_distance:
        return

    increase = min_distance - current_value
    slope[f'{dimension_type}_m'] = min_distance

    # 方向选择
    road_to_slope_vector = (
        slope['center_lon'] - ref_obj['center_lon'],
        slope['center_lat'] - ref_obj['center_lat']
    )

    if dimension_type == 'width':
        angles = [slope['angle_deg'], slope['angle_deg'] + 180]
    else:
        angles = [slope['angle_deg'] + 90, slope['angle_deg'] + 270]

    best_angle = None
    max_dot = -float('inf')
    for a in angles:
        rad = math.radians(a)
        dir_vec = (math.cos(rad), math.sin(rad))
        dot = road_to_slope_vector[0] * dir_vec[0] + road_to_slope_vector[1] * dir_vec[1]
        if dot > max_dot:
            max_dot = dot
            best_angle = rad

    delta_lon_m = increase * math.cos(best_angle)
    delta_lat_m = increase * math.sin(best_angle)
    delta_lon_deg = delta_lon_m / scale_lon
    delta_lat_deg = delta_lat_m / scale_lat

    slope[f'{dimension_type}_deg'] += math.sqrt(delta_lon_deg**2 + delta_lat_deg**2)
    slope['center_lon'] += delta_lon_deg / 2
    slope['center_lat'] += delta_lat_deg / 2


def update_geometry_from_params(obj):
    rect = ((obj['center_lon'], obj['center_lat']), (obj['width_deg'], obj['height_deg']), obj['angle_deg'])
    box_points = cv2.boxPoints(rect)
    closed_points = np.vstack([box_points, box_points[0]])
    obj['geometry'] = Polygon(closed_points)


def process_with_direction(obj, other_objs, extend_m, min_distance):
    """
    根据 width > height 判断主方向，统一处理扩展和最小距离调整
    """
    scale_lon, scale_lat = calculate_meters_per_degree_precise(obj['center_lon'], obj['center_lat'])

    if obj['width_m'] > obj['height_m']:
        if len(other_objs)==2:
            if abs(other_objs[0]['height_m']/1.75-other_objs[1]['height_m']/1.5) < abs(other_objs[0]['height_m']/1.5-other_objs[1]['height_m']/1.75):
                other_objs[0]['rel_h'] = other_objs[0]['height_m']/1.75
                other_objs[1]['rel_h'] = other_objs[1]['height_m']/1.5
            else:
                other_objs[0]['rel_h'] = other_objs[0]['height_m']/1.5
                other_objs[1]['rel_h'] = other_objs[1]['height_m']/1.75
        elif len(other_objs)==1:
            other_objs[0]['rel_h'] = other_objs[0]['height_m']/1.5
        extend_dimension(obj, 'width', extend_m, scale_lon, scale_lat)
        dim_to_adjust = 'height'
    else:
        if len(other_objs)==2:
            if abs(other_objs[0]['width_m']/1.75-other_objs[1]['width_m']/1.5) < abs(other_objs[0]['width_m']/1.5-other_objs[1]['width_m']/1.75):
                other_objs[0]['rel_h'] = other_objs[0]['width_m']/1.75
                other_objs[1]['rel_h'] = other_objs[1]['width_m']/1.5
            else:
                other_objs[0]['rel_h'] = other_objs[0]['width_m']/1.5
                other_objs[1]['rel_h'] = other_objs[1]['width_m']/1.75
        elif len(other_objs)==1:
            other_objs[0]['rel_h'] = other_objs[0]['width_m']/1.5

        extend_dimension(obj, 'height', extend_m, scale_lon, scale_lat)
        dim_to_adjust = 'width'

    for other in other_objs:
        other_scale_lon, other_scale_lat = calculate_meters_per_degree_precise(other['center_lon'], other['center_lat'])
        adjust_slope_dimension(other, obj, dim_to_adjust, min_distance, other_scale_lon, other_scale_lat)

    update_geometry_from_params(obj)

    for other in other_objs:
        update_geometry_from_params(other)


def process_two_slopes_mutual(slope1, slope2, extend_m=10, min_distance=30):

    center1 = (slope1['center_lon'], slope1['center_lat'])
    center2 = (slope2['center_lon'], slope2['center_lat'])
    delta_lon = center2[0] - center1[0]
    delta_lat = center2[1] - center1[1]
    vector_angle = math.degrees(math.atan2(delta_lat, delta_lon)) % 360
    slope_angle = slope1['angle_deg']
    angle_diff = min(abs(vector_angle - slope_angle), 180 - abs(vector_angle - slope_angle))

    scale1_lon, scale1_lat = calculate_meters_per_degree_precise(*center1)
    scale2_lon, scale2_lat = calculate_meters_per_degree_precise(*center2)

    if angle_diff > 45:  # 更接近 0° → 沿 width 扩展
        if abs(slope1['width_m'] / 1.75 - slope2['width_m'] / 1.5) < abs(
                slope1['width_m'] / 1.5 - slope2['width_m'] / 1.75):
            slope1['rel_h'] = slope1['width_m'] / 1.75
            slope2['rel_h'] = slope2['width_m'] / 1.5
        else:
            slope1['rel_h'] = slope1['width_m'] / 1.5
            slope2['rel_h'] = slope2['width_m'] / 1.75
        extend_dimension(slope1, 'width', extend_m, scale1_lon, scale1_lat)
        extend_dimension(slope2, 'width', extend_m, scale2_lon, scale2_lat)
        adjust_slope_dimension(slope1, slope2, 'height', min_distance, scale1_lon, scale1_lat)
        adjust_slope_dimension(slope2, slope1, 'height', min_distance, scale2_lon, scale2_lat)
    else:  # 更接近 90° → 沿 height 扩展
        if abs(slope1['height_m'] / 1.75 - slope2['height_m'] / 1.5) < abs(
                slope1['height_m'] / 1.5 - slope2['height_m'] / 1.75):
            slope1['rel_h'] = slope1['height_m'] / 1.75
            slope2['rel_h'] = slope2['height_m'] / 1.5
        else:
            slope1['rel_h'] = slope1['height_m'] / 1.5
            slope2['rel_h'] = slope2['height_m'] / 1.75
        extend_dimension(slope1, 'height', extend_m, scale1_lon, scale1_lat)
        extend_dimension(slope2, 'height', extend_m, scale2_lon, scale2_lat)
        adjust_slope_dimension(slope1, slope2, 'width', min_distance, scale1_lon, scale1_lat)
        adjust_slope_dimension(slope2, slope1, 'width', min_distance, scale2_lon, scale2_lat)

    update_geometry_from_params(slope1)
    update_geometry_from_params(slope2)


def extract_min_elevation_from_dem(gdf, dem_path):
    """
    从DEM数据中提取每个几何图形区域的最小高程值

    Args:
        gdf: GeoDataFrame，包含几何图形
        dem_path: DEM文件路径

    Returns:
        list: 每个几何图形对应的最小高程值
    """
    import rasterio
    from rasterio.mask import mask
    import numpy as np

    min_elevations = []

    with rasterio.open(dem_path) as dem_src:
        for idx, row in gdf.iterrows():
            try:
                # 使用几何图形裁剪DEM数据
                out_image, out_transform = mask(dem_src, [row['geometry']], crop=True)
                # 提取有效数据（去除nodata值）
                valid_data = out_image[out_image != dem_src.nodata]
                if len(valid_data) > 0:
                    min_elevation = np.min(valid_data)
                else:
                    min_elevation = None
                min_elevations.append(min_elevation)
            except Exception as e:
                print(f"处理索引 {idx} 时出错: {e}")
                min_elevations.append(None)

    return min_elevations

def update_dem_with_elevation_values(gdf, dem_path, output_dem_path):
    """
    基于 gdf 中的几何和 'elev' 字段，生成一个新的 DEM 文件：
      - 只要 DEM 像元与多边形有任意重叠（all_touched=True），就更新为指定高程
      - 自动跳过范围外或无效要素
      - 输出到新文件，不修改原 DEM
    """
    # 1. 读取原始 DEM 元数据
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
        dem_data = src.read(1)
        dem_nodata = src.nodata if src.nodata is not None else -9999
        dem_bounds = src.bounds
        dem_crs = src.crs

    # 2. 构建 DEM 范围（用于快速过滤）
    dem_extent = box(*dem_bounds)

    # 3. 过滤并投影 gdf
    gdf_valid = gdf[gdf.intersects(dem_extent)].copy()
    if gdf_valid.empty:
        print("⚠️ 无有效要素与 DEM 相交，输出原始 DEM。")
        # 可选：是否仍输出原始 DEM？这里选择输出
        with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
            dst.write(dem_data, 1)
        return

    if gdf_valid.crs != dem_crs:
        gdf_valid = gdf_valid.to_crs(dem_crs)

    # 4. 初始化输出数据（从原始 DEM 复制）
    updated_data = dem_data.copy()

    # 5. 遍历每个有效要素，更新相交像元
    with rasterio.open(dem_path) as src:  # 重新打开以便 mask 使用
        for idx, row in gdf_valid.iterrows():
            elev = row.get('elev')
            if pd.isna(elev) or elev <= 0:
                continue

            geom = row['geometry']
            try:
                # 关键：all_touched=True → 像元只要碰到多边形边界就算
                masked, _ = mask(src, [geom], crop=False, all_touched=True, nodata=dem_nodata)
                intersect_mask = masked[0] != dem_nodata
                updated_data[intersect_mask] = float(elev)
            except Exception as e:
                print(f"⚠️ 要素 {idx} 处理失败: {e}")
                continue

    # 6. 写入新文件
    with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
        dst.write(updated_data, 1)

    print(f"✅ 已将 {len(gdf_valid)} 个要素的高程更新写入: {output_dem_path}")


def process(args, logger: logging.Logger):

    crop_dem_by_shapefile_bounds(
        shp_path=args.shp_path,
        input_tif=args.input_tif,
        output_tif=args.output_tif,
        buffer_distance=args.buffer,
        logger=logger
    )

    gdf = gpd.read_file(args.shp_path)

    extend = args.extend
    distance = args.min_width

    for group_id, group_gdf in gdf.groupby('group_id'):
        label_counts = group_gdf['label'].value_counts()
        slope_count = label_counts.get('slope', 0)
        road_count = label_counts.get('road', 0)

        if slope_count == 2 and road_count == 1:
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            road = dict_from_row(group_gdf.loc[road_idx])
            s1, s2 = dict_from_row(group_gdf.loc[slope_idxs[0]]), dict_from_row(group_gdf.loc[slope_idxs[1]])
            process_with_direction(road, [s1, s2], extend_m=extend, min_distance=distance)
            update_gdf_from_dict(gdf, road_idx, road)
            update_gdf_from_dict(gdf, slope_idxs[0], s1)
            update_gdf_from_dict(gdf, slope_idxs[1], s2)

        elif slope_count == 1 and road_count == 1:
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            road = dict_from_row(group_gdf.loc[road_idx])
            slope = dict_from_row(group_gdf.loc[slope_idx])
            process_with_direction(road, [slope], extend_m=extend, min_distance=distance)
            update_gdf_from_dict(gdf, road_idx, road)
            update_gdf_from_dict(gdf, slope_idx, slope)

        elif slope_count == 2 and road_count == 0:
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            s1 = dict_from_row(group_gdf.loc[slope_idxs[0]])
            s2 = dict_from_row(group_gdf.loc[slope_idxs[1]])
            process_two_slopes_mutual(s1, s2, extend_m=extend, min_distance=distance)
            update_gdf_from_dict(gdf, slope_idxs[0], s1)
            update_gdf_from_dict(gdf, slope_idxs[1], s2)

        elif slope_count == 1 and road_count == 0:
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            slope = dict_from_row(group_gdf.loc[slope_idx])
            # 单坡度：沿主方向扩展，并确保另一维度 ≥ min_distance
            scale_lon, scale_lat = calculate_meters_per_degree_precise(slope['center_lon'], slope['center_lat'])
            if slope['width_m'] > slope['height_m']:
                slope['rel_h'] = slope['height_m'] / 1.5
                extend_dimension(slope, 'width', extend, scale_lon, scale_lat)
                if slope['height_m'] < distance * 2:
                    extend_dimension(slope, 'height', distance * 2 - slope['height_m'], scale_lon, scale_lat)
            else:
                slope['rel_h'] = slope['width_m'] / 1.5
                extend_dimension(slope, 'height', extend, scale_lon, scale_lat)
                if slope['width_m'] < distance * 2:
                    extend_dimension(slope, 'width', distance * 2 - slope['width_m'], scale_lon, scale_lat)
            update_geometry_from_params(slope)
            update_gdf_from_dict(gdf, slope_idx, slope)

    min_elevations = extract_min_elevation_from_dem(gdf, args.output_tif)

    # 添加最小高程列到GeoDataFrame
    gdf['min_elev'] = min_elevations

    # 根据原始数据计算高程值
    for group_id, group_gdf in gdf.groupby('group_id'):
        label_counts = group_gdf['label'].value_counts()
        slope_count = label_counts.get('slope', 0)
        road_count = label_counts.get('road', 0)

        if slope_count == 2 and road_count == 1:
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            r = dict_from_row(group_gdf.loc[road_idx])

            # 从更新后的gdf中获取min_elev和rel_h值
            s1_min_elev = gdf.loc[slope_idxs[0], 'min_elev']
            s2_min_elev = gdf.loc[slope_idxs[1], 'min_elev']
            s1_rel_h = gdf.loc[slope_idxs[0], 'rel_h']
            s2_rel_h = gdf.loc[slope_idxs[1], 'rel_h']

            # 计算道路的高程值
            r['elev'] = 0.5 * (s1_min_elev + s1_rel_h + s2_min_elev + s2_rel_h)

            # 更新回gdf
            update_gdf_from_dict(gdf, road_idx, r)

        elif slope_count == 1 and road_count == 1:
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            r = dict_from_row(group_gdf.loc[road_idx])

            # 从更新后的gdf中获取min_elev和rel_h值
            s_min_elev = gdf.loc[slope_idx, 'min_elev']
            s_rel_h = gdf.loc[slope_idx, 'rel_h']

            # 计算道路的高程值
            r['elev'] = s_min_elev + s_rel_h

            # 更新回gdf
            update_gdf_from_dict(gdf, road_idx, r)

        elif slope_count == 2 and road_count == 0:
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            s1 = dict_from_row(group_gdf.loc[slope_idxs[0]])
            s2 = dict_from_row(group_gdf.loc[slope_idxs[1]])

            # 从更新后的gdf中获取min_elev和rel_h值
            s1_min_elev = gdf.loc[slope_idxs[0], 'min_elev']
            s2_min_elev = gdf.loc[slope_idxs[1], 'min_elev']
            s1_rel_h = gdf.loc[slope_idxs[0], 'rel_h']
            s2_rel_h = gdf.loc[slope_idxs[1], 'rel_h']

            # 计算两个坡度的高程值
            elev_value = 0.5 * (s1_min_elev + s1_rel_h + s2_min_elev + s2_rel_h)
            s1['elev'] = elev_value
            s2['elev'] = elev_value
            # 更新回gdf
            update_gdf_from_dict(gdf, slope_idxs[0], s1)
            update_gdf_from_dict(gdf, slope_idxs[1], s2)

        elif slope_count == 1 and road_count == 0:
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            s = dict_from_row(group_gdf.loc[slope_idx])
            # 从更新后的gdf中获取min_elev和rel_h值
            s_min_elev = gdf.loc[slope_idx, 'min_elev']
            s_rel_h = gdf.loc[slope_idx, 'rel_h']
            # 计算坡度的高程值
            s['elev'] = s_min_elev + s_rel_h
            # 更新回gdf
            update_gdf_from_dict(gdf, slope_idx, s)

    gdf.to_file(r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\test.shp")

    update_dem_with_elevation_values(gdf, args.output_tif, r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\final.tif")


def crop_dem_by_shapefile_bounds(shp_path: str, input_tif: str, output_tif: str, buffer_distance: float, logger: logging.Logger):
    try:
        lon_min, lat_min, lon_max, lat_max = get_shp_bounds(shp_path)
        logger.info(f"原始范围 -> 经度: [{lon_min:.6f}, {lon_max:.6f}], 纬度: [{lat_min:.6f}, {lat_max:.6f}]")
        os.makedirs(os.path.dirname(output_tif), exist_ok=True)
        logger.info("调用 crop_tif_by_bounds 进行裁剪...")
        crop_tif_by_bounds(input_tif, output_tif, lon_min, lat_min, lon_max, lat_max, buffer_distance)
        logger.info("✅ DEM 裁剪任务成功完成！")
    except Exception as e:
        logger.exception(f"❌ 裁剪过程中发生错误: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="根据 Shapefile 范围裁剪 DEM TIF（带缓冲区）")
    parser.add_argument("--shp_path", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\check_dam_struct.shp")
    parser.add_argument("--input_tif", type=str, default=r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif")
    parser.add_argument("--output_tif", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\test.tif")
    parser.add_argument("--buffer", type=float, default=2.0)
    parser.add_argument("--log", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\crop_dem_log.txt")
    parser.add_argument("--extend", type=int, default=20)
    parser.add_argument("--min-width", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logger = setup_logger(args.log)

    logger.info("=== 开始执行 DEM 裁剪任务 ===")
    for name in ["Shapefile", "输入 DEM", "输出 DEM", "缓冲区"]:
        logger.info(f"{name}: {getattr(args, {'Shapefile':'shp_path','输入 DEM':'input_tif','输出 DEM':'output_tif','缓冲区':'buffer'}.get(name, ''))}")

    if not os.path.exists(args.shp_path):
        logger.error("Shapefile 不存在！"); return
    if not os.path.exists(args.input_tif):
        logger.error("输入 DEM TIF 不存在！"); return

    process(args, logger)


if __name__ == "__main__":
    main()