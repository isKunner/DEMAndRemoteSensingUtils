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
import numpy as np

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Polygon, box

from .get_flow_accumulation import calculate_flow_accumulation
from .coordinate_system import get_shp_bounds
from .crop_dem_from_cordinate import crop_tif_by_bounds
from .utils import calculate_meters_per_degree_precise


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
        return False

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

    return True

def update_geometry_from_params(obj, target_geo_fields: list|str):
    rect = ((obj['center_lon'], obj['center_lat']), (obj['width_deg'], obj['height_deg']), obj['angle_deg'])
    box_points = cv2.boxPoints(rect)
    closed_points = np.vstack([box_points, box_points[0]])
    if type(target_geo_fields)==str:
        obj[target_geo_fields] = Polygon(closed_points)
    else:
        for target_geo_field in target_geo_fields:
            obj[target_geo_field] = Polygon(closed_points)


def process_with_direction(obj, other_objs, extend_m, min_distance_flow, min_distance_elev):
    """
    根据 width > height 判断主方向，统一处理扩展和最小距离调整
    """
    scale_lon, scale_lat = calculate_meters_per_degree_precise(obj['center_lon'], obj['center_lat'])

    direction_dict = {0: "height", 1:"width"}
    direction = obj['width_m'] >= obj['height_m']
    obj['direction'] = direction_dict[direction]
    extend_dimension(obj, direction_dict[direction], extend_m, scale_lon, scale_lat)

    for other in other_objs:
        extend_dimension(other, direction_dict[direction], extend_m, scale_lon, scale_lat)
        dem_other = other.copy()
        other['direction'] = direction_dict[direction]
        other_scale_lon, other_scale_lat = calculate_meters_per_degree_precise(other['center_lon'], other['center_lat'])
        if adjust_slope_dimension(dem_other, obj, direction_dict[1 - direction], min_distance_elev, other_scale_lon, other_scale_lat):
            update_geometry_from_params(dem_other, "geo_for_elev")
            other['geo_for_elev'] = dem_other['geo_for_elev']
        del dem_other
        dem_other = other.copy()
        if adjust_slope_dimension(dem_other, obj, direction_dict[1 - direction], min_distance_flow, other_scale_lon, other_scale_lat):
            update_geometry_from_params(dem_other, "geo_for_flow")
            other['geo_for_flow'] = dem_other['geo_for_flow']
        del dem_other


    update_geometry_from_params(obj, ['geometry', 'geo_for_elev', 'geo_for_flow'])


def process_two_slopes_mutual(slope1, slope2, extend_m=10, min_distance_flow=30, min_distance_elev=10):
    center1 = (slope1['center_lon'], slope1['center_lat'])
    center2 = (slope2['center_lon'], slope2['center_lat'])
    delta_lon = center2[0] - center1[0]
    delta_lat = center2[1] - center1[1]
    vector_angle = math.degrees(math.atan2(delta_lat, delta_lon)) % 180
    slope_angle = slope1['angle_deg']
    angle_diff = min(abs(vector_angle - slope_angle), 180 - abs(vector_angle - slope_angle))

    scale1_lon, scale1_lat = calculate_meters_per_degree_precise(*center1)
    scale2_lon, scale2_lat = calculate_meters_per_degree_precise(*center2)

    direction_dict = {0: "height", 1: "width"}
    direction = angle_diff > 45
    slope1['direction'] = direction_dict[direction]
    slope2['direction'] = direction_dict[direction]

    extend_dimension(slope1, direction_dict[direction], extend_m, scale1_lon, scale1_lat)
    extend_dimension(slope2, direction_dict[direction], extend_m, scale2_lon, scale2_lat)
    update_geometry_from_params(slope1, ['geometry', 'geo_for_elev', 'geo_for_flow'])
    update_geometry_from_params(slope2, ['geometry', 'geo_for_elev', 'geo_for_flow'])

    slope1_temp = slope1.copy()
    slope2_temp = slope2.copy()

    if adjust_slope_dimension(slope1, slope2, direction_dict[1 - direction], min_distance_flow, scale1_lon, scale1_lat):
        update_geometry_from_params(slope1, "geo_for_flow")
    if adjust_slope_dimension(slope2, slope1, direction_dict[1 - direction], min_distance_flow, scale2_lon, scale2_lat):
        update_geometry_from_params(slope2, "geo_for_flow")

    if adjust_slope_dimension(slope1_temp, slope2_temp, direction_dict[1 - direction], min_distance_elev, scale1_lon, scale1_lat):
        update_geometry_from_params(slope1_temp, "geo_for_elev")
        slope1['geo_for_elev'] = slope1_temp['geo_for_elev']
    if adjust_slope_dimension(slope2_temp, slope1_temp, direction_dict[1 - direction], min_distance_elev, scale2_lon, scale2_lat):
        update_geometry_from_params(slope2_temp, "geo_for_elev")
        slope2['geo_for_elev'] = slope2_temp['geo_for_elev']

    del slope1_temp, slope2_temp


def extract_elevation_from_dem(gdf, dem_path, geo_field, type='min'):
    """
    从DEM数据中提取每个几何图形区域的最小（大）的值
    Args:
        gdf: GeoDataFrame，包含几何图形
        dem_path: DEM文件路径
    Returns:
        list: 每个几何图形对应的最小（大）的值
    """
    target_elevations = []
    with rasterio.open(dem_path) as dem_src:
        for idx, row in gdf.iterrows():
            try:
                out_image, out_transform = mask(dem_src, [row[geo_field]], crop=True, all_touched=True)
                valid_data = out_image[out_image != dem_src.nodata]
                if len(valid_data) > 0:
                    if type == 'min':
                        target_elevation = np.min(valid_data)
                    elif type == 'max':
                        target_elevation = np.max(valid_data)
                else:
                    target_elevation = None
                target_elevations.append(target_elevation)
            except Exception as e:
                print(f"处理索引 {idx} 时出错: {e}")
                target_elevations.append(None)
    return target_elevations


def update_dem_with_elevation_values(gdf, dem_path, output_dem_path):
    """
    基于 gdf 中的几何和 'elev' 字段，生成一个新的 DEM 文件（优化版）：
      - 批量栅格化几何，替代逐要素mask，大幅提速
      - 自动跳过范围外或无效要素
      - 输出到新文件，不修改原 DEM
    """
    # 1. 只读一次DEM（复用句柄，减少IO）
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
        dem_data = src.read(1)
        dem_nodata = src.nodata if src.nodata is not None else -9999
        dem_bounds = src.bounds
        dem_crs = src.crs
        transform = src.transform  # 保存仿射变换，用于栅格化

    # 2. 快速过滤：仅保留与DEM相交 + 高程有效 的要素
    dem_extent = box(*dem_bounds)
    gdf_valid = gdf[
    ~((gdf['group_type'].isin([11, 12])) & (gdf['label'] == 'slope')) &
    (gdf.intersects(dem_extent)) &
    (gdf['elev'].notna()) &
    (gdf['elev'] > 0)
    ].copy()


    if gdf_valid.empty:
        print("⚠️ 无有效要素与 DEM 相交/无有效高程，输出原始 DEM。")
        with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
            dst.write(dem_data, 1)
        return

    # 3. 统一投影（仅一次）
    if gdf_valid.crs != dem_crs:
        gdf_valid = gdf_valid.to_crs(dem_crs)

    # 4. 初始化输出数据
    updated_data = dem_data.copy()

    # 5. 批量栅格化所有有效几何 + 高程（核心提速点）
    # 构造 (几何, 高程) 元组列表
    shapes = [(geom, float(elev)) for geom, elev in zip(gdf_valid['geometry'], gdf_valid['elev'])]

    # 批量生成高程掩码（与DEM同尺寸）
    # all_touched=True 等价于原逻辑的all_touched
    elev_raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=dem_data.shape,
        transform=transform,
        fill=dem_nodata,  # 无几何覆盖的区域填充nodata
        all_touched=True,  # 与原逻辑一致：只要重叠就更新
        dtype=np.float32
    )

    # 6. 向量化更新DEM数据（替代逐要素循环）
    # 仅更新：有有效高程掩码 + DEM原值 < 目标高程 的区域
    mask_valid = (elev_raster != dem_nodata) & (updated_data < elev_raster)
    # mask_valid = (elev_raster != dem_nodata)
    updated_data[mask_valid] = elev_raster[mask_valid]

    # 7. 写入新文件
    with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
        dst.write(updated_data, 1)

    print(f"✅ 已将 {len(gdf_valid)} 个有效要素的高程更新写入: {output_dem_path}")


# ===================== 步骤1：裁剪DEM =====================
def step1_crop_dem(args):
    """步骤1：根据Shapefile范围裁剪DEM（带缓冲区）"""
    crop_dem_by_shapefile_bounds(
        shp_path=args.shp_path,
        input_tif=args.input_tif,
        output_tif=args.output_tif,
        buffer_distance=args.buffer
    )


def crop_dem_by_shapefile_bounds(shp_path: str, input_tif: str, output_tif: str, buffer_distance: float):
    try:
        lon_min, lat_min, lon_max, lat_max = get_shp_bounds(shp_path)
        print(f"原始范围 -> 经度: [{lon_min:.6f}, {lon_max:.6f}], 纬度: [{lat_min:.6f}, {lat_max:.6f}]")
        os.makedirs(os.path.dirname(output_tif), exist_ok=True)
        print("调用 crop_tif_by_bounds 进行裁剪...")
        crop_tif_by_bounds(input_tif, output_tif, lon_min, lat_min, lon_max, lat_max, buffer_distance)
        print("✅ DEM 裁剪任务成功完成！")
    except Exception as e:
        print(f"❌ 裁剪过程中发生错误: {e}")
        raise


# ===================== 步骤2：边缘扩展 =====================
def step2_edge_extension(gdf, args):
    gdf['o_width_m'] = gdf['width_m'].copy()
    gdf['o_width_deg'] = gdf['width_deg'].copy()
    gdf['o_height_m'] = gdf['height_m'].copy()
    gdf['o_height_deg'] = gdf['height_deg'].copy()
    gdf['geo_for_elev'] = gdf['geometry'].copy()  # 高程计算用几何
    gdf['geo_for_flow'] = gdf['geometry'].copy()  # 流量计算用几何

    gdf['type'] = None
    gdf['direction'] = None  # 0=height, 1=width
    gdf['angle_diff'] = None  # 仅two slopes无road场景
    gdf['rel_h'] = None

    extend = args.extend
    min_distance_elev = args.min_width_elev
    min_distance_flow = args.min_width_flow # 流量累计用最小宽度（30m分辨率）

    for group_id, group_gdf in gdf.groupby('group_id'):
        label_counts = group_gdf['label'].value_counts()
        slope_count = label_counts.get('slope', 0)
        road_count = label_counts.get('road', 0)

        if slope_count == 2 and road_count == 1:
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            road = dict_from_row(group_gdf.loc[road_idx])
            s1, s2 = dict_from_row(group_gdf.loc[slope_idxs[0]]), dict_from_row(group_gdf.loc[slope_idxs[1]])
            process_with_direction(road, [s1, s2], extend_m=extend, min_distance_elev=min_distance_elev, min_distance_flow=min_distance_flow)
            update_gdf_from_dict(gdf, road_idx, road)
            update_gdf_from_dict(gdf, slope_idxs[0], s1)
            update_gdf_from_dict(gdf, slope_idxs[1], s2)

        elif slope_count == 1 and road_count == 1:
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            road = dict_from_row(group_gdf.loc[road_idx])
            slope = dict_from_row(group_gdf.loc[slope_idx])
            process_with_direction(road, [slope], extend_m=extend, min_distance_elev=min_distance_elev, min_distance_flow=min_distance_flow)
            update_gdf_from_dict(gdf, road_idx, road)
            update_gdf_from_dict(gdf, slope_idx, slope)

        elif slope_count == 2 and road_count == 0:
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            s1 = dict_from_row(group_gdf.loc[slope_idxs[0]])
            s2 = dict_from_row(group_gdf.loc[slope_idxs[1]])
            process_two_slopes_mutual(s1, s2, extend_m=extend, min_distance_elev=min_distance_elev, min_distance_flow=min_distance_flow)
            update_gdf_from_dict(gdf, slope_idxs[0], s1)
            update_gdf_from_dict(gdf, slope_idxs[1], s2)

        elif slope_count == 1 and road_count == 0:
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            slope = dict_from_row(group_gdf.loc[slope_idx])
            # 单坡度：沿主方向扩展，并确保另一维度 ≥ min_distance
            scale_lon, scale_lat = calculate_meters_per_degree_precise(slope['center_lon'], slope['center_lat'])
            direction_dict = {0: "height", 1: "width"}
            direction_m_dict = {0: "height_m", 1: "width_m"}
            direction = slope['width_m'] > slope['height_m']
            slope["direction"] = direction_dict[direction]

            extend_dimension(slope, direction_dict[direction], extend, scale_lon, scale_lat)
            update_geometry_from_params(slope, ['geometry', 'geo_for_elev', 'geo_for_flow'])

            slope_temp = slope.copy()

            if slope[direction_m_dict[1-direction]] < min_distance_elev * 2:
                extend_dimension(slope, direction_dict[1-direction], min_distance_elev * 2 - slope[direction_m_dict[1-direction]],
                                 scale_lon, scale_lat)
                update_geometry_from_params(slope, 'geo_for_elev')

            if slope[direction_m_dict[1-direction]] < min_distance_flow * 2:
                extend_dimension(slope_temp, direction_dict[1-direction], min_distance_flow * 2 - slope[direction_m_dict[1-direction]],
                                 scale_lon, scale_lat)
                update_geometry_from_params(slope_temp, 'geo_for_flow')
                slope['geo_for_flow'] = slope_temp['geo_for_flow']

            del slope_temp
            update_gdf_from_dict(gdf, slope_idx, slope)
    return gdf


# ===================== 步骤3：DEM和流量计算 =====================
def step3_extract_elev_and_flow(gdf, args):
    """步骤3：提取DEM最小高程 + 计算流量累积 + 提取流量最大高程"""
    # 提取DEM最小高程
    min_elevations = extract_elevation_from_dem(gdf, args.elev_tif, geo_field='geo_for_elev')
    gdf['min_elev'] = min_elevations

    # 计算流量累积
    calculate_flow_accumulation(args.output_tif, args.output_tif_flow_accumulation)

    # 提取流量累积最大值
    max_flow_accum = extract_elevation_from_dem(gdf, args.output_tif_flow_accumulation, geo_field='geo_for_flow', type='max')
    gdf['flow_accum'] = max_flow_accum
    return gdf


# ===================== 步骤4：坝坡相对高度计算 =====================
def step4_calculate_rel_height(gdf):
    """步骤4：按group_id计算坝坡相对高度（type + rel_h）"""
    direction_dict = {0: "height", 1: "width"}
    dict_direction = {"height": 0, "width": 1}
    direction_m_dict = {0: "height_m", 1: "width_m"}
    ratio = {'downstream': 1.5, 'upstream': 1.75, 'unknown': 1.4}

    for group_id, group_gdf in gdf.groupby('group_id'):
        label_counts = group_gdf['label'].value_counts()
        slope_count = label_counts.get('slope', 0)
        road_count = label_counts.get('road', 0)

        # 场景1: 2 slopes + 1 road
        if slope_count == 2 and road_count == 1:
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            s1_idx, s2_idx = slope_idxs
            s1_mfa = gdf.loc[s1_idx, 'flow_accum']
            s2_mfa = gdf.loc[s2_idx, 'flow_accum']

            # 赋值type
            if s1_mfa > s2_mfa:
                gdf.at[s1_idx, 'type'] = 'downstream'
                gdf.at[s2_idx, 'type'] = 'upstream'
            elif s1_mfa < s2_mfa:
                gdf.at[s1_idx, 'type'] = 'upstream'
                gdf.at[s2_idx, 'type'] = 'downstream'
            else:
                gdf.at[s1_idx, 'type'] = 'unknown'
                gdf.at[s2_idx, 'type'] = 'unknown'

            # 计算rel_h（road的direction决定维度）
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            direction = dict_direction[gdf.loc[road_idx, 'direction']]
            gdf.at[road_idx, 'type'] = 'road'
            for s_idx in slope_idxs:
                slope_type = gdf.loc[s_idx, 'type']
                rel_h_dim = direction_m_dict[1 - direction]  # 1-direction是另一维度
                gdf.at[s_idx, 'rel_h'] = gdf.loc[s_idx, f"o_{rel_h_dim}"] / ratio[slope_type]

        # 场景2: 1 slope + 1 road
        elif slope_count == 1 and road_count == 1:
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            # 赋值type
            gdf.at[slope_idx, 'type'] = 'downstream'

            # 计算rel_h
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            gdf.at[road_idx, 'type'] = 'road'
            direction = dict_direction[gdf.loc[road_idx, 'direction']]
            slope_type = gdf.loc[slope_idx, 'type']
            rel_h_dim = direction_m_dict[1 - direction]
            gdf.at[slope_idx, 'rel_h'] = gdf.loc[slope_idx, f"o_{rel_h_dim}"] / ratio[slope_type]

        # 场景3: 2 slopes + 0 road
        elif slope_count == 2 and road_count == 0:
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            s1_idx, s2_idx = slope_idxs
            s1_mfa = gdf.loc[s1_idx, 'flow_accum']
            s2_mfa = gdf.loc[s2_idx, 'flow_accum']

            # 赋值type
            if s1_mfa > s2_mfa:
                gdf.at[s1_idx, 'type'] = 'downstream'
                gdf.at[s2_idx, 'type'] = 'upstream'
            elif s1_mfa < s2_mfa:
                gdf.at[s1_idx, 'type'] = 'upstream'
                gdf.at[s2_idx, 'type'] = 'downstream'
            else:
                gdf.at[s1_idx, 'type'] = 'unknown'
                gdf.at[s2_idx, 'type'] = 'unknown'

            # 计算rel_h（依赖angle_diff）
            angle_diff = gdf.loc[s1_idx, 'angle_diff']
            direction = dict_direction[gdf.loc[s1_idx, 'direction']]
            for s_idx in slope_idxs:
                slope_type = gdf.loc[s_idx, 'type']
                rel_h_dim = direction_m_dict[1-direction]
                gdf.at[s_idx, 'rel_h'] = gdf.loc[s_idx, f"o_{rel_h_dim}"] / ratio[slope_type]

        # 场景4: 1 slope + 0 road
        elif slope_count == 1 and road_count == 0:
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            # 赋值type
            gdf.at[slope_idx, 'type'] = 'downstream'

            # 计算rel_h
            direction = dict_direction[gdf.loc[slope_idx, 'direction']]
            slope_type = gdf.loc[slope_idx, 'type']
            rel_h_dim = direction_m_dict[1 - direction]  # 另一维度
            gdf.at[slope_idx, 'rel_h'] = gdf.loc[slope_idx, f"o_{rel_h_dim}"] / ratio[slope_type]
    return gdf


# ===================== 步骤5：高程值计算 =====================
def step5_calculate_elev_value(gdf):
    """步骤5：按group_id计算要素高程值（elev字段）"""
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
            update_gdf_from_dict(gdf, road_idx, r)
            gdf.loc[slope_idxs, 'elev'] = r['elev']

        elif slope_count == 1 and road_count == 1:
            road_idx = group_gdf[group_gdf['label'] == 'road'].index[0]
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            r = dict_from_row(group_gdf.loc[road_idx])

            # 从更新后的gdf中获取min_elev和rel_h值
            s_min_elev = gdf.loc[slope_idx, 'min_elev']
            s_rel_h = gdf.loc[slope_idx, 'rel_h']

            # 计算道路的高程值
            r['elev'] = s_min_elev + s_rel_h
            update_gdf_from_dict(gdf, road_idx, r)
            gdf.loc[slope_idx, 'elev'] = r['elev']

        elif slope_count == 2 and road_count == 0:
            slope_idxs = group_gdf[group_gdf['label'] == 'slope'].index.tolist()
            s1 = dict_from_row(group_gdf.loc[slope_idxs[0]])
            s2 = dict_from_row(group_gdf.loc[slope_idxs[1]])

            s1_min_elev = gdf.loc[slope_idxs[0], 'min_elev']
            s2_min_elev = gdf.loc[slope_idxs[1], 'min_elev']
            s1_rel_h = gdf.loc[slope_idxs[0], 'rel_h']
            s2_rel_h = gdf.loc[slope_idxs[1], 'rel_h']

            # 计算两个坡度的高程值
            elev_value = 0.5 * (s1_min_elev + s1_rel_h + s2_min_elev + s2_rel_h)
            s1['elev'] = elev_value
            s2['elev'] = elev_value
            update_gdf_from_dict(gdf, slope_idxs[0], s1)
            update_gdf_from_dict(gdf, slope_idxs[1], s2)

        elif slope_count == 1 and road_count == 0:
            slope_idx = group_gdf[group_gdf['label'] == 'slope'].index[0]
            s = dict_from_row(group_gdf.loc[slope_idx])
            s_min_elev = gdf.loc[slope_idx, 'min_elev']
            s_rel_h = gdf.loc[slope_idx, 'rel_h']
            s['elev'] = s_min_elev + s_rel_h
            update_gdf_from_dict(gdf, slope_idx, s)
    return gdf


# ===================== 主流程控制 =====================
def check_dam_info_extract(args):
    # 读取原始Shapefile
    gdf = gpd.read_file(args.shp_path)

    # 步骤1：裁剪DEM
    step1_crop_dem(args)

    # 步骤2：边缘扩展
    gdf = step2_edge_extension(gdf, args)

    # 步骤3：DEM最小高程 + 流量累积计算
    gdf = step3_extract_elev_and_flow(gdf, args)

    # 步骤4：坝坡相对高度计算
    gdf = step4_calculate_rel_height(gdf)

    # 步骤5：高程值计算
    gdf = step5_calculate_elev_value(gdf)

    # 更新DEM高程值
    update_dem_with_elevation_values(gdf, args.elev_tif, args.modified_tif)

    # 保存更新后的Shapefile
    if args.mode==1:
        gdf.drop(columns=['geo_for_flow', 'geo_for_elev'], errors='ignore').to_file(args.output_shp_path)
    elif args.mode==2:
        gdf.drop(columns=['geometry', 'geo_for_elev'], errors='ignore').to_file(args.output_shp_path)
    elif args.mode==3:
        gdf.drop(columns=['geometry', 'geo_for_flow'], errors='ignore').to_file(args.output_shp_path)


def main():
    parser = argparse.ArgumentParser(description="根据 Shapefile 范围裁剪 DEM TIF（带缓冲区）")
    parser.add_argument("--shp_path", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\check_dam_struct.shp")
    parser.add_argument("--output_shp_path", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\check_dam.shp")
    parser.add_argument("--input_tif", type=str, default=r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif")
    parser.add_argument("--output_tif", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\check_dam.tif")
    parser.add_argument("--output_tif_flow_accumulation", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\WMG_ACCUM.tif")
    parser.add_argument("--elev_tif", type=str, default=r"C:\Users\Kevin\Documents\ResearchData\WangMao\cleaned_dem.tif")
    parser.add_argument("--modified_tif", type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SlopeExtraction\final_dam.tif")
    parser.add_argument("--buffer", type=float, default=2.0)
    parser.add_argument("--extend", type=int, default=100)
    parser.add_argument("--min-width-flow", type=int, default=100, help="流量累计用最小宽度")
    parser.add_argument("--min-width-elev", type=int, default=5, help="高程计算用最小宽度")
    parser.add_argument("--mode", type=int, default=1, help="用于设置最终保存多边形的类型")

    args = parser.parse_args()
    if args.elev_tif is None:
        args.elev_tif = args.output_tif

    # 测试路径覆盖（可根据需要注释）
    args.shp_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\check_dam_struct.shp"
    args.output_shp_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\check_dam.shp"
    args.output_tif = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\check_dam.tif"
    args.output_tif_flow_accumulation = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\WMG_ACCUM.tif"
    args.modified_tif = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\WGM_MODIFY.tif"

    # 执行核心流程
    check_dam_info_extract(args)


if __name__ == "__main__":
    main()