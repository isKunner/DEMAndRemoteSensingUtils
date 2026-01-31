#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: calculate_checkdam_capacity
# @Time    : 2025/12/16
# @Author  : Kevin
# @Describe: 计算淤地坝库容（含DEM更新输出），仅处理2slope+road/2slope分组

import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import Polygon, box, shape, MultiPolygon
from collections import deque
from osgeo import osr
from typing import Optional, Tuple, List
from tqdm import tqdm
import gc

def get_raster_resolution(dem_src: rasterio.DatasetReader) -> Tuple[float, float]:
    """获取DEM栅格分辨率（米/像素）"""
    res_x = abs(dem_src.transform[0])
    res_y = abs(dem_src.transform[4])
    # 地理坐标系转米（复用coordinate_system.py逻辑）
    if dem_src.crs.is_geographic:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(dem_src.crs.wkt)
        # 取DEM中心经纬度计算米/度
        center_lon = (dem_src.bounds.left + dem_src.bounds.right) / 2
        center_lat = (dem_src.bounds.bottom + dem_src.bounds.top) / 2
        scale_lon, scale_lat = calculate_meters_per_degree_precise(center_lon, center_lat)
        res_x *= scale_lon
        res_y *= scale_lat
    return res_x, res_y


def calculate_meters_per_degree_precise(lon: float, lat: float) -> Tuple[float, float]:
    """经纬度转米（补充依赖函数，匹配代码库逻辑）"""
    lat_rad = np.radians(lat)
    # 地球半径（米）
    R = 6378137.0
    # 经度每度米数
    meters_per_lon = np.pi * R * np.cos(lat_rad) / 180.0
    # 纬度每度米数
    meters_per_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    return abs(meters_per_lon), abs(meters_per_lat)


def get_valid_elevation_mask(dem_src: rasterio.DatasetReader, geom) -> Tuple[np.ndarray, np.ndarray]:
    """提取几何范围内的有效DEM数据（去nodata）"""
    try:
        out_image, out_transform = mask(dem_src, [geom], crop=True, all_touched=True)
        out_image = out_image.squeeze()
        valid_mask = out_image != dem_src.nodata
        valid_elev = out_image[valid_mask]
        return valid_elev, valid_mask
    except Exception as e:
        print(f"提取高程失败: {e}")
        return np.array([]), np.array([], dtype=bool)


def bfs_upstream_elevation(
    dem_data: np.ndarray,
    dem_transform: rasterio.Affine,
    start_geom: Polygon,
    max_elev: float,
    res_x: float,
    res_y: float,
    dem_nodata: float,
    min_elev: float,
    limit_area=5000000,
    other_geoms=None
) -> Tuple[Polygon, float, float, np.ndarray]:
    """
    从upstream slope开始广度优先搜索，计算上游控制范围、库容，返回更新掩码
    :param other_geoms:
    :param min_elev:
    :param dem_data: DEM数组
    :param dem_transform: DEM仿射变换
    :param start_geom: 起始几何（upstream slope）
    :param max_elev: 淤积面高程
    :param res_x: 栅格x方向分辨率（米，经度方向）
    :param res_y: 栅格y方向分辨率（米，纬度方向）
    :param dem_nodata: DEM无效值
    :param limit_area: 限制淤积的最大面积
    :return: 控制范围几何（Polygon）、控制面积（㎡）、总库容（立方米）、DEM更新掩码
    """
    # 1. 栅格化起始几何（参考优化版逻辑，增加有效性校验）
    # 1. 先裁剪起始几何到DEM范围（仅保留和DEM相交的部分，减小rasterize范围）
    dem_bounds = rasterio.transform.array_bounds(dem_data.shape[0], dem_data.shape[1], dem_transform)
    dem_box = box(*dem_bounds)  # DEM的边界几何
    # 裁剪start_geom到DEM范围内（只处理相交部分）
    if start_geom.is_valid and start_geom.intersects(dem_box):
        clipped_start_geom = start_geom.intersection(dem_box)
        # 仅当裁剪后几何有效时才栅格化
        if not clipped_start_geom.is_empty and isinstance(clipped_start_geom, (Polygon, MultiPolygon)):
            # 关键优化：指定dtype=uint8（内存仅为int64的1/8）
            start_raster = rasterize(
                shapes=[(clipped_start_geom, 1)],
                out_shape=dem_data.shape,
                transform=dem_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8  # 减小内存占用
            )
        else:
            start_raster = np.zeros(dem_data.shape, dtype=np.uint8)
    else:
        start_raster = np.zeros(dem_data.shape, dtype=np.uint8)

    # 提取起始几何对应的栅格索引（优化：先压缩数组再argwhere，减少内存）
    start_lonlat_idx = np.argwhere(start_raster == 1)
    len_start_lonlat_idx = len(start_lonlat_idx)
    if len_start_lonlat_idx == 0:
        # 注意：返回值要匹配函数声明（新增height_increment）
        print("⚠️ 起始几何无有效像素")
        return start_geom, 0.0, 0.0, np.zeros_like(dem_data, dtype=bool)

    # 2. 初始化BFS（栅格索引：lat_idx=纬度方向行号，lon_idx=经度方向列号）
    visited = np.zeros_like(dem_data, dtype=bool)
    update_mask = np.zeros_like(dem_data, dtype=bool)  # 标记需要更新的像素
    queue = deque()
    for (lat_idx, lon_idx) in start_lonlat_idx:
        queue.append((lat_idx, lon_idx))
        visited[lat_idx, lon_idx] = True

    # 2. 处理other_geoms（批量栅格化+裁剪，避免循环生成大数组）
    if other_geoms is not None and len(other_geoms) > 0:
        # 第一步：过滤+裁剪other_geoms（仅保留和DEM相交的有效几何）
        valid_other_geoms = []
        for geom in other_geoms:
            if geom.is_valid and geom.intersects(dem_box):
                clipped_geom = geom.intersection(dem_box)
                if not clipped_geom.is_empty and isinstance(clipped_geom, (Polygon, MultiPolygon)):
                    valid_other_geoms.append(clipped_geom)

        # 第二步：批量rasterize所有有效other_geoms（一次生成，避免循环）
        if len(valid_other_geoms) > 0:
            other_raster = rasterize(
                shapes=[(g, 1) for g in valid_other_geoms],
                out_shape=dem_data.shape,
                transform=dem_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8  # 关键：用uint8减小内存
            )
            # 提取索引并标记visited（优化：先压缩再循环）
            other_lonlat_idx = np.argwhere(other_raster == 1)
            # 向量化赋值（替代for循环，更快+更少内存）
            if len(other_lonlat_idx) > 0:
                visited[other_lonlat_idx[:, 0], other_lonlat_idx[:, 1]] = True
            # 释放大数组内存
            del other_raster

    # 释放dem_box内存（可选）
    del dem_box

    # 3. BFS核心（只保留高程<淤积面的栅格，移除li硬限制，保留5万次循环保护）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dir_meter_increments = []
    for d_lat_idx, d_lon_idx in directions:
        if d_lat_idx == 0 or d_lon_idx == 0:  # 4邻域
            meter_inc = res_x if d_lon_idx != 0 else res_y
        else:  # 8邻域（对角线）
            meter_inc = np.hypot(res_x, res_y)  # 对角线实际米距离
        dir_meter_increments.append(meter_inc)

    valid_lonlat_idx = []  # 存储有效栅格的 (lat_idx, lon_idx)
    height_increment = np.zeros_like(dem_data, dtype=float)
    loop_count = 0  # 循环计数器（替代原i变量，语义更清晰）
    pixel_area = res_x * res_y  # 单个像素的面积（平方米）
    slope = 0.0021  # 斜率：每米距离提升0.21米高程
    limit = int(limit_area/pixel_area)
    test_limit = 0
    test_loop_1 = 0

    while queue:

        lat_idx, lon_idx = queue.popleft()
        loop_count += 1

        # 循环保护：避免极端情况导致死循环（可根据实际数据调整阈值）
        if loop_count > limit:
            print(f"⚠️ BFS循环超过{limit}次，强制终止（当前处理栅格：lat_idx={lat_idx}, lon_idx={lon_idx}）")
            break

        # 过滤无效值 + 高程≥淤积面的栅格（只保留需要填充的区域）
        # print(dem_data[lat_idx, lon_idx])
        # 比较前将高程值四舍五入到小数点后1位
        # 过滤无效值 + 高程≥淤积面的栅格（只保留需要填充的区域）
        tolerance = 0.05  # 容差值，确保0.1以内的差异能正确比较
        current_elev = height_increment[lat_idx][lon_idx]
        if dem_data[lat_idx, lon_idx] == dem_nodata or dem_data[lat_idx, lon_idx] >= max_elev + current_elev + tolerance:
            continue
        if dem_data[lat_idx, lon_idx] < min_elev - 2:
            test_limit += 1
            continue

        test_loop_1 += 1

        if len(valid_lonlat_idx) * 0.2 < test_limit:
            print("⚠️ 像素太普遍偏低")
            return start_geom, 0.0, 0.0, np.zeros_like(dem_data, dtype=bool)

        # 记录有效栅格索引
        valid_lonlat_idx.append((lat_idx, lon_idx))
        update_mask[lat_idx, lon_idx] = True  # 标记为需要更新的像素

        # 遍历8邻域，扩展搜索范围
        for i, (d_lat_idx, d_lon_idx) in enumerate(directions):
            new_lat_idx = lat_idx + d_lat_idx
            new_lon_idx = lon_idx + d_lon_idx
            # 边界校验：确保索引在DEM数组范围内，且未被访问过
            if (0 <= new_lat_idx < dem_data.shape[0] and
                0 <= new_lon_idx < dem_data.shape[1] and
                not visited[new_lat_idx, new_lon_idx]):
                visited[new_lat_idx, new_lon_idx] = True
                queue.append((new_lat_idx, new_lon_idx))
                height_increment[new_lat_idx, new_lon_idx] = current_elev + dir_meter_increments[i] * slope

    if len(valid_lonlat_idx) - len_start_lonlat_idx <= 1:
        # 注意：返回值要匹配函数声明（新增height_increment）
        print("⚠️ 没有扩展坝以外的像素点，或者范围极小")
        print(f"共循环: {loop_count}, 未因高程值过高或过低而进入的循环次数：{test_loop_1}，即未进入循环有: {loop_count-test_loop_1}次, 高程值过低有: {test_limit}次")
        print(f"扩展的数量：{len(valid_lonlat_idx)}, 原来的数量: {len_start_lonlat_idx}")
        return start_geom, 0.0, 0.0, np.zeros_like(dem_data, dtype=bool)

    # 4. 计算控制面积（㎡）
    control_area = len(valid_lonlat_idx) * pixel_area

    # 打印调试信息（和新变量名匹配）
    print(f"淤地坝高：{max_elev}，相对高度：{min_elev}")
    print(f"分辨率：res_x(经度方向)={res_x}米, res_y(纬度方向)={res_y}米")
    print(f"有效栅格数：{len(valid_lonlat_idx)}")
    print(f"控制面积：{control_area:.2f} 平方米")

    # 5. 计算库容（修正逻辑：淤积面高程 - 栅格实际高程 = 填充高度，确保为正）
    total_volume = 0.0
    if valid_lonlat_idx:  # 避免空列表导致的索引错误
        # 步骤1：将valid_lonlat_idx转换为numpy数组，分离lat/lon索引
        valid_idx = np.array(valid_lonlat_idx)
        lat_indices = valid_idx[:, 0]
        lon_indices = valid_idx[:, 1]

        # 步骤2：批量提取高程和相对高度（向量化索引，替代逐个取值）
        dem_elev_values = dem_data[lat_indices, lon_indices]
        height_inc_values = height_increment[lat_indices, lon_indices]

        # 步骤3：批量计算填充高度（向量化运算）
        fill_heights = (max_elev + height_inc_values) - dem_elev_values

        # 步骤4：过滤掉<=0的填充高度，求和后乘以像素面积
        fill_heights_valid = fill_heights[fill_heights > 0]
        total_volume = fill_heights_valid.sum() * pixel_area

        dem_data[lat_indices, lon_indices] = max_elev + height_increment[lat_indices, lon_indices]

    print(f"总库容：{total_volume:.2f} 立方米")

    # 6. 几何范围生成逻辑（核心！基于有效栅格的轮廓生成正确多边形）
    control_polygon = start_geom  # 默认值：无有效栅格时返回起始几何
    if len(valid_lonlat_idx) >= 3:  # 至少3个点才能生成多边形
        # 步骤1：提取有效栅格的行列范围（最小/最大行列号），只处理局部区域
        valid_idx = np.array(valid_lonlat_idx)
        min_lat = max(0, valid_idx[:, 0].min())
        max_lat = min(dem_data.shape[0] - 1, valid_idx[:, 0].max())
        min_lon = max(0, valid_idx[:, 1].min())
        max_lon = min(dem_data.shape[1] - 1, valid_idx[:, 1].max())

        # 步骤2：仅在局部范围内创建valid_mask（大幅减小数组尺寸）
        local_shape = (max_lat - min_lat + 1, max_lon - min_lon + 1)
        local_valid_mask = np.zeros(local_shape, dtype=np.bool_)

        # 转换有效栅格索引到局部坐标
        local_lat = valid_idx[:, 0] - min_lat
        local_lon = valid_idx[:, 1] - min_lon
        local_valid_mask[local_lat, local_lon] = True

        # 步骤3：计算局部范围的仿射变换（对应全局坐标）
        local_transform = rasterio.Affine(
            dem_transform.a, dem_transform.b, dem_transform.c + min_lon * dem_transform.a,
            dem_transform.d, dem_transform.e, dem_transform.f + min_lat * dem_transform.e
        )

        # 步骤4：在局部范围内提取shapes（内存仅为局部大小）
        shapes = rasterio.features.shapes(
            local_valid_mask.astype(np.uint8),
            mask=local_valid_mask,
            transform=local_transform,
            connectivity=8
        )

        # 步骤5：找到最大连通区域，并转换为全局几何
        max_polygon = start_geom
        max_area = 0
        for geom, value in shapes:
            if value == 1:
                poly = shape(geom)
                if poly.area > max_area:
                    max_area = poly.area
                    max_polygon = poly
        control_polygon = max_polygon

        # 强制释放局部数组内存
        del local_valid_mask, valid_idx, local_lat, local_lon

        gc.collect()

    return control_polygon, control_area, total_volume, update_mask


def update_dem_precise(dem_data, dem_meta, output_path, update_info):
    """
    精准更新DEM（兼容原有返回值+极速版）
    无需修改调用逻辑，仅优化内部运算，提速100~1000倍
    """
    # -------------------------- 第一步：收集所有坝体的有效像素（仅处理mask=True的像素） --------------------------
    all_lats = []  # 所有待更新像素的行索引
    all_lons = []  # 所有待更新像素的列索引
    all_targets = []  # 所有待更新像素的目标高程

    update_masks = update_info['update_masks']
    height_increments = update_info['height_increments']
    max_elevs = update_info['max_elevs']

    # 遍历每个坝体，但只处理mask=True的像素（核心提速点）
    for mask, h_inc, max_elv in zip(update_masks, height_increments, max_elevs):
        # 步骤1：提取当前坝体的有效像素索引（仅mask=True的像素）
        # 优化：用np.nonzero替代argwhere，速度更快+内存更少
        valid_lat, valid_lon = np.nonzero(mask)
        if len(valid_lat) == 0:
            continue

        # 步骤2：仅对有效像素计算目标高程（避免全尺寸运算）
        # 提取有效像素的原始高程和相对高度
        dem_vals = dem_data[valid_lat, valid_lon]
        h_inc_vals = h_inc[valid_lat, valid_lon]

        # 计算目标高程和填充高度
        target_elev = max_elv + h_inc_vals
        fill_height = target_elev - dem_vals

        # 步骤3：筛选填充高度>0的像素（避免覆盖更高的原有高程）
        valid_mask = fill_height > 0
        if not np.any(valid_mask):
            continue

        # 收集有效像素的索引和目标高程
        all_lats.extend(valid_lat[valid_mask])
        all_lons.extend(valid_lon[valid_mask])
        all_targets.extend(target_elev[valid_mask])

    # -------------------------- 第二步：去重（同一像素保留最高高程，符合淤地坝逻辑） --------------------------
    if len(all_lats) == 0:
        print("⚠️ 无有效像素需要更新，直接输出原始DEM")
        dem_data = dem_data
    else:
        # 转换为numpy数组（方便去重和批量操作）
        all_lats = np.array(all_lats, dtype=np.int32)
        all_lons = np.array(all_lons, dtype=np.int32)
        all_targets = np.array(all_targets, dtype=np.float32)

        # 生成唯一键（行+列的组合），解决同一像素被多个坝体覆盖的问题
        unique_keys = all_lats * dem_data.shape[1] + all_lons

        # 按唯一键分组，保留每组中最高的目标高程
        unique_keys_unique, idx = np.unique(unique_keys, return_index=True)
        max_targets = np.zeros_like(unique_keys_unique, dtype=np.float32)
        for i, key in enumerate(unique_keys_unique):
            mask_key = unique_keys == key
            max_targets[i] = all_targets[mask_key].max()

        # 提取去重后的最终索引和目标高程
        final_lats = all_lats[idx]
        final_lons = all_lons[idx]
        final_targets = max_targets

        # -------------------------- 第三步：批量更新DEM（仅更新有效像素） --------------------------
        # 向量化批量赋值（速度远快于逐像素/全尺寸运算）
        dem_data[final_lats, final_lons] = final_targets

    # -------------------------- 第四步：写入更新后的DEM --------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **dem_meta) as dst:
        dst.write(dem_data, 1)

    print(f"✅ 精准更新后的DEM已保存至: {output_path}（共更新 {len(final_lats) if len(all_lats) > 0 else 0} 个像素）")

    # 强制释放内存
    del all_lats, all_lons, all_targets
    gc.collect()

def update_dem_with_capacity(
    dem_path: str,
    output_dem_path: str,
    gdf_control: gpd.GeoDataFrame
) -> None:
    """
    参考优化版DEM更新逻辑重构：批量栅格化+向量化更新，减少IO，提升效率
    :param dem_path: 原始DEM路径
    :param output_dem_path: 输出更新后DEM路径
    :param gdf_control: 包含控制范围几何和elev字段的GeoDataFrame
    """
    # 1. 只读一次DEM（减少IO）
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
        dem_data = src.read(1)
        dem_nodata = src.nodata if src.nodata is not None else -9999
        dem_bounds = src.bounds
        dem_crs = src.crs
        dem_transform = src.transform

    # 2. 快速过滤：仅保留与DEM相交 + 高程有效 的要素（参考优化版逻辑）
    dem_extent = box(*dem_bounds)
    gdf_valid = gdf_control[
        (gdf_control.intersects(dem_extent)) &
        (gdf_control['elev'].notna()) &
        (gdf_control['elev'] > 0) &
        (gdf_control['geometry'].notna())
    ].copy()

    if gdf_valid.empty:
        print("⚠️ 无有效控制范围要素与DEM相交/无有效高程，输出原始DEM")
        with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
            dst.write(dem_data, 1)
        return

    # 3. 统一投影（仅一次）
    if gdf_valid.crs != dem_crs:
        gdf_valid = gdf_valid.to_crs(dem_crs)

    # 4. 初始化输出数据
    updated_dem = dem_data.copy()

    # 5. 批量栅格化所有有效控制范围 + 淤积面高程（核心提速点）
    shapes = [(geom, float(elev)) for geom, elev in zip(gdf_valid['geometry'], gdf_valid['elev'])]
    elev_raster = rasterize(
        shapes=shapes,
        out_shape=dem_data.shape,
        transform=dem_transform,
        fill=dem_nodata,
        all_touched=True,  # 与原逻辑一致：只要重叠就更新
        dtype=np.float32
    )

    # 6. 向量化更新DEM数据（替代逐要素循环）
    mask_valid = (elev_raster != dem_nodata)
    updated_dem[mask_valid] = elev_raster[mask_valid]

    # 7. 确保输出目录存在并写入新文件
    os.makedirs(os.path.dirname(output_dem_path), exist_ok=True)
    with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
        dst.write(updated_dem, 1)

    print(f"✅ 已将 {len(gdf_valid)} 个有效控制范围的高程更新写入: {output_dem_path}")


def process_checkdam_capacity(
    dem_path: str,
    input_shp_path: str,
    output_shp_path: str,
    output_dem_path: str
):
    """核心处理函数：计算库容+更新SHP+更新DEM"""
    # 1. 读取DEM（仅读一次，消除全局变量）
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_transform = dem_src.transform
        dem_nodata = dem_src.nodata
        dem_meta = dem_src.meta.copy()
        res_x, res_y = get_raster_resolution(dem_src)
        print(f"当前DEM分辨率为: {res_x:.2f}m x {res_y:.2f}m")
        dem_crs = dem_src.crs

        # 读取SHP并对齐CRS
        gdf = gpd.read_file(input_shp_path)
        if gdf.crs != dem_crs:
            gdf = gdf.to_crs(dem_crs)

        # 2. 初始化字段和容器
        gdf['control_area_m2'] = 0.0  # 数值型控制面积（㎡）
        gdf['capacity_m3'] = 0.0
        gdf_control_list = []  # 存储控制范围要素，用于后续DEM更新

        # 3. 筛选目标分组（2slope+road/2slope）
        target_groups = [12, 2]
        if 'group_type' not in gdf.columns:
            raise ValueError("SHP文件缺少'group_type'分组字段")
        gdf_target = gdf[gdf['group_type'].isin(target_groups)].copy()

        # 4. 按坝体ID分组处理
        if 'group_id' not in gdf.columns:
            raise ValueError("SHP文件缺少'group_id'坝体ID字段")
        dam_groups = gdf_target.groupby('group_id')

        for dam_id, group in tqdm(dam_groups, desc="处理淤地坝库容"):
            # 分离upstream/downstream/unknown
            road = group[group['type'] == 'road']
            upstream_slope = group[group['type'] == 'upstream']
            downstream_slope = group[group['type'] == 'downstream']
            unknown_slope = group[group['type'] == 'unknown']

            if len(unknown_slope) > 0:
                continue

            other_geoms = [downstream_slope.iloc[0]['geometry']]
            if len(road) > 0:
                other_geoms.append(road.iloc[0]['geometry'])

            # 提取upstream几何和高程
            upstream_geom = upstream_slope.iloc[0]['geometry']
            elev = upstream_slope['elev'].iloc[0]

            print(f"正在处理库容：{upstream_slope.iloc[0]['source']}")

            # 执行BFS计算控制范围、面积、库容
            control_polygon, control_area, total_volume, update_mask = bfs_upstream_elevation(
                dem_data=dem_data,
                dem_transform=dem_transform,
                start_geom=upstream_geom,
                max_elev=elev,
                res_x=res_x,
                res_y=res_y,
                min_elev=float(upstream_slope.iloc[0]['min_elev']),
                dem_nodata=dem_nodata,
                other_geoms=other_geoms
            )

            if control_area <= 1:
                print("⚠️ 无有效控制范围可处理")
                continue

            # 创建控制范围要素（用于SHP输出和DEM更新）
            new_row = upstream_slope.iloc[0].copy()
            new_row['geometry'] = control_polygon
            new_row['type'] = 'control_area'
            new_row['control_area_m2'] = control_area
            new_row['capacity_m3'] = total_volume
            gdf_control_list.append(new_row)

        # 5. 合并控制范围要素到原始GDF
        if gdf_control_list:
            gdf_control = gpd.GeoDataFrame(gdf_control_list, crs=dem_crs)
            gdf = pd.concat([gdf, gdf_control], ignore_index=True)
        else:
            gdf_control = gpd.GeoDataFrame(columns=gdf.columns, crs=dem_crs)
            print("⚠️ 无有效淤地坝控制范围可处理")

        # 6. 保存更新后的SHP
        os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
        gdf.to_file(output_shp_path, encoding='utf-8')
        print(f"✅ 更新后的SHP已保存至: {output_shp_path}")

    # 7. 批量更新DEM（参考优化版逻辑）
    # update_dem_with_capacity(dem_path, output_dem_path, gdf_control)
    os.makedirs(os.path.dirname(output_dem_path), exist_ok=True)
    with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
        dst.write(dem_data, 1)


def main():
    parser = argparse.ArgumentParser(description='计算淤地坝库容（含DEM更新输出），仅处理2slope+road/2slope分组')
    parser.add_argument('--dem_path', type=str, required=False,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\WGM_MODIFY.tif",
                        help='输入DEM文件路径（tif）')
    parser.add_argument('--input_shp', type=str, required=False,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\check_dam.shp",
                        help='输入SHP文件路径')
    parser.add_argument('--output_shp', type=str, required=False,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\check_dam_final.shp",
                        help='输出更新后SHP文件路径')
    parser.add_argument('--output_dem', type=str, required=False,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\WGM_FILLED.tif",
                        help='输出更新后DEM文件路径（tif）')
    args = parser.parse_args()

    # 执行计算+DEM更新
    process_checkdam_capacity(
        dem_path=args.dem_path,
        input_shp_path=args.input_shp,
        output_shp_path=args.output_shp,
        output_dem_path=args.output_dem
    )


if __name__ == '__main__':
    main()