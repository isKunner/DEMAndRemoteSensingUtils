#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: crop_shp_from_shp
# @Time    : 2026/1/24 22:44
# @Author  : Kevin
# @Describe:

import os
import geopandas as gpd
import time


def clip_points_by_boundary(points_shp_path, boundary_shp_path, output_shp_path):
    """
    根据边界多边形裁剪点数据

    Args:
        points_shp_path (str): 点数据SHP文件路径（全球水坝数据）
        boundary_shp_path (str): 边界多边形SHP文件路径（美国州边界）
        output_shp_path (str): 输出SHP文件路径

    Returns:
        str: 输出文件路径
    """
    start_time = time.time()
    print(f"开始处理: {os.path.basename(points_shp_path)}")

    # 读取点数据（水坝数据）
    print("读取点数据...")
    points_gdf = gpd.read_file(points_shp_path)
    print(f"  原始点数据包含 {len(points_gdf)} 个水坝点")

    # 读取边界数据（美国州边界）
    print("读取边界数据...")
    boundary_gdf = gpd.read_file(boundary_shp_path)
    print(f"  边界数据包含 {len(boundary_gdf)} 个州/区域")

    # 检查坐标系，如果不一致则转换
    if not points_gdf.crs.equals(boundary_gdf.crs):
        print(f"坐标系不一致，将点数据转换为边界数据的坐标系: {boundary_gdf.crs}")
        points_gdf = points_gdf.to_crs(boundary_gdf.crs)

    # 创建边界多边形的联合（union）以提高处理效率
    print("创建边界多边形的联合...")
    boundary_union = boundary_gdf.unary_union

    # 筛选位于边界内的点
    print("筛选位于美国境内的水坝点...")
    points_within_usa = points_gdf[points_gdf.geometry.within(boundary_union)].copy()

    print(f"  筛选后保留 {len(points_within_usa)} 个水坝点")
    print(f"  移除了 {len(points_gdf) - len(points_within_usa)} 个不在美国境内的水坝点")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_shp_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存结果
    print(f"保存结果到: {output_shp_path}")
    points_within_usa.to_file(output_shp_path)

    # 计算处理时间
    elapsed_time = time.time() - start_time
    print(f"处理完成，耗时: {elapsed_time:.2f} 秒")

    return output_shp_path


def clip_points_by_boundary_with_state_info(points_shp_path, boundary_shp_path, output_shp_path):
    """
    根据边界多边形裁剪点数据，并保留州信息

    Args:
        points_shp_path (str): 点数据SHP文件路径（全球水坝数据）
        boundary_shp_path (str): 边界多边形SHP文件路径（美国州边界）
        output_shp_path (str): 输出SHP文件路径

    Returns:
        str: 输出文件路径
    """
    start_time = time.time()
    print(f"开始处理: {os.path.basename(points_shp_path)}")

    # 读取点数据（水坝数据）
    print("读取点数据...")
    points_gdf = gpd.read_file(points_shp_path)
    print(f"  原始点数据包含 {len(points_gdf)} 个水坝点")

    # 读取边界数据（美国州边界）
    print("读取边界数据...")
    boundary_gdf = gpd.read_file(boundary_shp_path)
    print(f"  边界数据包含 {len(boundary_gdf)} 个州/区域")

    # 检查是否包含必要的字段
    required_fields = ['NAME', 'REGION', 'DIVISION']
    available_fields = [field for field in required_fields if field in boundary_gdf.columns]
    print(f"  边界数据中可用的字段: {available_fields}")

    # 检查坐标系，如果不一致则转换
    if not points_gdf.crs.equals(boundary_gdf.crs):
        print(f"坐标系不一致，将点数据转换为边界数据的坐标系: {boundary_gdf.crs}")
        points_gdf = points_gdf.to_crs(boundary_gdf.crs)

    # 执行空间连接，将点与边界多边形关联
    print("执行空间连接，确定每个水坝点所属的州...")
    joined_gdf = gpd.sjoin(points_gdf, boundary_gdf[['NAME', 'REGION', 'DIVISION', 'geometry']],
                           how='inner', predicate='within')

    print(f"  空间连接后保留 {len(joined_gdf)} 个水坝点")
    print(f"  移除了 {len(points_gdf) - len(joined_gdf)} 个不在美国境内的水坝点")

    # 统计各州的水坝数量
    if 'NAME' in joined_gdf.columns:
        state_counts = joined_gdf['NAME'].value_counts()
        print("\n各州水坝数量统计（前10个州）:")
        for state, count in state_counts.head(10).items():
            print(f"  {state}: {count} 个水坝")
        print(f"  总共 {len(state_counts)} 个州有水坝数据")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_shp_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存结果
    print(f"\n保存结果到: {output_shp_path}")
    # 移除不必要的列，如index_right
    if 'index_right' in joined_gdf.columns:
        joined_gdf = joined_gdf.drop(columns=['index_right'])

    joined_gdf.to_file(output_shp_path)

    # 计算处理时间
    elapsed_time = time.time() - start_time
    print(f"\n处理完成，耗时: {elapsed_time:.2f} 秒")

    return output_shp_path


if __name__ == '__main__':
    # 输入文件路径
    global_dams_shp = r"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\GeoDAR_v11_dams.shp"
    usa_states_shp = r"C:\Users\Kevin\Documents\ResearchData\RangeOfUSA\States.shp"

    # 输出文件路径
    output_dir = r"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\GeoDAR_v11_dams_of_USA"
    output_shp = os.path.join(output_dir, "GeoDAR_v11_dams_USA.shp")

    # 方法1: 简单裁剪（只保留点，不保留州信息）
    # clip_points_by_boundary(global_dams_shp, usa_states_shp, output_shp)

    # 方法2: 裁剪并保留州信息（推荐）
    clip_points_by_boundary_with_state_info(global_dams_shp, usa_states_shp, output_shp)

    print("\n处理完成！")