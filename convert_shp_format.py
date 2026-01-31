#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: convert_format_from_shp_to_json
# @Time    : 2026/1/24 23:19
# @Author  : Kevin
# @Describe: 将

import geopandas as gpd
import os
from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np
from shapely.ops import cascaded_union


def points_shp_to_merged_multipolygon(input_shp, output_shp, buffer_size_meters=100):
    """
    将点SHP文件转换为多边形，然后合并为单个MultiPolygon

    Args:
        input_shp (str): 输入点SHP文件路径
        output_shp (str): 输出SHP文件路径
        buffer_size_meters (float): 每个点的缓冲区大小（米），创建正方形区域
    """
    # 读取SHP文件
    print(f"读取SHP文件: {input_shp}")
    gdf = gpd.read_file(input_shp)

    # 检查数据类型
    print(f"原始数据包含 {len(gdf)} 个要素")
    geom_types = gdf.geometry.geom_type.value_counts()
    print("几何类型统计:")
    for geom_type, count in geom_types.items():
        print(f"  {geom_type}: {count} 个")

    # 检查是否为点数据
    if 'Point' in geom_types.index:
        print(f"\n检测到点数据，将为每个点创建 {buffer_size_meters}x{buffer_size_meters} 米的正方形区域...")

        # 转换为EPSG:4326 (WGS84) 如果不是
        if gdf.crs is None:
            print("警告: 输入数据无坐标系信息，假设为EPSG:4326")
            gdf.set_crs("EPSG:4326", inplace=True)

        # 临时转换为投影坐标系（单位：米）
        projected_crs = "EPSG:5070"  # NAD83 Conus Albers (美国本土)
        print(f"临时转换为投影坐标系 {projected_crs} 以创建精确缓冲区...")
        projected_gdf = gdf.to_crs(projected_crs)

        # 为每个点创建正方形缓冲区
        buffered_geoms = []
        for point in projected_gdf.geometry:
            # 创建正方形缓冲区
            half_size = buffer_size_meters / 2
            square = Polygon([
                (point.x - half_size, point.y - half_size),
                (point.x - half_size, point.y + half_size),
                (point.x + half_size, point.y + half_size),
                (point.x + half_size, point.y - half_size),
                (point.x - half_size, point.y - half_size)
            ])
            buffered_geoms.append(square)

        # 创建多边形GeoDataFrame
        polygons_gdf = gpd.GeoDataFrame(
            projected_gdf.drop(columns=['geometry']),
            geometry=buffered_geoms,
            crs=projected_crs
        )

        # 转回原始坐标系
        print(f"转换回原始坐标系: {gdf.crs}")
        polygons_gdf = polygons_gdf.to_crs(gdf.crs)

        print(f"成功创建 {len(polygons_gdf)} 个多边形区域")
    else:
        # 如果已经是多边形数据
        print("输入数据已经是多边形，直接处理...")
        polygons_gdf = gdf.copy()

    # 现在将所有多边形合并为MultiPolygon
    print("\n合并所有多边形为单个MultiPolygon...")

    # 过滤出Polygon和MultiPolygon
    polygon_gdf = polygons_gdf[polygons_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()

    # 将所有MultiPolygon拆分为单个Polygon
    all_polygons = []
    for geom in polygon_gdf.geometry:
        if geom.geom_type == 'Polygon':
            all_polygons.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            all_polygons.extend(list(geom.geoms))

    print(f"总共 {len(all_polygons)} 个单Polygon待合并")

    # 合并所有Polygon
    merged_geometry = cascaded_union(all_polygons)

    # 确保结果是MultiPolygon
    if merged_geometry.geom_type == 'Polygon':
        merged_geometry = MultiPolygon([merged_geometry])
    elif merged_geometry.geom_type != 'MultiPolygon':
        raise ValueError(f"合并结果不是MultiPolygon类型，而是: {merged_geometry.geom_type}")

    print(f"合并完成! 最终MultiPolygon包含 {len(merged_geometry.geoms)} 个子多边形")
    print(f"总面积: {merged_geometry.area:.2f} 平方单位")

    # 创建新的GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        {
            'id': [1],
            'original_count': [len(gdf)],
            'buffer_size': [buffer_size_meters],
            'merged_count': [len(merged_geometry.geoms)]
        },
        geometry=[merged_geometry],
        crs=gdf.crs
    )

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_shp))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存结果
    print(f"\n保存合并结果到: {output_shp}")
    merged_gdf.to_file(output_shp)

    # 验证结果
    if os.path.exists(output_shp):
        result_gdf = gpd.read_file(output_shp)
        print("验证结果:")
        print(f"  输出文件包含 {len(result_gdf)} 个要素")
        print(f"  几何类型: {result_gdf.geometry.iloc[0].geom_type}")
        print(f"  子多边形数量: {len(result_gdf.geometry.iloc[0].geoms)}")
    else:
        print("保存失败: 输出文件未创建")


def shp_points_to_polygons(input_shp, output_geojson, buffer_size_meters=10):
    """
    将点SHP文件转换为带缓冲区的多边形GeoJSON（EPSG:4326）

    Args:
        input_shp (str): 输入点SHP文件路径
        output_geojson (str): 输出GeoJSON文件路径
        buffer_size_meters (float): 缓冲区大小（米），创建正方形区域
    """
    # 读取点SHP文件
    print(f"读取点SHP文件: {input_shp}")
    points_gdf = gpd.read_file(input_shp)

    # 检查是否为点数据
    if not all(geom.geom_type == 'Point' for geom in points_gdf.geometry):
        raise ValueError("输入文件不是点数据！需要点SHP文件")

    print(f"找到 {len(points_gdf)} 个点要素")

    # 转换为EPSG:4326 (WGS84)
    if points_gdf.crs != "EPSG:4326":
        print("转换坐标系为EPSG:4326 (WGS84)...")
        # 先转换为投影坐标系进行缓冲，再转回WGS84
        # 使用美国常用的投影坐标系（适用于大部分美国本土）
        if points_gdf.crs is None:
            print("警告: 输入数据无坐标系信息，假设为EPSG:4326")
            points_gdf.set_crs("EPSG:4326", inplace=True)

        # 临时转换为投影坐标系（单位：米）
        projected_crs = "EPSG:5070"  # NAD83 Conus Albers (美国本土)
        print(f"临时转换为投影坐标系 {projected_crs} 以创建缓冲区...")
        projected_gdf = points_gdf.to_crs(projected_crs)

        # 为每个点创建正方形缓冲区（比圆形缓冲区更符合卫星图像需求）
        print(f"为每个点创建 {buffer_size_meters}x{buffer_size_meters} 米的正方形区域...")
        buffered_geoms = []
        for point in projected_gdf.geometry:
            # 创建正方形缓冲区
            half_size = buffer_size_meters / 2
            square = Polygon([
                (point.x - half_size, point.y - half_size),
                (point.x - half_size, point.y + half_size),
                (point.x + half_size, point.y + half_size),
                (point.x + half_size, point.y - half_size),
                (point.x - half_size, point.y - half_size)
            ])
            buffered_geoms.append(square)

        # 创建新的GeoDataFrame
        polygons_gdf = gpd.GeoDataFrame(
            projected_gdf.drop(columns=['geometry']),
            geometry=buffered_geoms,
            crs=projected_crs
        )

        # 转回WGS84
        print("转换回EPSG:4326坐标系...")
        polygons_gdf = polygons_gdf.to_crs("EPSG:4326")
    else:
        # 如果已经是EPSG:4326，需要估算米到度的转换（只适用于小范围）
        print("警告: 数据已在EPSG:4326，估算缓冲区大小（仅适用于小范围）")
        avg_lat = points_gdf.geometry.y.mean()
        meters_per_degree = 111319.5 * np.cos(np.radians(avg_lat))
        buffer_size_degrees = buffer_size_meters / meters_per_degree

        buffered_geoms = []
        for point in points_gdf.geometry:
            half_size = buffer_size_degrees / 2
            square = Polygon([
                (point.x - half_size, point.y - half_size),
                (point.x - half_size, point.y + half_size),
                (point.x + half_size, point.y + half_size),
                (point.x + half_size, point.y - half_size),
                (point.x - half_size, point.y - half_size)
            ])
            buffered_geoms.append(square)

        polygons_gdf = gpd.GeoDataFrame(
            points_gdf.drop(columns=['geometry']),
            geometry=buffered_geoms,
            crs="EPSG:4326"
        )

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_geojson))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存为GeoJSON
    print(f"保存为多边形GeoJSON: {output_geojson}")
    polygons_gdf.to_file(output_geojson, driver='GeoJSON')

    # 验证转换
    if os.path.exists(output_geojson):
        file_size = os.path.getsize(output_geojson) / 1024  # KB
        print(f"转换成功! GeoJSON文件大小: {file_size:.2f} KB")
        print(f"包含 {len(polygons_gdf)} 个多边形要素")
        print("几何类型统计:")
        geom_types = polygons_gdf.geometry.geom_type.value_counts()
        for geom_type, count in geom_types.items():
            print(f"  {geom_type}: {count} 个")
    else:
        print("转换失败: 输出文件未创建")


if __name__ == "__main__":
    # 输入点SHP文件路径

    i = 13
    j = 2

    input_shp = rf"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\USA_DAM_ByRegion\group_{i}_{j}\GeoDAR_v11_dams_of_USA_group{i}_{j}.shp"

    # 输出GeoJSON文件路径
    output_geojson = rf"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\USA_DAM_ByRegion\group_{i}_{j}\GeoDAR_v11_dams_of_USA_group{i}_{j}_merged.geojson"

    # 执行转换（创建100x100米的正方形区域）
    shp_points_to_polygons(input_shp, output_geojson, buffer_size_meters=10)


    # points_shp_to_merged_multipolygon(input_shp, output_geojson)