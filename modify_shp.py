#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: modify_shp
# @Time    : 2026/1/2 10:31
# @Author  : Kevin
# @Describe:
import os

import geopandas as gpd

def filter_shp(input_shp_path: str, output_shp_path: str, filter_type: str,
                      filter_mode: str = "include"):
    """
    读取SHP文件并根据type字段筛选或去除某些行

    参数:
    - input_shp_path: 输入SHP文件路径
    - output_shp_path: 输出SHP文件路径
    - filter_type: 要筛选的type值
    - filter_mode: "include"表示保留指定type的行，"exclude"表示去除指定type的行
    """
    # 读取SHP文件
    gdf = gpd.read_file(input_shp_path)

    if filter_mode == "include":
        # 保留指定type的行
        filtered_gdf = gdf[gdf['type'] == filter_type]
    elif filter_mode == "exclude":
        # 去除指定type的行
        filtered_gdf = gdf[gdf['type'] != filter_type]
    else:
        raise ValueError("filter_mode必须是'include'或'exclude'")

    # 保存筛选后的结果
    output_dir = os.path.dirname(output_shp_path)
    os.makedirs(output_dir, exist_ok=True)
    filtered_gdf.to_file(output_shp_path, encoding='utf-8')

    print(f"原始数据行数: {len(gdf)}")
    print(f"筛选后数据行数: {len(filtered_gdf)}")
    print(f"结果已保存至: {output_shp_path}")

if __name__ == '__main__':

    # 示例用法
    # 保留type为'upstream'的行
    filter_shp(
        input_shp_path=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\check_dam_final.shp",
        output_shp_path=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\silted_land.shp",
        filter_type="control_area",
        filter_mode="include"
    )
