#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: crop_shp_from_shp
# @Time    : 2026/1/24 22:44
# @Author  : Kevin
# @Describe:
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给SHP加缓冲区（14像素距离）→ 裁剪DEM → 切分
确保黄土高原范围内的瓦片都是完整的14×14，无填充
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.transform import Affine
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing


def clip_with_buffer(input_tif, shp_path, output_tif, pixel_buffer=14):
    """
    用SHP+缓冲区裁剪DEM

    Args:
        pixel_buffer: 缓冲的像素数（默认14）
    """
    print(f"读取SHP: {shp_path}")
    gdf = gpd.read_file(shp_path)

    # 确保坐标系一致
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # 读取DEM获取分辨率
    with rasterio.open(input_tif) as src:
        pixel_size = abs(src.transform.a)  # 通常是0.00027778度（约30米）
        print(f"DEM分辨率: {pixel_size}度/像素")

        # 计算14像素对应的地理距离（度）
        buffer_distance = pixel_buffer * pixel_size
        print(f"缓冲区: {pixel_buffer}像素 = {buffer_distance:.6f}度")

        # 给SHP加缓冲区（向外扩展14像素）
        gdf_buffered = gdf.copy()
        gdf_buffered['geometry'] = gdf.geometry.buffer(buffer_distance)

        print(f"原始范围: {gdf.total_bounds}")
        print(f"缓冲后范围: {gdf_buffered.total_bounds}")

    # 用缓冲后的SHP裁剪DEM
    shapes = [geom for geom in gdf_buffered.geometry]

    with rasterio.open(input_tif) as src:
        out_image, out_transform = mask(
            src,
            shapes,
            crop=True,
            all_touched=True,
            filled=False
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512
        })

        print(f"\n保存带缓冲区的裁剪结果: {output_tif}")
        print(f"尺寸: {out_image.shape[1]} x {out_image.shape[2]}")

        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)
            # 复制标签
            tags = src.tags()
            if tags:
                dest.update_tags(**tags)

    print("✅ 缓冲区裁剪完成！")
    return output_tif


def write_tile(args):
    """写瓦片"""
    tile_data, tile_transform, crs, nodata, output_dir, h_idx, w_idx = args
    tile_name = f"{h_idx}_{w_idx}.tif"
    tile_path = os.path.join(output_dir, tile_name)

    if os.path.exists(tile_path):
        return f"{h_idx}_{w_idx} 已存在", True

    try:
        with rasterio.open(
                tile_path, 'w',
                driver='GTiff',
                height=tile_data.shape[0],
                width=tile_data.shape[1],
                count=1,
                dtype=tile_data.dtype,
                crs=crs,
                transform=tile_transform,
                nodata=nodata,
                compress='lzw'
        ) as dst:
            dst.write(tile_data, 1)
        return f"{h_idx}_{w_idx} 成功", True
    except Exception as e:
        return f"{h_idx}_{w_idx} 失败: {str(e)}", False

if __name__ == '__main__':
    # 路径配置
    INPUT_TIF = r"D:\Data\ResearchData\Copernicus\20260404_CopernicusDEM.tif"
    SHP_PATH = r"C:\Users\Kevin\Documents\ResearchData\RangeOfLoessPlateau\Range_of_Loess_Plateau.shp"
    BUFFERED_TIF = r"D:\Data\ResearchData\Copernicus\LoessPlateau_Buffered.tif"
    OUTPUT_DIR = r"D:\Data\ResearchData\Copernicus\CopernicusDEMPatch"

    # 步骤1: 加缓冲区（14像素）并裁剪
    clip_with_buffer(INPUT_TIF, SHP_PATH, BUFFERED_TIF, pixel_buffer=30)