#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: crop_dem_from_shp
# @Time    : 2026/4/4 13:48
# @Author  : Kevin
# @Describe: 根据SHP范围裁剪DEM，保留完整像素（不resample）

import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np


def clip_dem_by_shp(input_tif, shp_path, output_tif):
    """
    根据SHP范围裁剪DEM
    """
    print(f"读取SHP: {shp_path}")
    gdf = gpd.read_file(shp_path)

    # 确保坐标系一致（转为WGS84）
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        print(f"转换坐标系: {gdf.crs} -> EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)

    # 获取几何形状（用于mask）
    shapes = [geom for geom in gdf.geometry]

    print(f"读取DEM: {input_tif}")
    with rasterio.open(input_tif) as src:
        print(f"原始尺寸: {src.height} x {src.width}")
        print(f"坐标系: {src.crs}")

        # 使用rasterio.mask裁剪（保留所有与shp相交的像素）
        out_image, out_transform = mask(
            src,
            shapes,
            crop=True,  # 裁剪到边界框
            all_touched=True,  # 保留所有接触的像素（包括边缘）
            filled=False  # 不填充nodata，保持原值
        )

        # 更新元数据
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

        # 保存
        print(f"保存裁剪结果: {output_tif}")
        print(f"裁剪后尺寸: {out_image.shape[1]} x {out_image.shape[2]}")

        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)

            # 复制原文件的tags（如果有EGM2008等）
            tags = src.tags()
            if tags:
                dest.update_tags(**tags)

    print("✅ 裁剪完成！")


if __name__ == '__main__':
    # 路径配置
    INPUT_TIF = r"D:\Data\ResearchData\Copernicus\20260404_CopernicusDEM.tif"
    SHP_PATH = r"C:\Users\Kevin\Documents\ResearchData\RangeOfLoessPlateau\Range_of_Loess_Plateau.shp"
    OUTPUT_TIF = r"D:\Data\ResearchData\Copernicus\LoessPlateau_Cropped.tif"

    clip_dem_by_shp(INPUT_TIF, SHP_PATH, OUTPUT_TIF)