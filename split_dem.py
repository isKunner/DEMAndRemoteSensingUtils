#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: clip_tif
# @Time    : 2025/7/24 17:28
# @Author  : Kevin
# @Describe: 对tif进行裁剪，裁剪为32*32大小的数据
import glob
import os

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

def resample_raster(src_data, src_transform, src_crs, dst_resolution_x, dst_resolution_y):
    """
    对栅格数据进行重采样
    :param src_data: 原始数据数组
    :param src_transform: 原始数据的仿射变换矩阵
    :param src_crs: 原始数据的坐标参考系统
    :param dst_resolution_x: 目标x方向分辨率
    :param dst_resolution_y: 目标y方向分辨率
    :return: 重采样后的数据、新仿射变换矩阵
    """
    # 校验分辨率参数
    if dst_resolution_x <= 0 or dst_resolution_y <= 0:
        raise ValueError("目标分辨率必须为正数")

    # 初始化输出数组和变换矩阵
    dst_transform = Affine(dst_resolution_x, 0.0, src_transform.c,
                           0.0, -dst_resolution_y, src_transform.f)

    # 计算目标形状，避免负值
    height = src_data.shape[0]
    width = src_data.shape[1]
    dst_height = int(np.ceil(abs(height * src_transform.e) / dst_resolution_y))
    dst_width = int(np.ceil(abs(width * src_transform.a) / dst_resolution_x))


    # 校验目标形状
    if dst_height <= 0 or dst_width <= 0:
        raise ValueError("计算出的目标形状无效，请检查输入参数和仿射变换矩阵")

    dst_shape = (dst_height, dst_width)
    dst_data = np.empty(dst_shape, dtype=src_data.dtype)

    # 执行重采样
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        resampling=Resampling.bilinear  # 可选：最近邻、双线性等
    )

    return dst_data, dst_transform



def split_tif(input_path, output_dir, tile_size, overlap, could_have_nodata=False, prefix="tile"):
    """
    按指定瓦片大小和重叠像素裁剪TIF文件
    :param input_path: 输入TIF文件路径
    :param output_dir: 瓦片输出目录
    :param tile_size: 瓦片尺寸（长宽相同，像素数）
    :param overlap: 重叠像素数
    :param could_have_nodata: 是否允许方块有无效值
    :param prefix: 文件的命名的前缀
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取TIF数据
    with rasterio.open(input_path) as src:
        data = src.read(1)  # 读取第一波段
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        print(f"按照y轴的分辨率大小进行像元的统一")
        # 重采样：以y方向分辨率为标准，统一像元大小
        target_resolution = abs(transform.e)  # 获取y方向分辨率（通常为负值，取绝对值）
        data, transform = resample_raster(data, transform, crs, target_resolution, target_resolution)

        # 更新高度和宽度
        height, width = data.shape

        step = tile_size - overlap

        if step <= 0:
            raise ValueError("步长必须为正数（瓦片尺寸需大于重叠像素）")

        # 计算瓦片数量
        h_count = (height - tile_size + step - 1) // step + 1  # 向上取整
        w_count = (width - tile_size + step - 1) // step + 1

        print(f"数据的大小是{height}*{width}，共有{h_count}行*{w_count}列的瓦片，每块大小为{tile_size}x{tile_size}，重叠像素为{overlap}")


        # 循环裁剪瓦片
        for h_idx in range(h_count):
            for w_idx in range(w_count):
                # 计算当前瓦片起始/结束索引
                start_h = h_idx * step
                end_h = start_h + tile_size
                start_w = w_idx * step
                end_w = start_w + tile_size

                # 处理边缘瓦片（避免超出范围）
                if end_h > height:
                    end_h = height
                    start_h = end_h - tile_size
                if end_w > width:
                    end_w = width
                    start_w = end_w - tile_size

                # 提取瓦片数据
                tile_data = data[start_h:end_h, start_w:end_w]


                if could_have_nodata:
                    if np.all((tile_data == nodata) | np.isnan(tile_data)):
                        continue
                else:
                    if np.any(tile_data == nodata) or np.any(np.isnan(tile_data)):
                        continue

                # 计算瓦片地理变换
                tile_transform = Affine(
                    transform.a, transform.b, transform.xoff + start_w * transform.a,
                    transform.d, transform.e, transform.yoff + start_h * transform.e
                )

                # 保存瓦片
                # 这里改变了类型，切记切记
                tile_name = f"{prefix}_{h_idx}_{w_idx}.tif"
                tile_path = os.path.join(output_dir, tile_name)
                with rasterio.open(
                        tile_path, 'w',
                        driver='GTiff',
                        height=tile_data.shape[0],
                        width=tile_data.shape[1],
                        count=1,
                        dtype=rasterio.float32,
                        crs=crs,
                        transform=tile_transform,
                        nodata=nodata
                ) as dst:
                    dst.write(tile_data, 1)


if __name__ == '__main__':
    # split_tif(input_path=r'C:\Users\Kevin\Documents\ResearchData\WangMao\wmg_cleaned_dem_1m.tif', output_dir=r'C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Test\WMG_HR_1m_420', tile_size=14, overlap=1)

    original_dir = r"E:\Data\ResearchData\USA_ByState\USGSDEM\Mountain_Original"

    for file in glob.glob('*tif', root_dir=original_dir):
        split_tif(input_path=os.path.join(original_dir, file),
                  output_dir=r'E:\Data\ResearchData\USA_ByState\USGSDEM\Mountain',
                  tile_size=448,
                  overlap=0,
                  could_have_nodata=False,
                  prefix=file.split('.')[0])