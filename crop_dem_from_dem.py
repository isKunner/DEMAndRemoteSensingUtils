#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: crop_dem_from_dem
# @Time    : 2025/7/24 17:52
# @Author  : Kevin
# @Describe: 根据已有的裁剪好的tif进行重新的裁剪采样(方形)

import os
import math
import numpy as np
import pandas as pd

import rasterio
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds, Window


def process_reference_with_sources(
        source_raster_paths,
        reference_raster_path,
        output_path,
        resampling_method=Resampling.nearest
):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    try:
        # 1. 打开所有Src（DEM分块）
        src_datasets = [rasterio.open(p) for p in source_raster_paths if os.path.exists(p)]
        if not src_datasets:
            return False

        # 获取Src的通道数（DEM通常是1）
        src_count = src_datasets[0].count
        src_crs_base = src_datasets[0].crs

        # 2. 打开Ref（只提供范围）
        with rasterio.open(reference_raster_path) as ref_ds:
            ref_crs = ref_ds.crs
            ref_bounds = ref_ds.bounds
            ref_height, ref_width = ref_ds.shape
            ref_transform = ref_ds.transform
            ref_nodata = -9999  # 强制用-9999

            ref_left, ref_bottom, ref_right, ref_top = ref_bounds

            # 初始化：按Src的通道数！
            dst_array = np.full((src_count, ref_height, ref_width), ref_nodata, dtype=np.float64)
            count_array = np.zeros((ref_height, ref_width), dtype=np.uint8)

            # 3. 遍历每个Src
            for src_ds in src_datasets:
                src_crs = src_ds.crs
                src_bounds = src_ds.bounds
                src_nodata = src_ds.nodata if src_ds.nodata is not None else np.nan

                # 检查重叠（转到Ref坐标系）
                if src_crs != ref_crs:
                    src_bounds_ref = transform_bounds(src_crs, ref_crs, *src_bounds)
                else:
                    src_bounds_ref = src_bounds

                src_left, src_bottom, src_right, src_top = src_bounds_ref

                if not (src_left < ref_right and src_right > ref_left and
                       src_top > ref_bottom and src_bottom < ref_top):
                    continue

                # 计算读取窗口
                pixel_x = abs(src_ds.transform.a)
                pixel_y = abs(src_ds.transform.e)

                if ref_crs != src_crs:
                    ref_bounds_src = transform_bounds(ref_crs, src_crs, *ref_bounds)
                else:
                    ref_bounds_src = ref_bounds

                rb_left, rb_bottom, rb_right, rb_top = ref_bounds_src

                window = from_bounds(
                    rb_left - pixel_x, rb_bottom - pixel_y,
                    rb_right + pixel_x, rb_top + pixel_y,
                    transform=src_ds.transform
                ).intersection(Window(0, 0, src_ds.width, src_ds.height))

                if window.width <= 0 or window.height <= 0:
                    continue

                src_data = src_ds.read(window=window)
                src_transform = rasterio.windows.transform(window, src_ds.transform)

                # 临时数组：按Src通道数！
                temp_array = np.full((src_count, ref_height, ref_width), ref_nodata, dtype=np.float64)

                # 重投影（逐band，用src_count）
                for band_idx in range(src_count):
                    reproject(
                        source=src_data[band_idx],
                        destination=temp_array[band_idx],
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=resampling_method,
                        src_nodata=src_nodata,
                        dst_nodata=ref_nodata
                    )

                # 合并逻辑（按src_count）
                if np.isnan(ref_nodata):
                    src_valid = ~np.isnan(temp_array[0])
                    dst_valid = ~np.isnan(dst_array[0])
                else:
                    src_valid = (temp_array[0] != ref_nodata)
                    dst_valid = (dst_array[0] != ref_nodata)

                # 1. Src有效，Dst无效 → 赋值
                mask_new = src_valid & (~dst_valid)
                for band in range(src_count):  # 逐band赋值
                    dst_array[band][mask_new] = temp_array[band][mask_new]
                count_array[mask_new] = 1

                # 2. 都有效 → 平均
                mask_both = src_valid & dst_valid
                if np.any(mask_both):
                    curr_counts = count_array[mask_both]
                    for band in range(src_count):
                        dst_array[band][mask_both] = (
                            dst_array[band][mask_both] * curr_counts +
                            temp_array[band][mask_both]
                        ) / (curr_counts + 1)
                    count_array[mask_both] += 1

            # 4. 保存（强制float32，用src_count）
            if np.sum(count_array > 0) == 0:
                print(f"警告: {os.path.basename(output_path)} 无有效数据")
                return False

            output_dtype = np.float32
            write_nodata = -9999

            result = dst_array.astype(output_dtype)
            invalid_mask = count_array == 0
            for band in range(src_count):
                result[band][invalid_mask] = write_nodata

            # 中间检查
            mid_row, mid_col = ref_height // 2, ref_width // 2
            mid_region = result[:, max(0,mid_row-10):min(ref_height,mid_row+10),
                              max(0,mid_col-10):min(ref_width,mid_col+10)]
            mid_valid = np.sum(mid_region != write_nodata)

            if mid_valid / mid_region.size < 0.99:
                print(f"警告: {os.path.basename(output_path)} 中间有效率低")

            # 写入（count=src_count！）
            with rasterio.open(output_path, 'w', driver='GTiff',
                               height=ref_height, width=ref_width,
                               count=src_count, dtype=output_dtype,  # ← 这里是src_count
                               crs=ref_crs, transform=ref_transform,
                               nodata=write_nodata,
                               compress='LZW') as dst:
                dst.write(result)
                dst.write_mask((count_array > 0).astype(np.uint8) * 255)

        for ds in src_datasets:
            ds.close()
        return True

    except Exception as e:
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def crop_source_to_reference(
        source_raster_path,
        reference_inputs,
        output_destination,
        output_suffix="",
        resampling_method=Resampling.nearest,
        log_csv=None
):
    """
    根据参考影像的范围和属性，从源影像中裁剪和重采样数据。

    支持灵活的输入输出组合：
    1. 单个参考文件 + 单个输出路径 (str) -> 写入指定路径
    2. 单个参考文件 + 输出目录 (str) -> 自动生成文件名放入目录
    3. 多个参考文件列表 + 输出目录 (str) -> 自动生成多个文件名放入目录
    4. 多个参考文件列表 + 输出路径列表 (list) -> 一对一写入指定路径
    5. 输入目录 + 输出目录 (str) -> 自动处理输入目录下所有.tif文件，生成对应文件名放入输出目录

    Args:
        source_raster_path (str): 源（大）影像文件的路径。
        reference_inputs (str or list of str):
            - 单个参考影像文件路径 (str)。
            - 包含多个参考影像文件路径的列表 (list of str)。
            - 或者，包含参考影像的目录路径 (str)。
        output_destination (str or list of str):
            - 如果 reference_inputs 是单个文件，它可以是：
                - 一个存在的目录路径 (str)，则自动生成文件名。
                - 一个具体的输出文件路径 (str)。
            - 如果 reference_inputs 是列表，它可以是：
                - 一个存在的目录路径 (str)，则为每个参考文件自动生成对应的输出文件名。
                - 一个包含对应输出文件路径的列表 (list of str)，长度必须与 reference_inputs 一致。
            - 如果 reference_inputs 是目录，则 output_destination 必须是目录路径 (str)。
        output_suffix (str, optional): 添加到自动生成的输出文件名的后缀。默认为空字符串。
        resampling_method (rasterio.warp.Resampling, optional): 重采样方法。默认为最近邻。
    """
    # 解析输入和输出，生成最终的处理对 (ref_path, out_path)
    processing_pairs = _resolve_input_output_pairs(reference_inputs, output_destination, output_suffix)

    log_content = []

    # --- 打开源影像 ---
    with rasterio.open(source_raster_path) as src_ds:
        print(f"已打开源影像: {os.path.basename(source_raster_path)}")

        # 获取源影像的基本元数据
        src_crs = src_ds.crs
        src_bounds = src_ds.bounds
        src_nodata = src_ds.nodata if src_ds.nodata is not None else np.nan
        src_count = src_ds.count
        src_transform = src_ds.transform

        # 循环处理每个生成的对
        for ref_path, out_path in processing_pairs:

            if not _process_single_reference_with_source_handle(
                src_ds, ref_path, out_path,
                src_crs, src_bounds, src_nodata, src_count, src_transform,
                resampling_method
            ):
                print(f"处理出错: {os.path.basename(ref_path)} -> {os.path.basename(out_path)}")
                log_content.append({
                    'source_path': source_raster_path,
                    "reference_path": ref_path,
                    "output_path": out_path,
                })

    if log_csv is not None:
        new_records_df = pd.DataFrame(log_content)

        # 检查CSV文件是否存在，如果存在则读取现有数据
        if os.path.isfile(log_csv):
            existing_df = pd.read_csv(log_csv)
            # 合并现有数据和新数据
            combined_df = pd.concat([existing_df, new_records_df], ignore_index=True)
        else:
            # 如果文件不存在，只使用新数据
            combined_df = new_records_df

        combined_df.to_csv(log_csv, index=False)

    print("所有影像处理完成。")

    return processing_pairs


def _resolve_input_output_pairs(reference_inputs, output_destination, output_suffix):
    """根据输入和输出参数，生成 (reference_path, output_path) 的列表。"""
    pairs = []

    # 检查输入类型
    if isinstance(reference_inputs, str):
        if os.path.isfile(reference_inputs):
            # 输入是单个文件
            ref_file = reference_inputs
            if os.path.isdir(output_destination):
                # 输出是目录 -> 自动生成文件名
                os.makedirs(output_destination, exist_ok=True)
                filename = os.path.splitext(os.path.basename(ref_file))[0] + output_suffix + ".tif"
                output_path = os.path.join(output_destination, filename)
                pairs.append((ref_file, output_path))
            elif isinstance(output_destination, str):
                # 输出是单个文件路径
                pairs.append((ref_file, output_destination))
            else:
                raise ValueError(
                    "当 reference_inputs 是单个文件时，output_destination 必须是目录路径或单个文件路径 (str)。")

        elif os.path.isdir(reference_inputs):
            # 输入是目录
            input_dir = reference_inputs
            os.makedirs(output_destination, exist_ok=True)

            # 遍历输入目录，为每个 .tif 文件生成输出路径
            tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]
            for tif_name in tif_files:
                ref_path = os.path.join(input_dir, tif_name)

                output_filename = os.path.splitext(tif_name)[0] + output_suffix + ".tif"
                output_path = os.path.join(output_destination, output_filename)

                pairs.append((ref_path, output_path))
        else:
            raise ValueError(f"reference_inputs '{reference_inputs}' 既不是文件也不是目录。")

    elif isinstance(reference_inputs, list):
        # 输入是文件列表
        ref_list = reference_inputs

        if isinstance(output_destination, list):
            # 输出是指定的文件路径列表
            if len(reference_inputs) != len(output_destination):
                raise ValueError("当 reference_inputs 和 output_destination 都是列表时，它们的长度必须相同。")
            for ref_file, out_file in zip(reference_inputs, output_destination):
                if not os.path.isfile(ref_file):
                    raise ValueError(f"列表中的路径不是文件: {ref_file}")
                pairs.append((ref_file, out_file))
        elif os.path.isdir(output_destination):
            # 输出是目录 -> 为列表中的每个文件自动生成文件名
            for ref_file in ref_list:
                if not os.path.isfile(ref_file):
                    raise ValueError(f"列表中的路径不是文件: {ref_file}")
                filename = os.path.splitext(os.path.basename(ref_file))[0] + output_suffix + ".tif"
                output_path = os.path.join(output_destination, filename)
                pairs.append((ref_file, output_path))
        else:
            raise ValueError(
                "当 reference_inputs 是列表时，output_destination 必须是目录路径 (str) 或输出路径列表 (list)。")
    else:
        raise ValueError("reference_inputs 必须是文件路径 (str)、目录路径 (str) 或文件路径列表 (list)。")

    return pairs


def _process_single_reference_with_source_handle(
        src_ds, reference_raster_path, output_path,
        src_crs, src_bounds, src_nodata, src_count, src_transform,
        resampling_method
):
    """处理单个参考影像文件的核心逻辑，接收已打开的源影像句柄。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with rasterio.open(reference_raster_path) as ref_ds:
            ref_crs = ref_ds.crs
            ref_bounds = ref_ds.bounds
            ref_height, ref_width = ref_ds.shape
            ref_transform = ref_ds.transform
            ref_nodata = ref_ds.nodata if ref_ds.nodata is not None else np.nan

            if ref_crs != src_crs:
                ref_bounds_in_src_crs = transform_bounds(
                    ref_crs, src_crs, ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top
                )
                ref_left, ref_bottom, ref_right, ref_top = ref_bounds_in_src_crs
            else:
                ref_left, ref_bottom, ref_right, ref_top = ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top

            overlap = (
                    ref_left < src_bounds.right and ref_right > src_bounds.left and
                    ref_top > src_bounds.bottom and ref_bottom < src_bounds.top
            )
            if not overlap:
                print(f"警告: 参考影像 {os.path.basename(reference_raster_path)} 与源影像无重叠，跳过。")
                return

            pixel_size_x = abs(src_transform.a)
            pixel_size_y = abs(src_transform.e)

            ref_left_exp = ref_left - pixel_size_x
            ref_right_exp = ref_right + pixel_size_x
            ref_bottom_exp = ref_bottom - pixel_size_y
            ref_top_exp = ref_top + pixel_size_y

            window = from_bounds(ref_left_exp, ref_bottom_exp, ref_right_exp, ref_top_exp, transform=src_transform)
            window = window.intersection(Window(0, 0, src_ds.width, src_ds.height))

            win_col_min = max(0, math.floor(window.col_off))
            win_row_min = max(0, math.floor(window.row_off))
            win_col_max = min(src_ds.width, math.ceil(window.col_off + window.width))
            win_row_max = min(src_ds.height, math.ceil(window.row_off + window.height))

            final_window = Window(
                col_off=win_col_min, row_off=win_row_min,
                width=win_col_max - win_col_min, height=win_row_max - win_row_min
            )

            src_window_data = src_ds.read(window=final_window)
            src_window_transform = rasterio.windows.transform(final_window, src_transform)
            src_window_mask = src_ds.dataset_mask(window=final_window)
            mask_valid_ratio = np.sum(src_window_mask == 255) / src_window_mask.size if src_window_mask.size > 0 else 0

            dst_array = np.full((src_count, ref_height, ref_width), src_nodata, dtype=src_window_data.dtype)

            for band_idx in range(src_count):
                reproject(
                    source=src_window_data[band_idx],
                    destination=dst_array[band_idx],
                    src_transform=src_window_transform,
                    src_crs=src_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling_method,
                    src_nodata=src_nodata,
                    dst_nodata=src_nodata,
                    src_mask=src_window_mask if mask_valid_ratio > 0 else np.nan
                )

            mid_row = ref_height // 2
            mid_col = ref_width // 2
            mid_region_slice = slice(max(0, mid_row - 10), min(ref_height, mid_row + 10)), \
                slice(max(0, mid_col - 10), min(ref_width, mid_col + 10))
            mid_region = dst_array[:, mid_region_slice[0], mid_region_slice[1]]
            mid_total = mid_region.size
            if ref_nodata is np.nan:
                mid_valid = mid_total - np.sum(np.isnan(mid_region))
            else:
                mid_valid = np.sum(mid_region != ref_nodata)

            if mid_valid / mid_total < 0.99:
                print(f"警告: 输出影像 {os.path.basename(output_path)} 中间区域有效率低于99%，请检查！")

            with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=ref_height,
                    width=ref_width,
                    count=src_count,
                    dtype=dst_array.dtype,
                    crs=ref_crs,
                    transform=ref_transform,
                    nodata=src_nodata,
                    compress='LZW'
            ) as dst:
                dst.write(dst_array)
                dst_mask = np.all(dst_array != src_nodata, axis=0)
                dst.write_mask(dst_mask.astype(rasterio.uint8) * 255)

    except Exception as e:
        print(f"处理参考影像 {os.path.basename(reference_raster_path)} 时出错: {e}")
        return False

    return True


def merge_sources_to_reference(reference_path, source_paths, output_path,
                               nodata_value=-9999, dtype='float32',
                               resampling_method=Resampling.nearest):
    """
    将多个源影像根据参考影像的范围进行拼接，支持坐标系转换。
    逻辑：优先保留先读取的数据，重叠区域取平均值。

    【最终修正版】
    1. 修复了坐标系不一致时，window计算错误导致的问题。
    2. 将CRS相同和不同的情况处理逻辑彻底分开，避免混合坐标系信息。
    3. 重构了Nodata值的处理逻辑，使其对有无定义Nodata的源文件都稳健。

    Args:
        reference_path (str): 参考影像路径（定义输出的范围和分辨率）。
        source_paths (list of str): 源影像文件路径列表（按处理优先级顺序排列）。
        output_path (str): 输出文件路径。
        nodata_value (float, optional): 输出的空值。默认为 -9999。
        dtype (str, optional): 输出数据类型。默认为 float32 以支持平均值计算。
        resampling_method (Resampling, optional): 重采样方法。默认为最近邻。
    """
    print(f"开始处理拼接，参考影像: {os.path.basename(reference_path)}")

    # --- 1. 读取参考影像元数据并初始化画布 ---
    with rasterio.open(reference_path) as ref_ds:
        ref_meta = ref_ds.meta.copy()
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_height = ref_ds.height
        ref_width = ref_ds.width

        # 创建目标画布，初始化为输出的nodata值
        target_raster = np.full((ref_height, ref_width), nodata_value, dtype=dtype)

        print(f"初始化画布: {ref_width}x{ref_height}, CRS: {ref_crs}")

    # --- 2. 遍历处理每个源文件 ---
    for src_idx, src_path in enumerate(source_paths):
        print(f"  处理源文件 {src_idx + 1}/{len(source_paths)}: {os.path.basename(src_path)}")
        try:
            with rasterio.open(src_path) as src_ds:
                src_nodata = src_ds.nodata
                src_crs = src_ds.crs

                # 【核心修正：彻底分离CRS相同和不同的处理逻辑】

                # --- 情况一：坐标系不同 ---
                if src_crs != ref_crs:
                    print(f"    坐标系转换: {src_crs} -> {ref_crs}")

                    # 创建与目标画布同样大小的临时数组用于重投影
                    reprojected_data = np.full_like(target_raster, nodata_value, dtype=dtype)

                    reproject(
                        source=src_ds.read(1),
                        destination=reprojected_data,
                        src_transform=src_ds.transform,
                        src_crs=src_crs,
                        src_nodata=src_nodata,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        dst_nodata=nodata_value,
                        resampling=resampling_method,
                    )

                    # 重投影后，数据已经与目标画布像素对齐，无需再计算窗口
                    # 直接进行全画布融合
                    target_patch = target_raster
                    src_patch = reprojected_data

                # --- 情况二：坐标系相同 ---
                else:
                    print(f"    坐标系一致: {src_crs}")
                    # 计算源影像在参考影像坐标系下的窗口
                    window = from_bounds(*src_ds.bounds, ref_transform)

                    # 【关键修正】使用稳健的窗口有效性检查
                    if window.width <= 0 or window.height <= 0:
                        print(f"    无重叠: {os.path.basename(src_path)}")
                        continue

                    # 确保窗口在参考影像范围内
                    window = window.intersection(rasterio.windows.Window(0, 0, ref_width, ref_height))
                    if window.width <= 0 or window.height <= 0:
                        print(f"    交集为空: {os.path.basename(src_path)}")
                        continue

                    # 计算目标画布上的切片 (行和列)
                    row_start, row_end = int(window.row_off), int(window.row_off + window.height)
                    col_start, col_end = int(window.col_off), int(window.col_off + window.width)

                    # 提取对应的切片进行操作
                    target_patch = target_raster[row_start:row_end, col_start:col_end]
                    src_patch = src_ds.read(1, window=window)

                # --- 统一的后续处理逻辑 ---
                # 【关键修正】稳健地处理源文件的nodata，为CRS相同的情况创建掩码
                if src_crs == ref_crs and src_nodata is not None:
                    src_valid_mask = (src_patch != src_nodata)
                else:
                    # 如果CRS不同，reproject已将nodata统一；或CRS相同但src_nodata为None
                    src_valid_mask = (src_patch != nodata_value)

                # 提取目标画布上已有数据的区域
                target_is_valid_mask = (target_patch != nodata_value)

                # 找出需要填充的区域 (目标为nodata，源有效)
                mask_fill = ~target_is_valid_mask & src_valid_mask

                # 找出需要混合的区域 (目标和源都有效)
                mask_blend = target_is_valid_mask & src_valid_mask

                # 更新画布
                # 1. 填充空区域
                target_patch[mask_fill] = src_patch[mask_fill]

                # 2. 混合重叠区域
                if np.any(mask_blend):
                    target_patch[mask_blend] = (target_patch[mask_blend] + src_patch[mask_blend]) / 2.0

                # 【重要】如果处理的是切片，需要写回大数组
                if src_crs == ref_crs:
                    target_raster[row_start:row_end, col_start:col_end] = target_patch
                # 如果CRS不同，target_patch就是target_raster的视图，修改会直接生效，无需写回

        except Exception as e:
            import traceback
            print(f"    处理源文件 {src_path} 时出错: {e}")
            print("    详细错误追踪:")
            traceback.print_exc()

    # --- 3. 写入输出文件 ---
    meta_update = {
        'dtype': dtype,
        'count': 1,
        'nodata': nodata_value,
        'compress': 'LZW'
    }
    ref_meta.update(meta_update)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        dst.write(target_raster, 1)
        print(f"拼接完成，输出至: {output_path}")


if __name__ == '__main__':

    # reference_path = r"D:\研究文件\ResearchData\USA\GoogleRemoteSensing\GeoDAR_v11_dams_of_USA_group1\0.tif"
    # source_paths = [r"C:\Users\Kevin\Downloads\Edge\USGS_1M_18_x62y457_CT_Statewide_C16.tif", r"C:\Users\Kevin\Downloads\Edge\USGS_one_meter_x62y457_CT_Sandy_2014.tif", r"C:\Users\Kevin\Downloads\Edge\USGS_1M_18_x62y457_NY_FEMAR2_Central_2018_D19.tif"]
    # output_path = r"C:\Users\Kevin\Downloads\Edge\0.tif"
    #
    # merge_sources_to_reference(reference_path, source_paths, output_path, dtype=np.float32)

    source_file = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    input_dir = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Test\WMG_1.0m_1024pixel"
    output_dir = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Test\Copernicus_1.0m_1024pixel"
    os.makedirs(output_dir, exist_ok=True)
    print("开始批量处理目录...")
    crop_source_to_reference(
        source_raster_path=source_file,
        reference_inputs=input_dir,  # 传入目录
        output_destination=output_dir,  # 传入目录
        output_suffix="",
        resampling_method=Resampling.nearest
    )
    print("批量处理完成。")