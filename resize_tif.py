#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: resize_tif
# @Time    : 2025/7/24 10:08
# @Author  : Kevin
# @Describe: 对DEM（TIF）进行分辨率的缩放，统一到目标DEM的分辨率和坐标系
import glob
import os
import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage


def unify_dem(input_dem_path, target_dem_path, output_path=None, buffer_pixels=1):
    """
    将输入DEM统一到目标DEM的分辨率和坐标系（支持多波段）

    参数:
        input_dem_path: 输入TIF路径
        target_dem_path: 目标TIF路径（提供目标分辨率和坐标系）
        output_path: 输出路径
        buffer_pixels: 边界缓冲像素数（默认1，避免边界裁剪）
    """

    # --------------------------
    # 1. 读取原始DEM和目标DEM的元数据
    # --------------------------
    with rasterio.open(input_dem_path) as input_src, \
            rasterio.open(target_dem_path) as target_src:

        # 原始DEM信息
        input_count = input_src.count  # 波段数
        input_array = input_src.read()  # 读取所有波段 (count, height, width)
        input_nodata = input_src.nodata
        # input_nodata = np.nan if input_nodata is None else input_nodata
        input_crs = input_src.crs
        input_transform = input_src.transform
        input_res = input_src.res

        # 目标DEM信息
        target_crs = target_src.crs
        target_res = target_src.res
        target_cell_x, target_cell_y_raw = target_res

        # 强制y方向像素大小为负值
        target_cell_y = abs(target_cell_y_raw) * -1

        print(f"输入TIF波段数: {input_count}")
        print(f"原始坐标系: {input_crs.to_string() if input_crs else '未知'}")
        print(f"目标坐标系: {target_crs.to_string() if target_crs else '未知'}")
        print(f"原始坐标系Nodata：{input_nodata}")
        print(f"原始像素尺寸: {input_res}")
        print(f"原始像素的像素：{target_src.width}行 x {target_src.height}列")
        print(f"目标像素尺寸: ({target_cell_x:.6f}, {target_cell_y:.6f})")
        print(f"目标像素的像素：{target_src.width}行 x {target_src.height}列")

    # --------------------------
    # 2. 计算原始DEM的有效数据范围（基于第一波段）
    # --------------------------
    first_band = input_array[0] if input_count > 0 else input_array
    valid_mask = (first_band != input_nodata) if input_nodata is not None else ~np.isnan(first_band)
    valid_rows, valid_cols = np.where(valid_mask)
    if len(valid_rows) == 0:
        raise ValueError("原始DEM无有效数据")

    # 计算有效范围
    min_row, min_col = valid_rows.min(), valid_cols.min()
    max_row, max_col = valid_rows.max(), valid_cols.max()

    # 获取边界坐标
    top_left_x, top_left_y = input_src.xy(min_row, min_col)
    bottom_right_x, bottom_right_y = input_src.xy(max_row, max_col)

    # 确保边界顺序正确
    valid_bounds = (
        min(top_left_x, bottom_right_x),  # 左边界
        min(top_left_y, bottom_right_y),  # 下边界
        max(top_left_x, bottom_right_x),  # 右边界
        max(top_left_y, bottom_right_y)  # 上边界
    )

    print(
        f"原始有效范围: 左={valid_bounds[0]:.2f}, 右={valid_bounds[2]:.2f}, 下={valid_bounds[1]:.2f}, 上={valid_bounds[3]:.2f}")

    # --------------------------
    # 3. 坐标系转换（如果需要）
    # --------------------------
    current_array = input_array
    current_bounds = valid_bounds
    current_transform = input_transform
    current_crs = input_crs

    if current_crs != target_crs:
        print("正在进行坐标系转换...")

        # 计算坐标系转换参数
        transform, width, height = calculate_default_transform(
            current_crs,
            target_crs,
            input_src.width,
            input_src.height,
            *input_src.bounds
        )

        # 为每个波段创建转换后的数组
        crs_converted_arrays = []
        for band_idx in range(input_count):
            band_array = input_array[band_idx]
            crs_converted_array = np.full((height, width), input_nodata, dtype=band_array.dtype)
            crs_converted_arrays.append(crs_converted_array)

        # 执行坐标系转换（逐波段）
        for band_idx in range(input_count):
            reproject(
                source=input_array[band_idx],
                destination=crs_converted_arrays[band_idx],
                src_transform=input_transform,
                src_crs=current_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=input_nodata,
                dst_nodata=input_nodata
            )

        # 合并为多维数组
        current_array = np.stack(crs_converted_arrays, axis=0)
        current_transform = transform
        current_crs = target_crs

        # 重新计算转换后的有效范围
        converted_bounds = (
            transform.c,  # 左边界
            transform.f + height * transform.e,  # 下边界
            transform.c + width * transform.a,  # 右边界
            transform.f  # 上边界
        )
        current_bounds = converted_bounds

        print(f"坐标系转换完成")

    # --------------------------
    # 4. 精确重采样：匹配目标像素尺寸
    # --------------------------
    # 获取当前分辨率
    current_cell_x = current_transform[0]
    current_cell_y = current_transform[4]

    # 检查是否需要重采样
    if (abs(current_cell_x - target_cell_x) < 1e-6 and
            abs(current_cell_y - target_cell_y) < 1e-6):
        print("像素大小已匹配，无需重采样")
        processed_array = current_array
        processed_transform = current_transform
    else:
        print(f"正在进行精确重采样...")

        # 计算精确的目标尺寸
        width_range = current_bounds[2] - current_bounds[0]  # x方向范围
        height_range = current_bounds[3] - current_bounds[1]  # y方向范围

        # 计算新尺寸
        new_width = max(1, int(np.ceil(width_range / target_cell_x)) + buffer_pixels * 2)
        new_height = max(1, int(np.ceil(height_range / abs(target_cell_y))) + buffer_pixels * 2)

        print(f"重采样尺寸: {new_height}行 x {new_width}列")

        # 构建变换矩阵
        new_transform = Affine(
            target_cell_x,  # x方向像素大小（正）
            0,  # 无旋转
            current_bounds[0] - buffer_pixels * target_cell_x,  # 左边界
            0,  # 无旋转
            target_cell_y,  # y方向像素大小（负）
            current_bounds[3] + buffer_pixels * target_cell_x  # 上边界
        )

        print(f"新变换矩阵:\n{new_transform}")

        # 为每个波段执行重采样
        resampled_bands = []
        for band_idx in range(input_count):
            band_array = current_array[band_idx]
            resampled_band = np.full((new_height, new_width), input_nodata, dtype=np.float64)
            reproject(
                source=band_array,
                destination=resampled_band,
                src_transform=current_transform,
                src_crs=current_crs,
                dst_transform=new_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=input_nodata,
                dst_nodata=np.nan if input_nodata is None else input_nodata
            )
            resampled_bands.append(resampled_band)

        # 合并波段
        processed_array = np.stack(resampled_bands, axis=0)

        # 验证重采样结果（检查第一波段）
        first_resampled = processed_array[0] if input_count > 0 else processed_array
        resampled_valid = (first_resampled != input_nodata) if input_nodata is not None else ~np.isnan(first_resampled)
        if not np.any(resampled_valid):
            raise ValueError("重采样后无有效数据")

        print(f"重采样完成，有效像素数: {np.sum(resampled_valid)}")
        processed_transform = new_transform

    # --------------------------
    # 5. 保存结果并验证
    # --------------------------
    final_meta = {
        "driver": "GTiff",
        "height": processed_array.shape[1],  # 高度
        "width": processed_array.shape[2],  # 宽度
        "count": processed_array.shape[0],  # 波段数
        "dtype": np.float64,
        "crs": target_crs,
        "transform": processed_transform,
        "nodata": np.nan if input_nodata is None else input_nodata
    }

    if output_path:
        with rasterio.open(output_path, 'w', **final_meta) as dst:
            dst.write(processed_array)  # 写入所有波段
        print(f"结果已保存至: {output_path}")

        # 验证最终范围
        with rasterio.open(output_path) as f:
            final_bounds = f.bounds
            print(f"最终输出范围:")
            print(f"  左: {final_bounds.left:.2f}, 右: {final_bounds.right:.2f}")
            print(f"  下: {final_bounds.bottom:.2f}, 上: {final_bounds.top:.2f}")

    return processed_array, final_meta


# 向后兼容的单波段版本
def unify_dem_single_band(input_dem_path, target_dem_path, output_path=None, buffer_pixels=1):
    """
    单波段版本的统一分辨率函数（保持原有接口兼容性）
    """
    result_array, meta = unify_dem(input_dem_path, target_dem_path, output_path, buffer_pixels)

    # 如果是多波段，返回第一波段
    if result_array.ndim == 3 and result_array.shape[0] > 1:
        return result_array[0], meta
    else:
        return result_array, meta


def resample_to_target_resolution(input_tif, output_tif, target_resolution):
    """
    简单重采样函数（适用于投影坐标系）

    参数:
        input_tif: 输入TIF路径
        output_tif: 输出TIF路径
        target_resolution: 目标分辨率（米）
    """

    with rasterio.open(input_tif) as src:
        # 确保是投影坐标系
        if src.crs.is_geographic:
            raise ValueError("此函数仅适用于投影坐标系，请使用其他函数处理地理坐标系")

        # 原始像元大小（米）
        src_cell_width, src_cell_height = src.res

        # 计算输出尺寸（像素）
        target_height = int(src.height * src_cell_width / target_resolution)
        target_width = int(src.width * src_cell_width / target_resolution)

        # 读取并重采样数据（双线性插值）
        data = src.read(
            out_shape=(src.count, target_height, target_width),
            resampling=Resampling.bilinear
        )

        # 更新变换参数
        transform = src.transform * src.transform.scale(
            (src.width / target_width),
            (src.height / target_height)
        )

        # 保存输出
        with rasterio.open(
                output_tif,
                'w',
                driver='GTiff',
                height=target_height,
                width=target_width,
                count=src.count,
                dtype=data.dtype,
                crs=src.crs,
                transform=transform,
                nodata=src.nodata
        ) as dst:
            dst.write(data)

        print(f"重采样完成: {input_tif} → {output_tif}")
        print(f"原始尺寸: {src.height}×{src.width} → 目标尺寸: {target_height}×{target_width}")


def resample_geography_to_target_resolution(input_tif, output_tif, target_resolution):
    """
    地理坐标系重采样函数（自动转换到UTM）

    参数:
        input_tif: 输入TIF路径
        output_tif: 输出TIF路径
        target_resolution: 目标分辨率（米）
    """

    with rasterio.open(input_tif) as src:
        print(f"DEBUG: 输入文件: {input_tif}")
        print(f"DEBUG: 形状 (HxW): {src.height} x {src.width}")
        print(f"DEBUG: 变换矩阵 (src.transform):\n{src.transform}")
        print(f"DEBUG: 分辨率 (src.res): {src.res}")  # (pixel_width, pixel_height)
        print(f"DEBUG: 边界 (src.bounds): {src.bounds}")  # (left, bottom, right, top)
        print(f"DEBUG: CRS: {src.crs}")
        print(f"DEBUG: NoData值: {src.nodata}")
        print("-" * 20)
        # 1. 判断输入数据的坐标系类型
        if src.crs.is_geographic:
            # 地理坐标系（经纬度）需要先投影到UTM
            print("输入数据为地理坐标系（经纬度），将转换为UTM投影")

            # 计算数据中心经纬度，确定UTM区带
            center_lon = (src.bounds.left + src.bounds.right) / 2
            center_lat = (src.bounds.top + src.bounds.bottom) / 2
            utm_zone = int((center_lon + 180) / 6) + 1

            # 构建目标UTM坐标系
            if center_lat > 0:  # 北半球
                dst_crs = f"EPSG:326{utm_zone:02d}"
            else:  # 南半球
                dst_crs = f"EPSG:327{utm_zone:02d}"

            print(f"数据中心: {center_lon:.4f}°, {center_lat:.4f}°")
            print(f"目标UTM区带: {'北半球' if center_lat > 0 else '南半球'} Zone {utm_zone} ({dst_crs})")

            # 计算投影转换参数
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height,
                *src.bounds,
                resolution=target_resolution
            )

        else:
            # 投影坐标系直接重采样
            print(f"输入数据为投影坐标系：{src.crs}")
            print(f"原始分辨率：{src.res[0]:.2f}米 × {src.res[1]:.2f}米")

            # 直接设置目标分辨率
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height,
                *src.bounds,
                resolution=target_resolution
            )

            dst_crs = src.crs

        # 2. 更新输出栅格的元数据
        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata
        })

        # 3. 执行重采样
        with rasterio.open(output_tif, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,  # 显式指定源 NoData 值
                    dst_nodata=src.nodata  # 显式指定目标 NoData 值 (通常与源相同)
                )

        print(f"地理坐标系重采样完成: {input_tif} → {output_tif}")
        print(f"输出尺寸: {height}×{width}")

def downsample_tif(input_tif_path, output_tif_path, factor=2, resampling_method=Resampling.average):
    """
    对输入的 TIFF 文件进行下采样（降低分辨率），支持多波段。

    参数:
        input_tif_path (str): 输入 TIFF 文件的路径。
        output_tif_path (str): 输出下采样后 TIFF 文件的路径。
        factor (int, optional): 下采样因子。例如，factor=2 表示分辨率变为原来的 1/2，
                               像素数量变为原来的 1/(2*2)=1/4。 默认为 2。
        resampling_method (rasterio.enums.Resampling, optional): 重采样方法。
                                默认使用 Resampling.average (平均值法)，适合 DEM/连续数据。
                                其他选项包括 Resampling.bilinear, Resampling.nearest 等。
    """
    if factor < 1:
        raise ValueError("下采样因子必须大于或等于 1。")

    if factor == 1:
        print(f"警告: 下采样因子为 1，正在复制文件: {input_tif_path} -> {output_tif_path}")
        # 因子为1时，直接复制文件
        import shutil
        shutil.copy2(input_tif_path, output_tif_path)
        print(f"文件复制完成: {output_tif_path}")
        return

    try:
        with rasterio.open(input_tif_path, 'r') as src:
            # 获取原始元数据
            original_width = src.width
            original_height = src.height
            original_transform = src.transform
            count = src.count  # 波段数
            dtype = src.dtypes[0] # 假设所有波段数据类型一致，否则可能需要特殊处理

            print(f"开始处理: {input_tif_path}")
            print(f"  原始尺寸: {original_height} (高) x {original_width} (宽)")
            print(f"  原始分辨率: {src.res}")
            print(f"  波段数: {count}")
            print(f"  下采样因子: {factor}")

            # 计算新的宽度和高度
            new_width = max(1, original_width // factor)
            new_height = max(1, original_height // factor)

            print(f"  目标尺寸: {new_height} (高) x {new_width} (宽)")

            # 计算新的仿射变换矩阵 (Affine Transform)
            # 像素大小变为原来的 factor 倍
            scale_transform = Affine.scale(factor, factor)
            new_transform = original_transform * scale_transform

            # 准备输出文件的元数据
            profile = src.profile.copy()
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': new_transform,
                'compress': 'lzw' # 可选：添加压缩以减小文件大小
            })

            # 读取并重采样所有波段数据
            # out_shape 的顺序通常是 (bands, height, width)
            print(f"  正在读取并重采样数据 (使用 {resampling_method.name} 方法)...")
            data = src.read(
                out_shape=(count, new_height, new_width),
                resampling=resampling_method,
                out_dtype=dtype # 保持原始数据类型
            )

            # 写入输出文件
            print(f"  正在写入文件: {output_tif_path}...")
            with rasterio.open(output_tif_path, 'w', **profile) as dst:
                dst.write(data)
                # 可以根据需要复制其他元数据，如标签等

            print(f"  下采样完成: {output_tif_path}\n")

    except Exception as e:
        print(f"处理文件 {input_tif_path} 时出错: {e}")
        raise # 重新抛出异常以便调用者（如批处理函数）处理


def downsample_directory(input_dir, output_dir, factor=2, resampling_method=Resampling.average, pattern="*.tif"):
    """
    对指定目录下的所有符合模式的 TIFF 文件进行批量下采样。

    参数:
        input_dir (str): 包含待处理 TIFF 文件的输入目录路径。
        output_dir (str): 存放下采样后 TIFF 文件的输出目录路径。
                          该目录会被自动创建（如果不存在）。
        factor (int, optional): 下采样因子。默认为 2。
        resampling_method (rasterio.enums.Resampling, optional): 重采样方法。默认为 Resampling.average。
        pattern (str, optional): 匹配 TIFF 文件的 glob 模式。默认为 "*.tif"。
                                 可以改为 "*.[tT][iI][fF]" 来匹配 .tif 或 .TIF。
    """
    input_dir = os.path.abspath(input_dir) # 转换为绝对路径
    output_dir = os.path.abspath(output_dir) # 转换为绝对路径

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始批量处理...")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  匹配模式: {pattern}")
    print(f"  下采样因子: {factor}")
    print(f"  重采样方法: {resampling_method.name}")
    print("-" * 40)

    # 使用 glob 查找所有匹配的文件
    search_pattern = os.path.join(input_dir, pattern)
    tif_files = glob.glob(search_pattern)

    if not tif_files:
        print(f"警告: 在目录 '{input_dir}' 中未找到匹配模式 '{pattern}' 的文件。")
        return

    success_count = 0
    failure_count = 0

    for input_file_path in tif_files:
        try:
            # 获取不带路径的文件名
            filename = os.path.basename(input_file_path)
            # 构造输出文件路径
            output_file_path = os.path.join(output_dir, filename)

            # 调用单个文件处理函数
            downsample_tif(input_file_path, output_file_path, factor=factor, resampling_method=resampling_method)
            success_count += 1

        except Exception as e:
            print(f"批量处理中跳过文件 {input_file_path} (原因: {e})")
            failure_count += 1
            # 继续处理下一个文件，不因单个文件失败而停止整个过程

    print("-" * 40)
    print(f"批量处理完成。成功: {success_count}, 失败: {failure_count}")

# --- 新增的高级下采样函数 downsample_tif_advanced ---
def downsample_tif_advanced(input_tif_path, output_tif_path, factor=2,
                            sharpen_weight=0.2, smooth_sigma=1.0,
                            resampling_method=Resampling.bilinear,
                            temp_working_dtype=np.float64):
    """
    对输入的 TIFF 文件进行高级下采样，包含特征增强步骤。

    步骤：
    1. 读取图像数据。
    2. 对每个波段进行预处理：
       a. 计算拉普拉斯锐化 (Laplacian Sharpening)。
       b. 计算高斯平滑 (Gaussian Smoothing)。
       c. 组合处理结果（先锐化再平滑）。
    3. 对预处理后的数据进行下采样。
    4. 保存结果。

    参数:
        input_tif_path (str): 输入 TIFF 文件的路径。
        output_tif_path (str): 输出下采样后 TIFF 文件的路径。
        factor (int, optional): 下采样因子。默认为 2。
        sharpen_weight (float, optional): 锐化强度权重 (应用于拉普拉斯结果)。默认 0.2。
                                          负值意味着从原图中减去拉普拉斯，起到锐化作用。
        smooth_sigma (float, optional): 高斯平滑的标准差。默认 1.0。
        resampling_method (rasterio.enums.Resampling, optional): 下采样时使用的重采样方法。
                                默认为 Resampling.bilinear。
        temp_working_dtype (numpy.dtype, optional): 内部处理时临时使用的浮点数据类型。
                                                    默认为 np.float64，以保证精度。
    """
    if factor < 1:
        raise ValueError("下采样因子必须大于或等于 1。")
    if factor == 1:
        print(f"警告: 下采样因子为 1，正在复制文件: {input_tif_path} -> {output_tif_path}")
        import shutil
        shutil.copy2(input_tif_path, output_tif_path)
        print(f"文件复制完成: {output_tif_path}")
        return

    try:
        with rasterio.open(input_tif_path, 'r') as src:
            # 1. 获取原始元数据
            original_width = src.width
            original_height = src.height
            original_transform = src.transform
            count = src.count
            original_dtype = src.dtypes[0]
            crs = src.crs
            nodata = src.nodata
            nodata = np.nan if nodata is None else nodata

            print(f"[高级下采样] 开始处理: {input_tif_path}")
            print(f"  原始尺寸: {original_height} x {original_width}")
            print(f"  波段数: {count}")
            print(f"  下采样因子: {factor}")
            print(f"  锐化权重: {sharpen_weight}")
            print(f"  平滑 Sigma: {smooth_sigma}")

            # 2. 计算新尺寸和变换
            new_width = max(1, original_width // factor)
            new_height = max(1, original_height // factor)
            scale_transform = Affine.scale(factor, factor)
            new_transform = original_transform * scale_transform

            # 3. 读取原始数据
            print(f"  正在读取数据...")
            original_data = src.read()

            # --- 预处理阶段 ---
            print(f"  正在进行预处理 (特征增强)...")
            preprocessed_data = np.empty_like(original_data, dtype=temp_working_dtype)

            for band_index in range(count):
                band_original = original_data[band_index].astype(temp_working_dtype)
                nan_mask = None
                if nodata is not None:
                    # 使用 NaN 处理 nodata，便于 scipy 处理边界
                    nan_mask = np.isclose(band_original, nodata, atol=1e-8) | (band_original == nodata)
                    band_original_nan = band_original.copy()
                    band_original_nan[nan_mask] = np.nan
                else:
                     band_original_nan = band_original

                # a. 拉普拉斯锐化
                laplacian = ndimage.laplace(band_original_nan)
                sharpened_band = band_original_nan - sharpen_weight * laplacian

                # b. 高斯平滑 (对锐化后的图像)
                smoothed_band = ndimage.gaussian_filter(sharpened_band, sigma=smooth_sigma)

                clipped_band = np.clip(smoothed_band, 0, 255)

                # c. 存储预处理结果
                preprocessed_data[band_index] = clipped_band

            # --- 下采样阶段 ---
            print(f"  正在下采样数据 (尺寸: {original_height}x{original_width} -> {new_height}x{new_width})...")

            downsampled_data = np.empty((count, new_height, new_width), dtype=temp_working_dtype)

            for band_index in range(count):
                temp_src_profile = src.profile.copy()
                temp_src_profile.update({
                    'height': original_height,
                    'width': original_width,
                    'transform': original_transform,
                    'dtype': temp_working_dtype,
                    'nodata': np.nan
                })

                with rasterio.io.MemoryFile() as memfile:
                    with memfile.open(**temp_src_profile) as mem_src:
                        mem_src.write(preprocessed_data[band_index], 1)

                        temp_dst_profile = temp_src_profile.copy()
                        temp_dst_profile.update({
                            'height': new_height,
                            'width': new_width,
                            'transform': new_transform,
                            'dtype': temp_working_dtype
                        })

                        with rasterio.io.MemoryFile() as memfile_dst:
                            with memfile_dst.open(**temp_dst_profile) as mem_dst:
                                reproject(
                                    source=rasterio.band(mem_src, 1),
                                    destination=rasterio.band(mem_dst, 1),
                                    src_transform=original_transform,
                                    src_crs=crs,
                                    dst_transform=new_transform,
                                    dst_crs=crs,
                                    resampling=resampling_method,
                                    src_nodata=nodata,
                                    dst_nodata=nodata
                                )
                                downsampled_data[band_index] = np.clip(mem_dst.read(1), 0, 255)


            print(f"    波段 {band_index + 1} 预处理完成: ")
            print(type(downsampled_data))
            downsampled_data = downsampled_data.astype(original_dtype)
            print(np.min(downsampled_data))
            print(np.max(downsampled_data))
            print(np.mean(downsampled_data))

            # --- 后处理和写入 ---
            print(f"  正在准备写入数据...")

            profile = src.profile.copy()
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': new_transform,
                'compress': 'lzw',
            })

            print(f"  正在写入文件: {output_tif_path}...")
            with rasterio.open(output_tif_path, 'w', **profile) as dst:
                dst.write(downsampled_data)

            print(f"  [高级下采样] 完成: {output_tif_path}\n")

    except Exception as e:
        print(f"[高级下采样] 处理文件 {input_tif_path} 时出错: {e}")
        raise


# --- 新增的目录批量处理函数 downsample_directory_advanced ---
def downsample_directory_advanced(input_dir, output_dir, factor=2,
                                  sharpen_weight=0.2, smooth_sigma=1.0,
                                  resampling_method=Resampling.bilinear,
                                  temp_working_dtype=np.float64,
                                  pattern="*.tif"):
    """
    对指定目录下的所有符合模式的 TIFF 文件进行批量高级下采样。

    参数:
        input_dir (str): 包含待处理 TIFF 文件的输入目录路径。
        output_dir (str): 存放下采样后 TIFF 文件的输出目录路径。
        factor (int, optional): 下采样因子。默认为 2。
        sharpen_weight (float, optional): 传递给 downsample_tif_advanced 的锐化权重。
        smooth_sigma (float, optional): 传递给 downsample_tif_advanced 的高斯平滑 sigma。
        resampling_method (rasterio.enums.Resampling, optional): 重采样方法。
        temp_working_dtype (numpy.dtype, optional): 传递给 downsample_tif_advanced 的内部工作类型。
        pattern (str, optional): 匹配 TIFF 文件的 glob 模式。默认为 "*.tif"。
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"开始批量高级下采样...")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  匹配模式: {pattern}")
    print(f"  下采样因子: {factor}")
    print(f"  锐化权重: {sharpen_weight}")
    print(f"  平滑 Sigma: {smooth_sigma}")
    print(f"  重采样方法: {resampling_method.name}")
    print("-" * 50)

    search_pattern = os.path.join(input_dir, pattern)
    tif_files = glob.glob(search_pattern)

    if not tif_files:
        print(f"警告: 在目录 '{input_dir}' 中未找到匹配模式 '{pattern}' 的文件。")
        return

    success_count = 0
    failure_count = 0

    for input_file_path in tif_files:
        try:
            filename = os.path.basename(input_file_path)
            output_file_path = os.path.join(output_dir, filename)

            downsample_tif_advanced(
                input_tif_path=input_file_path,
                output_tif_path=output_file_path,
                factor=factor,
                sharpen_weight=sharpen_weight,
                smooth_sigma=smooth_sigma,
                resampling_method=resampling_method,
                temp_working_dtype=temp_working_dtype
            )
            success_count += 1

        except Exception as e:
            print(f"批量处理中跳过文件 {input_file_path} (原因: {e})")
            failure_count += 1

    print("-" * 50)
    print(f"批量高级下采样完成。成功: {success_count}, 失败: {failure_count}")

if __name__ == '__main__':
    # 示例使用
    # target_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Google\3951.tif"
    # input_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\DEM\3951.tif"
    # output_path = r"C:\Users\Kevin\Desktop\test.tif"

    # 使用最优的统一分辨率函数
    # unify_dem(input_path, target_path, output_path, buffer_pixels=1)

    # 或者使用其他函数
    # resample_to_target_resolution(input_path, output_path, 30)
    # resample_geography_to_target_resolution(input_path, output_path, 30)

    input_dir = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Google"
    output_dir = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Google_downsample_2"

    downsample_directory(input_dir, output_dir, factor=4)
    # downsample_directory_advanced(input_dir, output_dir, factor=8, sharpen_weight=0.9, smooth_sigma=1)