#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: modify_dem_batch
# @Time : 2026/1/19 10:00
# @Author : Your Name (Based on original by Kevin)
# @Describe: 批量修改多个TIF文件的数值，或单个文件处理


import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def calculate_robust_min_max_modified_zscore_vectorized(valid_data_array, threshold=3.5):
    """
    对 3D 数组 (N, H, W) 或 2D 数组 (H, W) 使用修正Z-Score方法计算稳健的最小值和最大值。
    N 是文件数量，H, W 是每个文件的高度和宽度。

    Args:
        valid_data_array (numpy.ndarray): 3D array of shape (N, H, W) or 2D array (H, W).
        threshold (float): 修正Z-Score的阈值，默认为3.5。

    Returns:
        tuple: (robust_min_array, robust_max_array, original_min_array, original_max_array)
               where each is a 1D array of shape (N,) or (1,) for 2D input.
    """
    # Flatten along the spatial dimensions (H*W) for efficient calculation
    # Shape becomes (N, H*W) or (1, H*W)
    flattened = valid_data_array.reshape(valid_data_array.shape[0], -1)

    # Calculate original min/max for comparison
    original_mins = np.min(flattened, axis=1)
    original_maxs = np.max(flattened, axis=1)

    # Calculate median along the flattened spatial dimension (axis=1)
    medians = np.median(flattened, axis=1, keepdims=True)  # Shape: (N, 1)

    # Calculate MAD along the flattened spatial dimension (axis=1)
    mads = np.median(np.abs(flattened - medians), axis=1, keepdims=True)  # Shape: (N, 1)

    # Avoid division by zero (if MAD is 0, min/max should be the median/unique value)
    # Create a mask for files where MAD is 0
    zero_mad_mask = (mads == 0)
    # Replace 0 MADs with 1 temporarily to avoid division errors, we'll handle them later
    mads_safe = np.where(zero_mad_mask, 1, mads)

    # Calculate modified Z-scores
    modified_z_scores = 0.6745 * (flattened - medians) / mads_safe  # Shape: (N, H*W)

    # Identify outliers based on threshold
    outlier_mask = np.abs(modified_z_scores) > threshold  # Shape: (N, H*W)

    # Apply outlier mask to get filtered data
    # We need to iterate over each file or use a more complex masking approach
    # A direct vectorized way to get min/max after masking is tricky with variable remaining elements
    # The most efficient way might still involve some loop, but on the file dimension which is hopefully smaller than H*W
    robust_mins = []
    robust_maxs = []

    for i in range(flattened.shape[0]):  # Loop over N (number of files)
        current_file_data = flattened[i]
        current_outlier_mask = outlier_mask[i]

        if zero_mad_mask[i, 0]:  # Handle case where MAD was 0 for this file
            print(f"      警告: 文件索引 {i} 的 MAD 为 0，可能所有有效数据点都相同或存在大量重复值。使用原始 min/max。")
            # If MAD is 0, the data is constant or has many duplicates; min and max are the same as unique values
            unique_vals = np.unique(current_file_data)
            if len(unique_vals) == 1:
                robust_mins.append(float(unique_vals[0]))
                robust_maxs.append(float(unique_vals[0]))
            else:
                # Should ideally not happen if MAD is 0, but just in case
                robust_mins.append(float(np.min(current_file_data)))
                robust_maxs.append(float(np.max(current_file_data)))
            continue

        filtered_current_data = current_file_data[~current_outlier_mask]

        if filtered_current_data.size == 0:
            print(f"      警告: 文件索引 {i} 使用修正Z-Score (阈值 {threshold}) 过滤后没有剩余数据。")
            robust_mins.append(None)  # Or raise an error for this specific file
            robust_maxs.append(None)
        else:
            robust_mins.append(float(np.min(filtered_current_data)))
            robust_maxs.append(float(np.max(filtered_current_data)))

    robust_mins_array = np.array(robust_mins)
    robust_maxs_array = np.array(robust_maxs)

    return robust_mins_array, robust_maxs_array, original_mins, original_maxs


def batch_modify_tifs_vectorized(tif_paths, input_matrices, output_paths=None, max_workers=4, zscore_threshold=3.5):
    """
    批量修改多个TIF文件的数值 (向量化 + 并行预处理)。
    首先并行读取所有TIF数据和计算稳健范围，然后进行向量化处理。
    要求所有TIF文件和对应的input_matrices尺寸必须相同。
    """
    if not (isinstance(tif_paths, (list, np.ndarray)) or isinstance(input_matrices, (list, np.ndarray))):
        raise ValueError("`tif_paths` 和 `input_matrices` 必须是列表或numpy数组。")

    if len(tif_paths) != len(input_matrices):
        raise ValueError("输入的TIF路径列表和矩阵列表长度必须相同。")

    num_files = len(tif_paths)
    if output_paths is not None:
        if not isinstance(output_paths, list) or len(output_paths) != num_files:
            raise ValueError("`output_paths` 必须是与输入路径列表长度相同的列表。")
    else:
        output_paths = tif_paths  # 原地修改

    print(f"开始批量处理 {num_files} 个文件...")

    # --- 步骤 1: 并行读取TIF信息和数据 (检查形状一致性) ---
    print("  1. 读取TIF元数据并检查尺寸一致性...")
    all_profiles = []
    all_masks = []
    all_shapes = []

    # Helper function to read single TIF info
    def read_tif_info_and_prepare(path):
        with rasterio.open(path) as src:
            shape = src.read(1).shape
            profile = src.profile.copy()
            nodata = src.nodata
            bounds = src.bounds
            crs = src.crs
            transform = src.transform
        return {
            'shape': shape,
            'profile': profile,
            'nodata': nodata,
            'bounds': bounds,
            'crs': crs,
            'transform': transform
        }

    # Parallel read of metadata
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_info = {executor.submit(read_tif_info_and_prepare, path): path for path in tif_paths}
        results_info = {}
        for future in as_completed(futures_info):
            path = futures_info[future]
            try:
                results_info[path] = future.result()
            except Exception as exc:
                print(f'    读取 {path} 元数据时生成异常: {exc}')
                return [False] * num_files  # Indicate failure for all

    # Retrieve ordered info and check shapes
    for path in tif_paths:
        info = results_info[path]
        all_shapes.append(info['shape'])
        all_profiles.append(info['profile'])
        all_masks.append(info['nodata'])  # Store nodata for later mask creation

    # Check if all TIF shapes are the same
    if len(set(all_shapes)) > 1:
        raise ValueError(f"错误: 发现不同尺寸的TIF文件。所有TIF文件尺寸必须相同。发现的尺寸: {set(all_shapes)}")

    # Check if input_matrices shapes match the TIF shapes
    for i, (tif_shape, mat_shape) in enumerate(zip(all_shapes, [m.shape for m in input_matrices])):
        if tif_shape != mat_shape:
            raise ValueError(f"错误: 第 {i} 个TIF文件的尺寸 {tif_shape} 与对应的输入矩阵尺寸 {mat_shape} 不匹配。")

    target_shape = all_shapes[0]  # All shapes are confirmed to be the same
    print(f"    所有TIF文件和输入矩阵尺寸均为: {target_shape}")

    # --- Step 2: Read all TIF data and prepare for vectorized min/max calculation ---
    print("  2. 并行读取TIF数据...")

    def read_tif_data(path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)  # Ensure float32 for consistency
            nodata = src.nodata
        return data, nodata

    all_data_blocks = [None] * num_files
    all_nodatas = [None] * num_files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_data = {executor.submit(read_tif_data, path): i for i, path in enumerate(tif_paths)}
        for future in as_completed(futures_data):
            idx = futures_data[future]
            try:
                data_block, nd = future.result()
                all_data_blocks[idx] = data_block
                all_nodatas[idx] = nd
            except Exception as exc:
                print(f'    读取 {tif_paths[idx]} 数据时生成异常: {exc}')
                return [False] * num_files

    # Stack data blocks into a 3D array (N, H, W)
    print("  3. 构建数据栈...")
    all_data_stacked = np.stack(all_data_blocks, axis=0)  # Shape: (N, H, W)
    all_nodatas_array = np.array(all_nodatas)  # Shape: (N,)

    # --- Step 4: Apply masks and calculate robust ranges ---
    print("  4. 应用NoData掩码并计算稳健范围 (Min/Max)...")
    all_masks_stacked = np.zeros_like(all_data_stacked, dtype=bool)
    for i in range(num_files):
        if all_nodatas_array[i] is not None:
            all_masks_stacked[i] = (all_data_stacked[i] == all_nodatas_array[i])

    # Extract only valid data points for each file within the stacked array
    # This is tricky because masks are per pixel now. We need to apply mask and then flatten each separately.
    # Vectorization across files for min/max *after* outlier removal is difficult due to variable remaining counts.
    # The best compromise for min/max might be the previous looped version INSIDE the vectorized function.
    # Let's call the function designed for (N, H, W) input.
    # First, get valid data for each layer (file)
    robust_mins_list = []
    robust_maxs_list = []
    original_mins_list = []
    original_maxs_list = []

    for i in range(num_files):
        current_data = all_data_stacked[i]
        current_mask = all_masks_stacked[i]
        if current_mask.any():
            valid_data_current = current_data[~current_mask]
        else:
            valid_data_current = current_data

        # Call the vectorized function on the single file's valid data (reshaped for the function)
        # Note: We pass the *unstacked* valid data here for efficiency per file
        valid_data_flat_current = valid_data_current.flatten()
        mins_arr, maxs_arr, orig_mins_arr, orig_maxs_arr = calculate_robust_min_max_modified_zscore_vectorized(
            valid_data_flat_current.reshape(1, -1), threshold=zscore_threshold
        )
        robust_mins_list.append(mins_arr[0])  # Get scalar from array of shape (1,)
        robust_maxs_list.append(maxs_arr[0])
        original_mins_list.append(orig_mins_arr[0])
        original_maxs_list.append(orig_maxs_arr[0])

    robust_mins = np.array(robust_mins_list)
    robust_maxs = np.array(robust_maxs_list)
    original_mins = np.array(original_mins_list)
    original_maxs = np.array(original_maxs_list)

    # Print comparison
    print("\n--- 范围比较 (原始 vs Z-Score 过滤后) ---")
    for i in range(num_files):
        print(
            f"  文件 {i}: 原始 [{original_mins[i]:.2f}, {original_maxs[i]:.2f}] -> Z-Score [{robust_mins[i]:.2f}, {robust_maxs[i]:.2f}]")
    print("----------------------------------------\n")

    # Check for failures
    if np.any(np.isnan(robust_mins)) or np.any(np.isnan(robust_maxs)):
        print("  错误: 某些文件计算稳健范围失败 (NaN found)。")
        return [False] * num_files

    print("  5. 向量化应用修改...")
    # Now perform the main calculation in a fully vectorized way
    # Shape: (N, H, W) = (N, H, W) * ((N,) - (N,)) + (N,)
    # Broadcasting rules: (N, 1, 1) is added/subtracted/multiplied to (N, H, W)
    range_diffs = robust_maxs[:, np.newaxis, np.newaxis] - robust_mins[:, np.newaxis, np.newaxis]  # Shape: (N, 1, 1)
    mins_reshaped = robust_mins[:, np.newaxis, np.newaxis]  # Shape: (N, 1, 1)

    # Perform the transformation: matrix * (max - min) + min
    # all_data_stacked is (N, H, W), input_matrices is (N, H, W) (already checked shape match)
    modified_data_stacked = np.array(input_matrices) * range_diffs + mins_reshaped  # Shape: (N, H, W)

    # Apply original masks back to the modified data
    modified_data_stacked[all_masks_stacked] = np.nan  # Or set to original nodata value if preferred
    # It's better to restore the original nodata from profiles if available
    for i in range(num_files):
        orig_nodata = all_profiles[i].get('nodata', np.nan)
        mask_for_file = all_masks_stacked[i]
        modified_data_stacked[i, mask_for_file] = orig_nodata

    print("  6. 并行写入输出文件...")

    # Write each modified block back to its file in parallel
    def write_output_task(args):
        idx, mod_data, out_pth, prof = args
        prof_update = prof.copy()
        prof_update.update({'dtype': mod_data.dtype})
        try:
            with rasterio.open(out_pth, 'w', **prof_update) as dst:
                dst.write(mod_data.astype(prof_update['dtype']), 1)
            return True
        except Exception as e:
            print(f"    写入 {out_pth} 时出错: {e}")
            return False

    write_tasks = [(i, modified_data_stacked[i], output_paths[i], all_profiles[i]) for i in range(num_files)]
    write_successes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_write = executor.map(write_output_task, write_tasks)
        write_successes = list(futures_write)

    print("批量处理完成。")
    return write_successes


# --- 示例用法 ---
if __name__ == "__main__":
    import glob
    import os

    # 假设你有这些数据
    folder_path = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Test\test"
    dem_paths = glob.glob(os.path.join(folder_path, "*.tif"))

    matrices = [np.random.rand(1024, 1024).astype(np.float32) for _ in range(len(dem_paths))]

    output_paths = [dem_path.replace(".tif", "_modified.tif") for dem_path in dem_paths]

    print("--- 开始向量化批量处理 ---")
    start_time = time.time()
    # 可以在这里调整 zscore_threshold
    results = batch_modify_tifs_vectorized(dem_paths, matrices, output_paths, max_workers=4, zscore_threshold=3.5)
    end_time = time.time()
    print(f"向量化批量处理结果: {results}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    # # 如果想原地修改，可以这样调用：
    # # results = batch_modify_tifs_vectorized(dem_paths, matrices, zscore_threshold=3.0) # output_paths 为 None