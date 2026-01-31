#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: get_flow_direction
# @Time    : 2025/8/27 10:11
# @Author  : Kevin
# @Describe: 获取Tif文件的流向累计
import os
import whitebox
from osgeo import gdal
import gc

# --------------------------
# 新增：关闭TIFF文件句柄的核心函数（仅关文件，无额外判断）
# --------------------------
def close_tiff_handle(tif_path):
    """显式关闭TIFF文件句柄，释放内存/文件占用"""
    if os.path.exists(tif_path):
        try:
            # GDAL打开并立即置空，强制关闭句柄
            ds = gdal.Open(tif_path)
            if ds:
                ds = None  # GDAL关闭句柄的核心操作
            gc.collect()  # 强制垃圾回收，释放内存
        except Exception as e:
            # 仅打印警告，不中断流程（按你的要求不加额外判断）
            print(f"关闭TIFF句柄警告: {tif_path} - {str(e)}")

# --------------------------
# 原有：参数设置（仅新增关闭句柄逻辑）
# --------------------------
def calculate_flow_accumulation(dem_path, flow_accum_path, temp_dir=None, visible=False, wbt_exe_dir=r"C:\Users\Kevin\anaconda3\envs\geo-torch\Lib\site-packages\whitebox\WBT"):
    """
    计算DEM的汇流累积量，支持自定义临时文件路径，仅删除中间文件不删目录

    参数:
        dem_path: 输入DEM的文件路径（.tif格式）
        flow_accum_path: 汇流累积量输出文件路径（.tif格式）
        temp_dir: 临时文件存储目录（None则使用dem_path同目录，非None则创建该目录）
    """
    # 初始化Whitebox工具
    wbt = whitebox.WhiteboxTools()
    if wbt_exe_dir is not None:
        wbt.exe_path = wbt_exe_dir
        if not os.path.exists(wbt_exe_dir):
            raise FileNotFoundError(f"The specified WhiteboxTools executable directory does not exist: {wbt_exe_dir}")

    wbt.verbose = visible  # 关闭冗余日志

    intermediate_files = []  # 收集所有中间文件路径，用于后续删除
    if temp_dir is None:
        # temp_dir为None时，中间文件放在input_dem同路径下
        temp_dir = os.path.dirname(dem_path)
    else:
        # temp_dir不为None时，创建指定目录（确保存在）
        os.makedirs(temp_dir, exist_ok=True)

    filled_dem_path = os.path.join(temp_dir, "filled_dem.tif")
    intermediate_files.append(filled_dem_path)

    flow_dir_path = os.path.join(temp_dir, "flow_direction.tif")
    intermediate_files.append(flow_dir_path)

    intermediate_fill_path = os.path.join(temp_dir, "intermediate_filled.tif")
    intermediate_files.append(intermediate_fill_path)

    # 1. DEM预处理：填洼
    fill_result1 = wbt.fill_depressions(
        dem=dem_path,
        output=intermediate_fill_path,
        fix_flats=True,
        flat_increment=0.01,
        max_depth=None  # 不限制填充深度，处理所有洼地
    )
    if fill_result1 != 0 or not os.path.exists(intermediate_fill_path):
        error_msg = f"The first stage of depression filling failed with code: {fill_result1}"
        if fill_result1 == 1:
            error_msg += " (Missing required input file)"
        elif fill_result1 == 2:
            error_msg += " (Invalid input value/combination)"
        elif fill_result1 == 3:
            error_msg += " (Error in input file)"
        elif fill_result1 == 4:
            error_msg += " (I/O error)"
        elif fill_result1 == 5:
            error_msg += " (Unsupported data type)"
        elif fill_result1 == 32:
            error_msg += " (Timeout error)"
        print(f"dem_path: {dem_path}")
        print(f"intermediate_fill_path: {intermediate_fill_path}")
        raise RuntimeError(error_msg)

    close_tiff_handle(intermediate_fill_path)
    close_tiff_handle(intermediate_fill_path)

    # 2. DEM预处理：破坝（输入改为修复后的文件）
    breach_result1 = wbt.breach_depressions(
        dem=intermediate_fill_path,  # 关键：使用修复后的文件
        output=filled_dem_path,
        max_depth=10.0,  # 限制浅洼地的最大处理深度（米）
        max_length=50,  # 限制 breach 通道长度（网格单元数）
        flat_increment=0.001,  # 保持平坦区域连续性
        fill_pits=True  # 填充单像素坑洼
    )
    if breach_result1 != 0 or not os.path.exists(filled_dem_path):
        error_msg = f"The first DEM peak shaving process failed with code: {breach_result1}"
        if breach_result1 == 1:
            error_msg += " (Missing required input file)"
        elif breach_result1 == 2:
            error_msg += " (Invalid input value/combination)"
        elif breach_result1 == 3:
            error_msg += " (Error in input file)"
        elif breach_result1 == 4:
            error_msg += " (I/O error)"
        elif breach_result1 == 5:
            error_msg += " (Unsupported data type)"
        elif breach_result1 == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    close_tiff_handle(intermediate_fill_path)
    close_tiff_handle(filled_dem_path)

    # 3. 计算流向
    flow_dir_result = wbt.d8_pointer(
        dem=filled_dem_path,
        output=flow_dir_path,
        esri_pntr=False
    )
    if flow_dir_result != 0 or not os.path.exists(flow_dir_path):
        error_msg = f"Flow direction calculation failed with code: {flow_dir_result}"
        if flow_dir_result == 1:
            error_msg += " (Missing required input file)"
        elif flow_dir_result == 2:
            error_msg += " (Invalid input value/combination)"
        elif flow_dir_result == 3:
            error_msg += " (Error in input file)"
        elif flow_dir_result == 4:
            error_msg += " (I/O error)"
        elif flow_dir_result == 5:
            error_msg += " (Unsupported data type)"
        elif flow_dir_result == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    close_tiff_handle(filled_dem_path)
    close_tiff_handle(flow_dir_path)

    # 4. 计算汇流累积量（最终需要保存的结果）
    output_dir = os.path.dirname(flow_accum_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    accum_result = wbt.d8_flow_accumulation(
        i=flow_dir_path,
        output=flow_accum_path,
        pntr=True,
        esri_pntr=False
    )
    if accum_result != 0 or not os.path.exists(flow_accum_path):
        error_msg = f"The calculation of the accumulated amount of the confluence failed with code: {accum_result}"
        if accum_result == 1:
            error_msg += " (Missing required input file)"
        elif accum_result == 2:
            error_msg += " (Invalid input value/combination)"
        elif accum_result == 3:
            error_msg += " (Error in input file)"
        elif accum_result == 4:
            error_msg += " (I/O error)"
        elif accum_result == 5:
            error_msg += " (Unsupported data type)"
        elif accum_result == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    close_tiff_handle(flow_dir_path)
    close_tiff_handle(flow_accum_path)

    # 删除中间文件前，先关闭所有中间文件句柄（新增）
    for file_path in intermediate_files:
        close_tiff_handle(file_path)
    # 原有：删除中间文件
    for file_path in intermediate_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 输入DEM路径
    input_dem = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\WMG.tif"
    # 输出汇流累积量路径
    output_accum = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\WMG_ACCUM.tif"

    # 调用示例1：temp_dir=None（中间文件存在input_dem同目录，仅删文件）
    calculate_flow_accumulation(input_dem, output_accum)