#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: get_slope_length
# @Time    : 2026/1/28 18:57
# @Author  : Kevin
# @Describe:

import os
import os.path as osp
import sys
current_file_path = osp.abspath(__file__)
sys.path.append(osp.join(osp.dirname(current_file_path), ".."))

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import whitebox

from LocalPath import White_Box_exe_dir

wbt = whitebox.WhiteboxTools()
wbt.exe_path = White_Box_exe_dir

def check_and_project_dem(input_path):
    """
    检查坐标系，如果是地理坐标系则自动投影到UTM
    返回: (处理后的路径, 分辨率米, 是否为临时文件)
    """
    with rasterio.open(input_path) as src:
        original_crs = src.crs
        print(f"原始坐标系: {original_crs}")

        # 检查是否已经是投影坐标系
        if not original_crs.is_geographic:
            print("数据已经是投影坐标系，无需转换")
            # 计算像元尺寸（取平均）
            res = (abs(src.transform[0]) + abs(src.transform[4])) / 2
            return input_path, res, False

        # 地理坐标系需要投影
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.top + bounds.bottom) / 2

        # 计算UTM带号
        utm_zone = int((center_lon + 180) / 6) + 1
        epsg_code = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
        utm_crs = f'EPSG:{epsg_code}'

        print(f"正在投影到 {utm_crs} (UTM Zone {utm_zone})...")

        transform, width, height = calculate_default_transform(
            original_crs, utm_crs, src.width, src.height, *bounds
        )

        dir_name = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        projected_path = os.path.join(dir_name, f"{base_name}_utm.tif")

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': utm_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'compress': 'lzw'
        })

        with rasterio.open(projected_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=original_crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.bilinear
                )

        res = transform[0]  # 分辨率（米）
        print(f"投影完成，分辨率: {res:.2f} 米")
        return projected_path, res, True


def calculate_slope_and_ls(dem_path, resolution, output_dir=None):
    """
    计算坡长和LS因子（土壤侵蚀计算所需）
    """
    if output_dir is None:
        output_dir = os.path.dirname(dem_path)

    base_name = os.path.splitext(os.path.basename(dem_path))[0]

    # 定义输出文件路径
    filled_dem = os.path.join(output_dir, f"{base_name}_filled.tif")
    flow_dir = os.path.join(output_dir, f"{base_name}_flowdir.tif")
    slope_degrees = os.path.join(output_dir, f"{base_name}_slope_deg.tif")  # 坡度（度）
    slope_length_path = os.path.join(output_dir, f"{base_name}_slopelength.tif")
    ls_factor_path = os.path.join(output_dir, f"{base_name}_LS_factor.tif")  # LS因子

    # 步骤1: 填洼（修正后的函数名）
    print("\n步骤 1/5: 填洼处理 (Whitebox)...")
    wbt.fill_depressions_wang_and_liu(
        dem_path,
        filled_dem,
        flat_increment=0.001
    )

    # 步骤2: 计算D8流向
    print("步骤 2/5: 计算D8流向...")
    wbt.d8_pointer(
        filled_dem,
        flow_dir,
        esri_pntr=False
    )

    # 步骤3: 计算坡度（用于后续的S因子）
    print("步骤 3/5: 计算坡度...")
    wbt.slope(
        filled_dem,
        slope_degrees,
        units="degrees"
    )

    # 步骤4: 计算坡长（基于D8流向的流线追踪）
    print("步骤 4/5: 计算坡长...")

    with rasterio.open(flow_dir) as dir_src:
        flow_dir_data = dir_src.read(1)
        profile = dir_src.profile

    rows, cols = flow_dir_data.shape

    # D8流向编码映射 (1,2,4,8,16,32,64,128) -> (dx, dy, 距离系数)
    flow_map = {
        1: (0, 1, 1.0),  # East
        2: (1, 1, 1.414),  # SE
        4: (1, 0, 1.0),  # South
        8: (1, -1, 1.414),  # SW
        16: (0, -1, 1.0),  # West
        32: (-1, -1, 1.414),  # NW
        64: (-1, 0, 1.0),  # North
        128: (-1, 1, 1.414)  # NE
    }

    # 初始化坡长栅格
    slope_length = np.zeros((rows, cols), dtype=np.float32)

    # 方法：从每个像元向下游追踪直到分水岭，累积距离
    # 使用迭代方法（效率更高的算法）

    # 先初始化：所有像元的初始坡长设为分辨率（源头像元）
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if flow_dir_data[i, j] != 0:
                slope_length[i, j] = resolution  # 最少算作一个像元

    # 迭代更新：对于每个像元，如果上游有来水，则坡长 = 上游坡长 + 当前步长
    # 需要按高程从低到高处理（这里简化为多次迭代直至收敛）
    print("  正在追踪流线（迭代计算）...")

    max_iterations = 500  # 最大迭代次数（防止无限循环）
    for iteration in range(max_iterations):
        print(f"  迭代次数: {iteration + 1}")
        new_slope_length = slope_length.copy()
        max_change = 0

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                direction = flow_dir_data[i, j]
                if direction == 0 or direction not in flow_map:
                    continue

                # 找到下游像元
                di, dj, dist_factor = flow_map[direction]
                ni, nj = i + di, j + dj

                if 0 <= ni < rows and 0 <= nj < cols and flow_dir_data[ni, nj] != 0:
                    step_dist = resolution * dist_factor
                    # 下游像元的坡长 = max(自身原值, 当前像元坡长 + 步长)
                    proposed_length = slope_length[i, j] + step_dist
                    if proposed_length > new_slope_length[ni, nj]:
                        change = proposed_length - new_slope_length[ni, nj]
                        if change > max_change:
                            max_change = change
                        new_slope_length[ni, nj] = proposed_length

        slope_length = new_slope_length

        # 如果变化很小，提前收敛
        if max_change < resolution * 0.01:  # 变化小于1%分辨率
            print(f"  迭代收敛于第 {iteration + 1} 次")
            break

    # 应用最大坡长限制（USLE通常限制在200-300米）
    MAX_SLOPE_LENGTH = 300  # 米，可根据需要调整
    slope_length = np.clip(slope_length, 0, MAX_SLOPE_LENGTH)

    # 保存坡长结果
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=0)
    with rasterio.open(slope_length_path, 'w', **profile) as dst:
        dst.write(slope_length, 1)

    print(f"  ✓ 坡长栅格已保存: {slope_length_path}")
    print(f"    坡长范围: {np.min(slope_length[slope_length > 0]):.1f} - {np.max(slope_length):.1f} 米")

    # 步骤5: 计算LS因子（L*S）
    print("步骤 5/5: 计算LS因子（用于土壤侵蚀模型）...")

    # 读取坡度数据（度）
    with rasterio.open(slope_degrees) as slope_src:
        slope_deg = slope_src.read(1)

    # 将坡度转换为弧度计算sin
    slope_rad = np.deg2rad(slope_deg)
    sin_slope = np.sin(slope_rad)

    # 计算 m 指数（根据坡度变化）
    # 经验公式：坡度<1°:0.2, 1-3°:0.3, 3-5°:0.4, >5°:0.5
    m = np.piecewise(slope_deg,
                     [slope_deg < 1, (slope_deg >= 1) & (slope_deg < 3),
                      (slope_deg >= 3) & (slope_deg < 5), slope_deg >= 5],
                     [0.2, 0.3, 0.4, 0.5])

    # 计算 L 因子: (λ/22.13)^m
    # 避免除以零，只计算坡长>0的区域
    L = np.zeros_like(slope_length)
    mask = slope_length > 0
    L[mask] = np.power(slope_length[mask] / 22.13, m[mask])

    # 计算 S 因子 (McCool et al., 1987)
    # θ < 9% (约5.14°): S = 10.8 * sin(θ) + 0.03
    # θ >= 9%: S = 16.8 * sin(θ) - 0.5
    S = np.zeros_like(slope_deg)
    gentle = slope_deg < 5.14  # < 9%
    steep = ~gentle

    S[gentle] = 10.8 * sin_slope[gentle] + 0.03
    S[steep] = 16.8 * sin_slope[steep] - 0.5

    # 确保S非负
    S = np.maximum(S, 0)

    # LS = L * S
    LS = L * S

    # 保存LS因子
    with rasterio.open(ls_factor_path, 'w', **profile) as dst:
        dst.write(LS.astype(np.float32), 1)

    print(f"  ✓ LS因子栅格已保存: {ls_factor_path}")
    print(f"    LS因子范围: {np.min(LS[LS > 0]):.4f} - {np.max(LS):.4f}")

    # 清理中间文件（可选，如需保留流向和填洼结果，注释掉以下代码）
    for temp_file in [filled_dem, flow_dir, slope_degrees]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return slope_length_path, ls_factor_path


def main(input_file):
    print(f"处理文件: {input_file}")

    # 1. 坐标系检查与投影
    processed_dem, resolution, is_temp = check_and_project_dem(input_file)

    try:
        # 2. 计算坡长和LS因子
        sl_path, ls_path = calculate_slope_and_ls(processed_dem, resolution)

        print(f"\n{'=' * 50}")
        print("处理完成！输出文件：")
        print(f"1. 坡长 (Slope Length): {sl_path}")
        print(f"   - 单位: 米")
        print(f"   - 含义: 每个像元到分水岭的流线距离")
        print(f"")
        print(f"2. LS因子 (LS Factor): {ls_path}")
        print(f"   - 单位: 无量纲")
        print(f"   - 含义: 用于USLE/RUSLE土壤侵蚀计算 [A = R*K*L*S*C*P]")
        print(f"{'=' * 50}")

        # 读取并显示统计信息
        for name, path in [("坡长", sl_path), ("LS因子", ls_path)]:
            with rasterio.open(path) as src:
                data = src.read(1)
                data = data[data > 0]  # 忽略0值
                print(f"\n{name}统计:")
                print(f"  均值: {np.mean(data):.2f}")
                print(f"  标准差: {np.std(data):.2f}")
                print(f"  最大值: {np.max(data):.2f}")
                print(f"  最小值: {np.min(data):.2f}")

    finally:
        # 清理临时投影文件
        if is_temp and os.path.exists(processed_dem):
            os.remove(processed_dem)
            print(f"\n已清理临时文件: {processed_dem}")


if __name__ == "__main__":
    input_path = r"D:\研究文件\ResearchData\USA\CopernicusDEM\GeoDAR_v11_dams_of_USA_group1_paired\0.tif"

    if os.path.exists(input_path):
        main(input_path)
    else:
        print(f"文件不存在: {input_path}")