#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: conver_coordinate
# @Time    : 2026/1/22 11:35
# @Author  : Kevin
# @Describe:

import pyproj
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS


def convert_dem_egm2008_to_navd88_foot(
        src_dem_path,  # 源DEM文件路径（WGS84+EGM2008，米）
        dst_dem_path,  # 输出DEM文件路径（NAD83 HARN+NAVD88，英尺）
        lon_center=-82.5,  # DEM中心经度（替换为你的DEM实际中心经度）
        lat_center=39.5  # DEM中心纬度（替换为你的DEM实际中心纬度）
):
    """
    将EGM2008（米）的DEM转换为NAVD88（US Survey Foot）的DEM，水平坐标系同步转为NAD83 HARN Ohio South
    """
    # ---------------------- 1. 定义坐标系 ----------------------
    # 源坐标系：WGS84（水平） + EGM2008（垂直，米）
    crs_wgs84 = CRS.from_epsg(4326)
    crs_egm2008 = pyproj.CRS.from_epsg(3855)

    # 目标坐标系：NAD83 HARN Ohio South（水平，英尺） + NAVD88（垂直，英尺）
    crs_nad83_harn_ohio = CRS.from_epsg(3754)
    crs_navd88 = pyproj.CRS.from_epsg(5703)

    # ---------------------- 2. 读取源DEM ----------------------
    try:
        with rasterio.open(src_dem_path) as src:
            # 获取源DEM元数据
            src_meta = src.meta.copy()
            src_data = src.read(1)  # 读取高程数据（单波段）
            src_gt = src.transform  # 源地理变换（米）
            src_crs = src.crs

            # 跳过无数据值（通常为-9999或nan）
            nodata = src.nodata if src.nodata is not None else -9999.0
            src_data[src_data == nodata] = np.nan

            # ---------------------- 3. 水平坐标转换（WGS84→NAD83 HARN Ohio South） ----------------------
            # 计算目标地理变换和尺寸
            transform, width, height = calculate_default_transform(
                crs_wgs84, crs_nad83_harn_ohio, src.width, src.height, *src.bounds
            )

            # 更新输出元数据（水平坐标系）
            dst_meta = src_meta.copy()
            dst_meta.update({
                'crs': crs_nad83_harn_ohio,
                'transform': transform,
                'width': width,
                'height': height,
                'dtype': np.float32  # 保证高程精度
            })

            # ---------------------- 4. 垂直基准+单位转换（EGM2008米→NAVD88英尺） ----------------------
            # 定义垂直基准转换器（EGM2008→NAVD88）
            transform_vertical = pyproj.Transformer.from_crs(
                f"EPSG:4326+{crs_egm2008.to_epsg()}",
                f"EPSG:4326+{crs_navd88.to_epsg()}",
                always_xy=True
            )

            # 对有效高程值进行转换（逐像素，或用中心坐标近似（效率更高））
            # 注：逐像素转换更精准但慢，中心坐标近似适合小范围DEM（如你的数据）
            # --- 方式1：中心坐标近似（效率高，小范围DEM足够精准） ---
            _, _, dst_elevation_m_navd88 = transform_vertical.transform(
                lon_center, lat_center, src_data
            )
            # --- 方式2：逐像素转换（精准但慢，注释掉方式1后启用） ---
            # # 生成每个像素的经纬度
            # rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
            # xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            # lons = np.array(xs)
            # lats = np.array(ys)
            # # 垂直基准转换
            # _, _, dst_elevation_m_navd88 = transform_vertical.transform(
            #     lons, lats, src_data
            # )

            # 单位转换：NAVD88（米）→NAVD88（US Survey Foot）
            # US Survey Foot = 0.3048006096012192 米 → 1米 = 1/0.3048006096012192 英尺
            us_survey_foot_per_meter = 1.0 / 0.3048006096012192
            dst_elevation_foot = dst_elevation_m_navd88 * us_survey_foot_per_meter

            # 恢复无数据值
            dst_elevation_foot[np.isnan(dst_elevation_foot)] = nodata

            # ---------------------- 5. 写入输出DEM（带正确元数据） ----------------------
            with rasterio.open(dst_dem_path, 'w', **dst_meta) as dst:
                # 写入高程数据
                dst.write(dst_elevation_foot.astype(np.float32), 1)

                # 补充单位元数据（关键：让ArcGIS/QGIS识别为英尺）
                dst.update_tags(
                    VERTICAL_UNIT="US Survey Foot",
                    HORIZONTAL_UNIT="US Survey Foot",
                    UNIT_CONVERSION="1 US Survey Foot = 0.3048006096012192 meters",
                    SOURCE_CRS="WGS84+EGM2008 (meters)",
                    TARGET_CRS="NAD83(HARN) Ohio South+NAVD88 (US Survey Foot)"
                )

        print(f"✅ 转换完成！输出文件：{dst_dem_path}")
        print(f"📌 验证：源高程272.5米 → 转换后高程约 {dst_elevation_foot[np.isclose(src_data, 272.5)][0]:.2f} 英尺")

    except FileNotFoundError:
        print(f"❌ 错误：源文件不存在 → {src_dem_path}")
    except Exception as e:
        print(f"❌ 转换失败：{str(e)}")


# ---------------------- 主函数调用（替换为你的文件路径） ----------------------
if __name__ == "__main__":
    # 替换为你的源DEM和输出DEM路径
    SRC_DEM_PATH = r"C:\Users\Kevin\Desktop\Copernicus_DSM_10_N39_00_W083_00_DEM.tif"
    DST_DEM_PATH = r"C:\Users\Kevin\Desktop\Copernicus_DSM_10_N39_00_W083_00_DEM_Convert.tif"

    # 替换为你的DEM中心经纬度（从ArcGIS/QGIS中获取）
    DEM_CENTER_LON = -82.5
    DEM_CENTER_LAT = 39.5

    # 执行转换
    convert_dem_egm2008_to_navd88_foot(
        src_dem_path=SRC_DEM_PATH,
        dst_dem_path=DST_DEM_PATH,
        lon_center=DEM_CENTER_LON,
        lat_center=DEM_CENTER_LAT
    )