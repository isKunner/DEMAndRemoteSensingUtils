#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: utils
# @Time    : 2025/8/10 13:47
# @Author  : Kevin
# @Describe: 地理坐标系的处理工具
import os
import glob
import shutil
import subprocess
import sys

import geopandas as gpd
from osgeo import osr, gdal
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def add_vertical_datum_with_backup(tif_path, vertical_datum="EGM2008"):
    """
    安全地为 GeoTIFF DEM 添加垂直基准：
    - 自动备份原文件（仅当备份不存在时）
    - 修改原文件的 CRS 元数据（不改变像素值）
    - 支持 EGM2008 或 EGM96
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"输入文件不存在: {tif_path}")

    # 1. 确定复合 CRS
    crs_map = {
        "EGM2008": "EPSG:4326+3855",
        "EGM96": "EPSG:4326+5773"
    }
    if vertical_datum not in crs_map:
        raise ValueError("仅支持 'EGM2008' 或 'EGM96'")
    compound_crs = crs_map[vertical_datum]

    # 2. 自动创建备份（如果还没有）
    backup_path = tif_path.replace(".tif", "_backup.tif")
    if not os.path.exists(backup_path):
        print(f"📁 正在创建备份: {backup_path}")
        shutil.copy2(tif_path, backup_path)
        print("✅ 备份完成！")
    else:
        print(f"ℹ️ 备份已存在，跳过: {backup_path}")

    # 3. 找到 gdal_edit.py
    if sys.platform == "win32":
        gdal_edit = os.path.join(os.path.dirname(sys.executable), "Scripts", "gdal_edit.py")
    else:
        gdal_edit = os.path.join(os.path.dirname(sys.executable), "gdal_edit.py")

    if not os.path.exists(gdal_edit):
        gdal_edit = "gdal_edit.py"  # 假设在 PATH 中

    # 4. 执行 gdal_edit
    cmd = [sys.executable, gdal_edit, "-a_srs", compound_crs, tif_path]
    print(f"\n🔧 正在为 {os.path.basename(tif_path)} 添加垂直基准: {vertical_datum}")
    print(f"命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ 垂直基准已成功添加！")
        print(f"📌 文件现在使用复合 CRS: {compound_crs}")
    else:
        print("❌ GDAL 命令失败:")
        print(result.stderr)
        print("\n⚠️ 但你的原始数据是安全的！备份位于:")
        print(backup_path)
        raise RuntimeError("添加垂直基准失败")


def reproject_raster_file(input_path, output_path, target_crs):
    """
    Reprojects a raster file (e.g., GeoTIFF) to a new coordinate reference system (CRS).

    Args:
        input_path (str): LocalPath to the input raster file (.tif, .tiff).
        output_path (str): LocalPath where the reprojected file will be saved.
        target_crs (str or int): Target CRS in EPSG code (int) or WKT string (str).
                                 Example: 4326 for WGS84 (lat/lon), 3857 for Web Mercator.
    """
    print(f"Reading input file: {input_path}")
    with rasterio.open(input_path) as src:
        # 1. Get source CRS and other properties
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        src_count = src.count  # Number of bands
        src_dtype = src.dtypes[0]  # Data type (e.g., float32, uint16)

        print(f"Source CRS: {src_crs}")
        print(f"Source dimensions: {src_width} x {src_height}")
        print(f"Source data type: {src_dtype}")

        # 2. Calculate the transform and dimensions for the destination
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, target_crs, src_width, src_height, *src.bounds
        )

        print(f"Target CRS: {target_crs}")
        print(f"Calculated destination dimensions: {dst_width} x {dst_height}")
        print(f"Calculated destination transform: {dst_transform}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 3. Open source again and create destination dataset for writing
    print(f"Starting reprojection...")
    with rasterio.open(input_path) as src:
        # Prepare metadata for the output dataset
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height
        })

        # Open output file for writing
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            # Iterate over all bands (if there are multiple)
            for i in range(1, src_count + 1):
                # Reproject each band individually
                reproject(
                    source=rasterio.band(src, i),  # Source band
                    destination=rasterio.band(dst, i),  # Destination band
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  # Choose appropriate resampling method
                )
    print("Reprojection completed successfully!")


def batch_set_coordinate_system(tif_dir, srs_name="WGS84"):
    """
    批量为目录下的所有TIF文件设置坐标系

    参数:
        tif_dir: TIF文件所在目录路径
        srs_name: 坐标系名称，默认为"WGS84"，支持EPSG代码如"EPSG:4326"等
    """
    # 支持的TIF文件扩展名
    extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tif_files = []

    # 获取目录下所有TIF文件
    for ext in extensions:
        tif_files.extend(glob.glob(os.path.join(tif_dir, ext)))
        tif_files.extend(glob.glob(os.path.join(tif_dir, ext.lower())))

    # 创建空间参考系统对象
    srs = osr.SpatialReference()

    # 根据输入设置坐标系
    if srs_name.upper() == "WGS84":
        srs.SetWellKnownGeogCS("WGS84")
    elif srs_name.upper().startswith("EPSG:"):
        epsg_code = int(srs_name.split(':')[1])
        srs.ImportFromEPSG(epsg_code)
    else:
        # 尝试直接导入坐标系定义
        try:
            srs.SetWellKnownGeogCS(srs_name)
        except:
            try:
                srs.ImportFromEPSG(int(srs_name))
            except:
                raise ValueError(f"无法识别的坐标系: {srs_name}")

    # 设置坐标系的WKT字符串
    wkt = srs.ExportToWkt()

    print(f"正在处理目录: {tif_dir}")
    print(f"目标坐标系: {srs_name}")
    print(f"共找到 {len(tif_files)} 个TIF文件")

    success_count = 0
    fail_count = 0

    for tif_path in tif_files:
        try:
            # 打开TIF文件
            dataset = gdal.Open(tif_path, gdal.GA_Update)
            if dataset is not None:
                # 设置坐标系
                dataset.SetProjection(wkt)
                print(f"✓ 已为 {os.path.basename(tif_path)} 设置坐标系为 {srs_name}")
                success_count += 1
            else:
                print(f"✗ 无法打开文件: {tif_path}")
                fail_count += 1
        except Exception as e:
            print(f"✗ 处理文件 {tif_path} 时出错: {str(e)}")
            fail_count += 1
        finally:
            # 确保数据集被正确关闭
            if dataset:
                dataset = None

    print(f"\n处理完成！成功: {success_count}, 失败: {fail_count}")

def set_coordinate_system_for_tif(tif_path, srs_name="WGS84"):
    """
    为单个TIF文件设置坐标系

    参数:
        tif_path: TIF文件路径
        srs_name: 坐标系名称，默认为"WGS84"
    """
    # 创建空间参考系统对象
    srs = osr.SpatialReference()

    # 根据输入设置坐标系
    if srs_name.upper() == "WGS84":
        srs.SetWellKnownGeogCS("WGS84")
    elif srs_name.upper().startswith("EPSG:"):
        epsg_code = int(srs_name.split(':')[1])
        srs.ImportFromEPSG(epsg_code)
    else:
        # 尝试直接导入坐标系定义
        try:
            srs.SetWellKnownGeogCS(srs_name)
        except:
            try:
                srs.ImportFromEPSG(int(srs_name))
            except:
                raise ValueError(f"无法识别的坐标系: {srs_name}")

    # 设置坐标系的WKT字符串
    wkt = srs.ExportToWkt()

    # 打开TIF文件并设置坐标系
    dataset = gdal.Open(tif_path, gdal.GA_Update)
    if dataset is not None:
        dataset.SetProjection(wkt)
        print(f"✓ 已为 {os.path.basename(tif_path)} 设置坐标系为 {srs_name}")
        dataset = None  # 关闭数据集
    else:
        raise ValueError(f"无法打开TIF文件: {tif_path}")


def get_shp_bounds(shp_path: str):
    """
    读取 Shapefile 并返回其在 WGS84 (EPSG:4326) 下的边界范围（经纬度）。

    Returns:
        tuple: (lon_min, lat_min, lon_max, lat_max)
    """
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile 不存在: {shp_path}")

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Shapefile 为空，无法获取范围。")

    if gdf.crs != "EPSG:4326":
        if gdf.crs is None:
            raise ValueError("Shapefile 缺少 CRS 信息，无法安全转换到 WGS84。")
        gdf = gdf.to_crs("EPSG:4326")

    bounds = gdf.total_bounds
    return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])


def create_coordinate_transformer(src_srs, target_srs=None):
    """
    创建可靠的坐标转换对象，支持源坐标系为地理/投影类型，确保转换结果准确

    参数:
        src_srs: 源坐标系 (osr.SpatialReference对象，必须有效)
        target_srs: 目标坐标系，默认WGS84(EPSG:4326)，支持EPSG代码/ WKT / osr对象

    返回:
        target_srs_obj: 目标坐标系空间参考对象
        transform_func: 坐标转换函数，输入(x,y[,z])返回转换后坐标
    """
    # --------------------------
    # 1. 源坐标系有效性校验
    # --------------------------
    if not isinstance(src_srs, osr.SpatialReference):
        raise TypeError("src_srs必须是osr.SpatialReference对象")
    try:
        # 尝试获取坐标系权威信息来验证有效性
        if not src_srs.GetAttrValue('AUTHORITY', 0):
            raise ValueError("源坐标系缺少权威信息，可能无效")
    except:
        raise ValueError("源坐标系无效，请检查src_srs")

    # 获取源坐标系关键信息
    src_is_geo = src_srs.IsGeographic()
    src_is_proj = src_srs.IsProjected()
    src_epsg = src_srs.GetAttrValue('AUTHORITY', 1)
    src_datum = src_srs.GetAttrValue("DATUM") or "未知"

    # --------------------------
    # 2. 目标坐标系处理与校验
    # --------------------------
    if target_srs is None:
        # 默认目标：WGS84地理坐标系
        target_srs_obj = osr.SpatialReference()
        target_srs_obj.ImportFromEPSG(4326)
    elif isinstance(target_srs, int):
        target_srs_obj = osr.SpatialReference()
        if target_srs_obj.ImportFromEPSG(target_srs) != 0:
            raise ValueError(f"EPSG代码无效: {target_srs}")
    elif isinstance(target_srs, str):
        target_srs_obj = osr.SpatialReference()
        if target_srs_obj.ImportFromWkt(target_srs) != 0:
            raise ValueError(f"WKT字符串无效: {target_srs[:50]}...")
    elif isinstance(target_srs, osr.SpatialReference):
        target_srs_obj = target_srs
        if not target_srs_obj.IsValid():
            raise ValueError("目标坐标系对象无效")
    else:
        raise TypeError(f"不支持的目标坐标系类型: {type(target_srs)}")

    # 获取目标坐标系关键信息
    target_is_geo = target_srs_obj.IsGeographic()
    target_is_proj = target_srs_obj.IsProjected()
    target_epsg = target_srs_obj.GetAttrValue('AUTHORITY', 1)
    target_datum = target_srs_obj.GetAttrValue("DATUM") or "未知"

    # 打印调试信息（关键）
    print(f"\n[坐标系信息]")
    print(f"源坐标系 - 类型: {'地理' if src_is_geo else '投影'}, EPSG: {src_epsg}, 基准面: {src_datum}")
    print(f"目标坐标系 - 类型: {'地理' if target_is_geo else '投影'}, EPSG: {target_epsg}, 基准面: {target_datum}")

    # --------------------------
    # 3. 坐标系组合合理性校验
    # --------------------------
    # 地理→地理：基准面不一致警告
    if src_is_geo and target_is_geo and src_datum != target_datum:
        print(f"⚠️ 警告：地理坐标系基准面不同（{src_datum} → {target_datum}），转换可能有误差")

    # 投影→投影：建议通过地理坐标系中转（如果基准面不同）
    if src_is_proj and target_is_proj:
        src_geo = src_srs.CloneGeogCS()
        target_geo = target_srs_obj.CloneGeogCS()
        if src_geo.GetAttrValue("DATUM") != target_geo.GetAttrValue("DATUM"):
            print(f"⚠️ 警告：投影基准面不同，将自动通过WGS84中转")
            # 强制使用中间转换
            return _create_composite_transform(src_srs, target_srs_obj)

    # --------------------------
    # 4. 创建并测试转换对象
    # --------------------------
    try:
        # 尝试直接转换
        direct_transform = osr.CoordinateTransformation(src_srs, target_srs_obj)

        # 生成合理的测试点（避免用(0,0)这种可能在无效区域的点）
        test_x, test_y = _get_valid_test_point(src_is_geo, src_epsg)

        # 测试转换
        test_result = direct_transform.TransformPoint(test_x, test_y)
        if not _is_valid_coordinate(test_result[0], test_result[1], target_is_geo):
            raise ValueError("直接转换结果超出合理范围")

        # 封装转换函数（统一接口）
        def transform_func(x, y, z=0):
            res = direct_transform.TransformPoint(x, y, z)
            return (res[0], res[1]) if len(res) >= 2 else (None, None)

        print("✅ 直接转换验证通过")
        return target_srs_obj, transform_func

    except Exception as e:
        print(f"❌ 直接转换失败: {str(e)}, 尝试中间转换...")
        # 尝试通过WGS84中转
        return _create_composite_transform(src_srs, target_srs_obj)


def _create_composite_transform(src_srs, target_srs_obj):
    """创建通过WGS84中转的复合转换"""
    try:
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)

        # 源→WGS84转换
        transform1 = osr.CoordinateTransformation(src_srs, wgs84)
        # WGS84→目标转换
        transform2 = osr.CoordinateTransformation(wgs84, target_srs_obj)

        # 测试中转转换
        test_x, test_y = _get_valid_test_point(src_srs.IsGeographic(), src_srs.GetAttrValue('AUTHORITY', 1))
        step1 = transform1.TransformPoint(test_x, test_y)
        step2 = transform2.TransformPoint(step1[0], step1[1])
        if not _is_valid_coordinate(step2[0], step2[1], target_srs_obj.IsGeographic()):
            raise ValueError("中间转换结果超出合理范围")

        # 封装复合转换函数
        def composite_func(x, y, z=0):
            step1 = transform1.TransformPoint(x, y, z)
            step2 = transform2.TransformPoint(step1[0], step1[1], step1[2])
            return (step2[0], step2[1]) if len(step2) >= 2 else (None, None)

        print("✅ 中间转换验证通过")
        return target_srs_obj, composite_func

    except Exception as e2:
        raise ValueError(f"❌ 所有转换方案失败: {str(e2)}")


def _get_valid_test_point(is_geographic, epsg):
    """生成适合当前坐标系的测试点（避免无效区域）"""
    if is_geographic:
        # 地理坐标系：使用中纬度地区有效经纬度（避免极点、国际日期变更线附近）
        return 105.0, 35.0  # 中国中部附近经纬度
    else:
        # 投影坐标系：使用UTM等投影的典型有效范围（假设米制）
        if epsg and epsg.startswith('326'):  # UTM北半球
            return 500000, 4000000  # UTM典型坐标
        else:
            return 100000, 100000  # 通用投影坐标


def _is_valid_coordinate(x, y, is_target_geographic):
    """验证转换后的坐标是否在合理范围内"""
    if is_target_geographic:
        # 地理坐标：经度[-180,180]，纬度[-90,90]
        return (-180 <= x <= 180) and (-90 <= y <= 90)
    else:
        # 投影坐标：通常在[-1e7, 1e7]米范围内（根据常见投影调整）
        return (-1e7 <= x <= 1e7) and (-1e7 <= y <= 1e7)


def transform_coordinates(transform_func, x, y, z=0):
    """
    执行坐标转换，封装错误处理

    参数:
        transform_func: create_coordinate_transformer返回的转换函数
        x, y: 源坐标
        z: 高程（可选，默认0）

    返回:
        (tx, ty): 转换后的坐标
    """
    if not callable(transform_func):
        raise TypeError("transform_func必须是可调用的转换函数")

    try:
        tx, ty = transform_func(x, y, z)
        if tx is None or ty is None:
            raise ValueError("转换返回空值")
        return tx, ty
    except Exception as e:
        raise RuntimeError(f"坐标转换执行失败 (x={x}, y={y}): {str(e)}")

if __name__ == "__main__":
    # Define your paths and target CRS
    input_file = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"           # Replace with your input TIF file path
    output_file = r"C:\Users\Kevin\Documents\ResearchData\ZhouTun\zhou_tun_gou_WGS84.tif"         # Replace with your desired output TIF file path
    target_epsg_code = 4326                               # Replace with your target EPSG code (e.g., 3857, 2154, etc.)

    try:
        add_vertical_datum_with_backup(input_file, vertical_datum="EGM2008")
        print("\n🎉 操作成功完成！")
    except Exception as e:
        print(f"\n💥 发生错误: {e}")
        sys.exit(1)