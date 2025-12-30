#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: generated_shp_from_tif
# @Time    : 2025/12/30 15:08
# @Author  : Kevin
# @Describe: 生成TIF文件对应的SHP矩形框，矩形的边界与TIF文件的边界相同

import os
from osgeo import gdal, ogr, osr

def create_boundary_shp_from_dem(dem_path, output_shp_path):
    """
    从DEM文件生成边界矩形的SHP文件

    参数:
        dem_path: 输入DEM文件路径（.tif格式）
        output_shp_path: 输出SHP文件路径
    """
    # 打开DEM文件
    dataset = gdal.Open(dem_path)
    if dataset is None:
        raise FileNotFoundError(f"无法打开DEM文件: {dem_path}")

    try:
        # 获取DEM的地理变换参数
        geotransform = dataset.GetGeoTransform()
        if geotransform is None or geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            raise ValueError(f"DEM文件 {dem_path} 没有有效的地理变换信息")

        # 获取DEM的尺寸
        x_size = dataset.RasterXSize
        y_size = dataset.RasterYSize

        # 计算四个角点的坐标
        # 左上角
        x_min = geotransform[0]
        y_max = geotransform[3]
        # 右下角
        x_max = geotransform[0] + geotransform[1] * x_size + geotransform[2] * y_size
        y_min = geotransform[3] + geotransform[4] * x_size + geotransform[5] * y_size

        # 确保正确的坐标顺序
        if geotransform[5] < 0:  # 通常情况下y方向是负的
            y_min = geotransform[3] + geotransform[5] * y_size

        # 创建SHP文件
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(output_shp_path):
            driver.DeleteDataSource(output_shp_path)

        shapefile = driver.CreateDataSource(output_shp_path)
        if shapefile is None:
            raise RuntimeError(f"无法创建SHP文件: {output_shp_path}")

        # 获取DEM的投影信息
        spatial_ref = osr.SpatialReference()
        proj = dataset.GetProjection()
        if proj:
            spatial_ref.ImportFromWkt(proj)
        else:
            # 如果没有投影信息，使用默认的WGS84
            spatial_ref.ImportFromEPSG(4326)

        # 创建图层
        layer = shapefile.CreateLayer('boundary', srs=spatial_ref, geom_type=ogr.wkbPolygon)

        # 创建多边形要素
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_min, y_max)  # 左上
        ring.AddPoint(x_max, y_max)  # 右上
        ring.AddPoint(x_max, y_min)  # 右下
        ring.AddPoint(x_min, y_min)  # 左下
        ring.AddPoint(x_min, y_max)  # 闭合环

        # 创建多边形
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)

        # 创建要素
        feature_def = layer.GetLayerDefn()
        feature = ogr.Feature(feature_def)
        feature.SetGeometry(polygon)

        # 添加要素到图层
        layer.CreateFeature(feature)

        print(f"成功生成边界SHP文件: {output_shp_path}")
        print(f"边界坐标: X({x_min}, {x_max}), Y({y_min}, {y_max})")

    finally:
        # 关闭数据集
        dataset = None
        shapefile = None

def batch_create_boundary_shp_from_dir(input_dir, output_dir=None, file_extension='.tif'):
    """
    从目录中批量读取TIF文件，并生成对应的边界SHP文件

    参数:
        input_dir: 输入目录路径，包含TIF文件
        output_dir: 输出目录路径，如果为None则保存到TIF文件所在目录
        file_extension: 要处理的文件扩展名，默认为'.tif'
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 获取目录中所有TIF文件
    tif_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(file_extension.lower()):
                tif_files.append(os.path.join(root, file))

    if not tif_files:
        print(f"在目录 {input_dir} 中未找到 {file_extension} 文件")
        return

    print(f"找到 {len(tif_files)} 个TIF文件，开始处理...")

    processed_count = 0
    failed_files = []

    for tif_path in tif_files:
        try:
            # 确定输出SHP文件路径
            if output_dir is None:
                # 保存到TIF文件所在目录
                shp_filename = os.path.splitext(os.path.basename(tif_path))[0] + '_boundary.shp'
                shp_path = os.path.join(os.path.dirname(tif_path), shp_filename)
            else:
                # 保存到指定输出目录
                os.makedirs(output_dir, exist_ok=True)
                shp_filename = os.path.splitext(os.path.basename(tif_path))[0] + '_boundary.shp'
                shp_path = os.path.join(output_dir, shp_filename)

            # 调用单个文件处理函数
            create_boundary_shp_from_dem(tif_path, shp_path)
            processed_count += 1

        except Exception as e:
            print(f"处理文件失败 {tif_path}: {str(e)}")
            failed_files.append(tif_path)

    print(f"\n处理完成！成功处理 {processed_count}/{len(tif_files)} 个文件")
    if failed_files:
        print(f"失败的文件数量: {len(failed_files)}")
        for failed_file in failed_files:
            print(f"  - {failed_file}")

def main():
    input_dir = None
    output_dir = None
    batch_create_boundary_shp_from_dir(input_dir, output_dir)
if __name__ == "__main__":
    main()
