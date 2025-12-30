#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: modify_dem_coordinate_fixed
# @Time    : 2025/8/15 00:17
# @Author  : Kevin
# @Describe: 修复了x和y坐标系颠倒问题的DEM修改代码
import heapq
import json

import geopandas as gpd
import numpy as np
import os.path as osp

import pandas as pd
import rasterio
import cv2
from affine import Affine
from rasterio.features import shapes  # 提取掩膜轮廓
from shapely.geometry import shape, MultiPolygon

from DataProcessing.get_information import get_pixel_size_accurate, geo_to_pixel, pixel_to_geo
from DataProcessing.rectangle_utils import calculate_rectangle_midpoints_and_extend_long_sides, calculate_rectangle_with_extend
from Logger import LoggerManager


def find_max_distance_combination(group1, group2):
    """从两组点中各选一个点，找出能达到最大距离的组合"""
    max_distance = 0
    farthest_point1 = None
    farthest_point2 = None

    for point1 in group1:
        for point2 in group2:
            # 注意：这里保持原始计算，因为是图像内部的相对距离
            distance = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
            if distance > max_distance:
                max_distance = distance
                farthest_point1 = point1
                farthest_point2 = point2

    return farthest_point1, farthest_point2

def calculate_dem_crest_height(slope_len_0, slope_len_1, height_0, height_1, gradient_0=1.75, gradient_1=1.5):
    if abs(height_0 + slope_len_0 / gradient_0 - height_1 - slope_len_1 / gradient_1) > abs(
            height_0 + slope_len_0 / gradient_1 - height_1 - slope_len_1 / gradient_0):
        return True, (height_0 + slope_len_0 / gradient_0 + height_1 + slope_len_1 / gradient_1) / 2
    else:
        return False, (height_0 + slope_len_0 / gradient_1 + height_1 + slope_len_1 / gradient_0) / 2

def get_small_mask_info(small_google_file, small_mask_file, offset_x, offset_y, logger, extend=100):

    pixel_size_m_x, pixel_size_m_y = get_pixel_size_accurate(small_google_file)

    # 读取掩膜文件
    with open(small_mask_file, "r", encoding="utf-8") as f:
        # json.load()：将JSON字符串解析为Python字典/列表
        result_dict = json.load(f)

        if result_dict == {}:
            return None

    # # 转换成cv2对应的(col, row)格式
    # for key, values in result_dict.items():
    #     for i, value in enumerate(values):
    #         x, y = value['x_center'], value['y_center']
    #         result_dict[key][i]['x_center'] = y
    #         result_dict[key][i]['y_center'] = x

    logger.info(f"现在处理的是文件{small_mask_file}")

    # 标准情况：一个坝顶+两个坝坡
    if len(result_dict.get("slope", [])) == 2 and len(result_dict.get("road", [])) == 1:

        logger.info(f"这个文件是 一个坝顶+两个坝坡 的情况")

        crest_side_width, crest_side_height = result_dict['road'][0]['width'], result_dict['road'][0]['height']

        # 这个代表淤地坝的方向
        sides_select = crest_side_width > crest_side_height

        # 处理坝坡信息
        temp_dict = {0: 'width', 1: 'height'}
        dem_slope_len_0 = result_dict['slope'][0][temp_dict[sides_select]]
        dem_slope_len_1 = result_dict['slope'][1][temp_dict[sides_select]]

        # 转换为实际长度（米）
        if abs(pixel_size_m_x - pixel_size_m_y) < 1:
            dem_slope_len_0 = dem_slope_len_0 * pixel_size_m_x
            dem_slope_len_1 = dem_slope_len_1 * pixel_size_m_y
        else:
            raise ValueError(f"像素尺寸不一致，{pixel_size_m_x}*{pixel_size_m_y}，相关的处理逻辑需要添加")

        # 计算中点
        if sides_select:

            dem_0_mid_0, dem_0_mid_1, _, _ = calculate_rectangle_midpoints_and_extend_long_sides(
                (result_dict['slope'][0]['x_center'], result_dict['slope'][0]['y_center']),
                (result_dict['slope'][0]['width'], result_dict['slope'][0]['height']),
                result_dict['slope'][0]['angle_rad'])
            dem_1_mid_0, dem_1_mid_1, _, _ = calculate_rectangle_midpoints_and_extend_long_sides(
                (result_dict['slope'][1]['x_center'], result_dict['slope'][1]['y_center']),
                (result_dict['slope'][1]['width'], result_dict['slope'][1]['height']),
                result_dict['slope'][1]['angle_rad'])
        else:
            _, _, dem_0_mid_0, dem_0_mid_1 = calculate_rectangle_midpoints_and_extend_long_sides(
                (result_dict['slope'][0]['x_center'], result_dict['slope'][0]['y_center']),
                (result_dict['slope'][0]['width'], result_dict['slope'][0]['height']),
                result_dict['slope'][0]['angle_rad'])
            _, _, dem_1_mid_0, dem_1_mid_1 = calculate_rectangle_midpoints_and_extend_long_sides(
                (result_dict['slope'][1]['x_center'], result_dict['slope'][1]['y_center']),
                (result_dict['slope'][1]['width'], result_dict['slope'][1]['height']),
                result_dict['slope'][1]['angle_rad'])

        # 结构是 (col, row)
        dem_center_0, dem_center_1 = find_max_distance_combination((dem_0_mid_0, dem_0_mid_1),
                                                                   (dem_1_mid_0, dem_1_mid_1))

        # 结构是 (row, col)
        dem_center_0 = (int(dem_center_0[1] + offset_y), int(dem_center_0[0] + offset_x))
        dem_center_1 = (int(dem_center_1[1] + offset_y), int(dem_center_1[0] + offset_x))

        # 获取坝顶的旋转矩形信息
        extend = float(extend) / pixel_size_m_y  # 获取实际的像素长度应该是多少

        _, _, _, _, point1, point2, point3, point4 = calculate_rectangle_midpoints_and_extend_long_sides(
            (result_dict['road'][0]['x_center'], result_dict['road'][0]['y_center']),
            (result_dict['road'][0]['width'], result_dict['road'][0]['height']),
            result_dict['road'][0]['angle_rad'], extend=extend)

        # 结构是 (row, col)
        extend_points = (
            (point1[1] + offset_y, point1[0] + offset_x),
            (point2[1] + offset_y, point2[0] + offset_x),
            (point3[1] + offset_y, point3[0] + offset_x),
            (point4[1] + offset_y, point4[0] + offset_x)
        )

        slope_vertice_0 = calculate_rectangle_with_extend(
            (result_dict['slope'][0]['x_center'], result_dict['slope'][0]['y_center']),
            (result_dict['slope'][0]['width'],result_dict['slope'][0]['height']),
            result_dict['slope'][0]['angle_rad'], extend=50 / pixel_size_m_y, direction=temp_dict[1-sides_select])

        slope_vertice_1 = calculate_rectangle_with_extend(
            (result_dict['slope'][1]['x_center'], result_dict['slope'][1]['y_center']),
            (result_dict['slope'][1]['width'], result_dict['slope'][1]['height']),
            result_dict['slope'][1]['angle_rad'], extend=50 / pixel_size_m_y, direction=temp_dict[1-sides_select])

        # 结构是 (row, col)
        adjusted_slope_vertice_0 = [[vertex[1] + offset_y, vertex[0] + offset_x] for vertex in slope_vertice_0]
        adjusted_slope_vertice_1 = [[vertex[1] + offset_y, vertex[0] + offset_x] for vertex in slope_vertice_1]

        return dem_center_0, dem_center_1, dem_slope_len_0, dem_slope_len_1, adjusted_slope_vertice_0, adjusted_slope_vertice_1, extend_points

    else:
        logger.info(f"当前情况不足以计算，舍弃")

    return None


def order_points(pts):
    """
    对四个点进行顺序化（适用于矩形情况），返回[左上, 右上, 右下, 左下]的顺序
    """
    # 展平点数组
    pts = pts.reshape((-1, 2))
    ordered_pts = np.zeros((4, 2), dtype="float32")

    # 计算每个点的x+y和x-y
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)  # 实际上是y - x

    # 左上角点x+y最小，右下角点x+y最大
    ordered_pts[0] = pts[np.argmin(s)]  # 左上
    ordered_pts[2] = pts[np.argmax(s)]  # 右下

    # 剩下的两个点中，x-y较小的是右上，较大的是左下
    # 先找到不是左上和右下的点
    remaining = np.delete(pts, [np.argmin(s), np.argmax(s)], axis=0)

    # 计算这两个点相对于左上角的向量
    vec1 = remaining[0] - ordered_pts[0]
    vec2 = remaining[1] - ordered_pts[0]

    # 向量叉积：正为逆时针，负为顺时针
    cross = np.cross(vec1, vec2)

    if cross > 0:
        ordered_pts[1] = remaining[1]  # 右上
        ordered_pts[3] = remaining[0]  # 左下
    else:
        ordered_pts[1] = remaining[0]  # 右上
        ordered_pts[3] = remaining[1]  # 左下

    return ordered_pts.reshape((-1, 1, 2))

def get_mask(mask, points, src, large_src, input_is_row_col=False):

    if input_is_row_col:
        points = [[col, row] for (row, col) in points]

    # print("before")
    # print(points)

    adjusted_points = []
    for (col, row) in points:
        adjusted_points.append(geo_to_pixel(large_src, *pixel_to_geo(src, row, col), is_cv=True))
    # 将点转换为OpenCV需要的格式
    # print("after")
    # print(adjusted_points)
    points = np.array(adjusted_points, dtype=np.int32)
    points = order_points(points)

    # 绘制填充的旋转多边形（保持旋转角度）
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)

    return mask

def modify_large_dem(small_google_files, small_mask_files, large_dem_file, flow_accumulation_file, repeat_file, logger, output_dem_file, output_shp_file, extend=100, offset_x=0, offset_y=0, gradient=0.0, is_move_dem=False) -> [list, list]:
    """
    修改大范围DEM中对应小DEM的坝顶高程

    该函数通过以下步骤实现DEM修改：
    1. 读取小DEM和掩膜文件，获取需要修改的高程值和区域
    2. 将小DEM中的修改区域坐标转换为经纬度(WGS84)
    3. 将经纬度转换为大DEM中的像素坐标
    4. 在大DEM中修改对应区域的高程值并保存

    Args:
        small_google_files (list): 小范围DEM文件路径，用于获取对应mask位置的经纬度
        large_dem_file (str): 大范围DEM文件路径，将在此DEM中应用修改
        small_mask_files (list): 掩膜文件路径，为json格式，用于标识小DEM中需要修改的区域，原地修改
        flow_accumulation_file (str): 标记流向的方向
        repeat_file (str): 用来判断是否有重复采样的文件
        offset_x (int, optional): X方向偏移量，用于调整坐标转换，默认为0
        offset_y (int, optional): Y方向偏移量，用于调整坐标转换，默认为0

    Returns:
        str: 修改后的大DEM文件路径，如果修改失败则返回原始大DEM文件路径
    """

    logger.info(f"""现在根据识别到的坝坡和坝高进行DEM的修改：
    1.{small_mask_files}：{small_google_files} 上识别得到的坝坡和坝高的信息
    2.根据识别结果，对 {large_dem_file} 进行淤地坝区域的高程进行修正
    3.其中对应的偏移量是 {offset_x}，{offset_y}""")

    start_point_masks = []
    crest_point_masks = []
    check_dam_heights = []

    infos = []

    logger.info(f"共有{len(small_google_files)}个坝待识别计算，经检验获取有用信息的数目：{len(infos)}")

    for small_google_file, small_mask_file in zip(small_google_files, small_mask_files):

        temp = get_small_mask_info(small_google_file, small_mask_file, offset_x, offset_y, extend=extend, logger=logger)

        if temp is not None:
            infos.append(temp)

        logger.info(f"经检验获取有用信息的数目：{len(infos)}")

    with rasterio.open(large_dem_file) as large_src, rasterio.open(flow_accumulation_file) as flow_src:

        if large_src.crs != flow_src.crs:
            raise ValueError(f"{large_dem_file}和{flow_accumulation_file}的坐标系不一致，请检查")
        elif large_src.shape != flow_src.shape:
            raise ValueError(f"{large_dem_file}和{flow_accumulation_file}的分辨率不一致，请检查")

        large_dem_array = large_src.read(1)
        large_dem_array_copy = large_dem_array.copy()
        large_meta = large_src.meta.copy()
        large_transform = large_src.transform

        flow_accumulation_array = flow_src.read(1)

        # 用来判断有没有重复识别
        if not osp.exists(repeat_file):
            repeat_array = np.full(large_dem_array.shape, False, dtype=bool)
        else:
            repeat_array = np.load(repeat_file)

        repeat_array_copy = repeat_array.copy()

        for info, small_google_file in zip(infos, small_google_files):

            slope_0, slope_1, dem_slope_len_0, dem_slope_len_1, dem_vertice_0, dem_vertice_1, extend_points = info

            src = rasterio.open(small_google_file)
            logger.info(f"处理{small_google_file}")

            dem_vertice_0 = np.array(order_points(np.array(dem_vertice_0))).reshape(-1, 2).tolist()
            dem_vertice_1 = np.array(order_points(np.array(dem_vertice_1))).reshape(-1, 2).tolist()

            # 坝坡掩膜
            mask_slope0 = np.zeros_like(flow_accumulation_array, dtype=np.uint8)
            mask_slope1 = np.zeros_like(flow_accumulation_array, dtype=np.uint8)
            logger.info(f"针对当前区域，开始处理坝坡0")
            mask_slope0 = get_mask(mask_slope0, dem_vertice_0, src, large_src, input_is_row_col=True)
            logger.info(f"针对当前区域，开始处理坝坡1")
            mask_slope1 = get_mask(mask_slope1, dem_vertice_1, src, large_src, input_is_row_col=True)

            # 坝顶掩膜
            logger.info(f"针对当前区域，开始处理坝顶")

            mask = np.zeros_like(large_dem_array, dtype=np.uint8)
            mask = get_mask(mask, extend_points, src, large_src, input_is_row_col=True)

            temp_numbers = np.sum(mask == 1) + np.sum(mask_slope1 == 1) + np.sum(mask_slope0 == 1) + 1
            temp_already = np.sum(repeat_array[mask == 1]) + np.sum(repeat_array[mask_slope1 == 1]) + np.sum(repeat_array[mask_slope0 == 1])

            repeat_array = repeat_array | mask | mask_slope0 | mask_slope1

            logger.info(f"针对当前区域的{temp_numbers}个像素点，已经修改了{temp_already}")
            if temp_already / temp_numbers > 0.5:
                LoggerManager.get_logger().warning(f"{small_google_file}的坝坡和坝高已经修改过了，请勿重复修改")
                continue

            slope0_accumulation = np.max(flow_accumulation_array[mask_slope0 == 1])
            slope1_accumulation = np.max(flow_accumulation_array[mask_slope1 == 1])
            crest_slope_0 = np.min(large_dem_array[mask_slope0 == 1])
            crest_slope_1 = np.min(large_dem_array[mask_slope1 == 1])

            temp_bool = slope0_accumulation < slope1_accumulation

            if temp_bool:
                # slope0是上游
                # 假定测试的越长越精准，选择长的坝
                if dem_slope_len_0 > dem_slope_len_1:
                    crest_height = dem_slope_len_0 / 1.75 + crest_slope_0
                else:
                    crest_height = dem_slope_len_1 / 1.5 + crest_slope_1
                mask_slope = mask_slope0
            else:
                # slope1是上游
                if dem_slope_len_0 > dem_slope_len_1:
                    crest_height = dem_slope_len_0 / 1.5 + crest_slope_0
                else:
                    crest_height = dem_slope_len_1 / 1.75 + crest_slope_1
                mask_slope = mask_slope1


            logger.info(f"""{small_google_file}在大范围DEM中的信息如下
            {geo_to_pixel(large_src, *pixel_to_geo(src, slope_0[0], slope_0[1]))}对应的坡长是{dem_slope_len_0}，坡度斜率是{1.75 if temp_bool else 1.5}，对应高程是{crest_slope_0}， 坐标是{dem_vertice_0}, 
            {geo_to_pixel(large_src, *pixel_to_geo(src, slope_1[0], slope_1[1]))}对应的坡长是{dem_slope_len_1}，坡度斜率是{1.5 if temp_bool else 1.75}，对应高程是{crest_slope_1}， 坐标是{dem_vertice_1}""")

            crest_point_masks.append(mask)

            # 修改高程值
            large_dem_array[(mask == 1) & (large_dem_array < crest_height)] = crest_height
            # large_dem_array[mask_slope == 1] = crest_height

            large_src.close()

            if is_move_dem:
                new_transform = Affine(
                    large_transform.a,  # x方向缩放
                    large_transform.b,  # x方向旋转
                    large_transform.c - offset_x,  # x方向偏移
                    large_transform.d,  # y方向旋转
                    large_transform.e,  # y方向缩放
                    large_transform.f - offset_y  # y方向偏移
                )

                large_meta['transform'] = new_transform

            with rasterio.open(large_dem_file, 'w', **large_meta) as dst:
                dst.write(large_dem_array, 1)

            src.close()

            logger.info(f"""现在是对应一个坝顶两个坝坡的组合：修改后的高程值应该是：{crest_height}，对应的点位是：{extend_points}""")

            result_judge = calculate_total_volume(input_dem_file=large_dem_file,
                                                   start_point_masks=[mask_slope],
                                                   crest_point_masks=[mask],
                                                   check_dam_heights=[crest_height],
                                                   output_dem_path=output_dem_file,
                                                   output_shp_file=output_shp_file,
                                                   gradient=gradient,
                                                   logger=logger
                                                   )

            if not result_judge[0]:

                logger.info(
                    f"""坝坡的方向似乎有误判，现在换了方向重新尝试""")

                if not temp_bool:
                    # slope0是上游
                    # 假定测试的越长越精准，选择长的坝
                    if dem_slope_len_0 > dem_slope_len_1:
                        crest_height = dem_slope_len_0 / 1.75 + crest_slope_0
                    else:
                        crest_height = dem_slope_len_1 / 1.5 + crest_slope_1
                    mask_slope = mask_slope0
                else:
                    # slope1是上游
                    if dem_slope_len_0 > dem_slope_len_1:
                        crest_height = dem_slope_len_0 / 1.5 + crest_slope_0
                    else:
                        crest_height = dem_slope_len_1 / 1.75 + crest_slope_1
                    mask_slope = mask_slope1

                logger.info(f"""{small_google_file}在大范围DEM中的信息如下
                {geo_to_pixel(large_src, *pixel_to_geo(src, slope_0[0], slope_0[1]))}对应的坡长是{dem_slope_len_0}，坡度斜率是{1.75 if temp_bool else 1.5}，对应高程是{crest_slope_0}， 坐标是{dem_vertice_0}, 
                {geo_to_pixel(large_src, *pixel_to_geo(src, slope_1[0], slope_1[1]))}对应的坡长是{dem_slope_len_1}，坡度斜率是{1.5 if temp_bool else 1.75}，对应高程是{crest_slope_1}， 坐标是{dem_vertice_1}""")

                crest_point_masks.append(mask)

                large_dem_array = large_dem_array_copy.copy()

                src.close()

                # 修改高程值
                large_dem_array[(mask == 1) & (large_dem_array < crest_height)] = crest_height
                # large_dem_array[mask_slope==1] = crest_height

                if is_move_dem:
                    new_transform = Affine(
                        large_transform.a,  # x方向缩放
                        large_transform.b,  # x方向旋转
                        large_transform.c - offset_x,  # x方向偏移
                        large_transform.d,  # y方向旋转
                        large_transform.e,  # y方向缩放
                        large_transform.f - offset_y  # y方向偏移
                    )

                    large_meta['transform'] = new_transform

                if large_src and not large_src.closed:
                    large_src.close()

                with rasterio.open(large_dem_file, 'w', **large_meta) as dst:
                    dst.write(large_dem_array, 1)

                logger.info(f"""现在是对应一个坝顶两个坝坡的组合：修改后的高程值应该是：{crest_height}，对应的点位是：{extend_points}""")

                result_judge = calculate_total_volume(input_dem_file=large_dem_file,
                                                      start_point_masks=[mask_slope],
                                                      crest_point_masks=[mask],
                                                      check_dam_heights=[crest_height],
                                                      output_dem_path=output_dem_file,
                                                      output_shp_file=output_shp_file,
                                                      gradient=gradient,
                                                      logger=logger,
                                                      offset_x = offset_x,
                                                      offset_y = offset_y
                                                      )

                if not result_judge[0]:

                    large_dem_array = large_dem_array_copy.copy()

                    if is_move_dem:
                        new_transform = Affine(
                            large_transform.a,  # x方向缩放
                            large_transform.b,  # x方向旋转
                            large_transform.c - offset_x,  # x方向偏移
                            large_transform.d,  # y方向旋转
                            large_transform.e,  # y方向缩放
                            large_transform.f - offset_y  # y方向偏移
                        )

                        large_meta['transform'] = new_transform

                    if large_src and not large_src.closed:
                        large_src.close()

                    with rasterio.open(large_dem_file, 'w', **large_meta) as dst:
                        dst.write(large_dem_array, 1)

                    repeat_array = repeat_array_copy.copy()

                    logger.info(f"""坝坡的两个方向都有错误，不做修改""")
                    continue

            start_point_masks.append(mask_slope)
            check_dam_heights.append(crest_height)

    np.save(repeat_file, repeat_array)

    if is_move_dem:
        with rasterio.open(flow_accumulation_file) as src:
            data = src.read(1)
            meta = src.meta.copy()
            transform = src.transform

            new_transform = Affine(
                transform.a,  # x方向缩放
                transform.b,  # x方向旋转
                transform.c - offset_x,  # x方向偏移
                transform.d,  # y方向旋转
                transform.e,  # y方向缩放
                transform.f - offset_y  # y方向偏移
            )

            meta['transform'] = new_transform

        with rasterio.open(flow_accumulation_file, 'w', **meta) as dst:
            dst.write(data, 1)

    return start_point_masks, crest_point_masks, check_dam_heights

def check_edge(n, m, x, y):
    if 0 <= x < n and 0 <= y < m:
        return True
    return False

def calculate_total_volume(input_dem_file, start_point_masks, crest_point_masks, check_dam_heights, output_dem_path, output_shp_file, logger, gradient=0.0000, offset_x=0, offset_y=0):
    """
    核心迭代计算逻辑：计算单个淤地的库容

    该函数通过逐步抬高淤地区域的高程来模拟淤积过程，并计算总体积。
    使用BFS算法从最低点开始逐步填充，直到达到淤地坝的高度限制。

    Args:
        input_dem_file (str): DEM数据数组，包含原始高程信息
        start_point_masks (list): 包含每一个坝坡（起始扩展点的）信息
        crest_point_masks (list)：
        check_dam_heights (list): 坝坡对应淤地坝的高度,
        output_shp_file (str):
        output_dem_path (str): 输出

    Returns:
        tuple: 包含两个元素的元组：
            - total_volume (float): 计算得到的库容体积（立方米）
            - current_dem (numpy.ndarray): 计算完成后的DEM数组（淤积后的高程数据）

    """
    pixel_size_x, pixel_size_y = get_pixel_size_accurate(input_dem_file)
    pixel_area = abs(pixel_size_x * pixel_size_y)  # 确保面积为正

    with rasterio.open(input_dem_file) as dem_src:
        transform = dem_src.transform
        dem_array = dem_src.read(1)
        dem_meta = dem_src.meta

        dem_src.close()

    dem_array = dem_array.astype(np.float64)

    rows, cols = dem_array.shape

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8方向

    sorted_pairs = sorted(zip(start_point_masks, crest_point_masks, check_dam_heights), key=lambda x: x[2], reverse=True)
    total_volumes = []  # 记录每个坝的体积
    all_gdfs = []  # 收集所有区域的GeoDataFrame用于合并

    logger.info(f"开始计算库容")

    result_judge = []

    for idx, (start_point_mask, crest_point_mask, check_dam_height) in enumerate(sorted_pairs):

        dem_array_copy = dem_array.copy()

        current_judge = True

        logger.info(f"对第{idx}个组别的淤地坝和坝坡进行处理")

        total_volume = 0

        heap = []
        visited = set()
        processed = set()  # 用于跟踪已经处理过的节点
        shp_mask = set() # 用于进行shp的存储

        points = np.argwhere(crest_point_mask == 1)

        for point in points:
            r, c = point
            if dem_array[r][c] <= check_dam_height:
                temp_r = r + offset_y
                temp_c = c + offset_x
                if check_edge(rows, cols, temp_r, temp_c):
                    shp_mask.add((r, c))
                if dem_array[r][c] < check_dam_height:
                    dem_array[r][c] = check_dam_height
            processed.add((r, c))

        points = np.argwhere(start_point_mask == 1)

        bench_mark = 100000

        for point in points:

            r, c = point

            if dem_array[r][c] >= check_dam_height:
                continue

            bench_mark = min(bench_mark, dem_array[r][c])

            temp_r = r + offset_y
            temp_c = c + offset_x
            if check_edge(rows, cols, temp_r, temp_c):
                shp_mask.add((r, c))

            total_volume += (check_dam_height-dem_array[r][c])
            dem_array[r][c] = check_dam_height

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if nr<0 or nr>=rows or nc<0 or nc>=cols or ((nr, nc) in visited) or (start_point_mask[nr, nc]==1) or dem_array[nr][nc] >= check_dam_height:
                    continue
                visited.add((nr, nc))
                heapq.heappush(heap, (check_dam_height, nr, nc))
                total_volume += (check_dam_height-dem_array[nr][nc])
                dem_array[nr][nc] = check_dam_height

        gradient *= pixel_size_x

        logger.info(f"""开始进行淤积填充，根据{bench_mark}进行基准判断""")

        counter = 0
        counter_map = 0

        while heap:

            current_height, r, c = heapq.heappop(heap)

            if current_height > dem_array[r][c]:
                continue

            if (r, c) in processed:
                continue

            counter += 1
            counter_map += (dem_array[r][c]<bench_mark)

            temp_r = r + offset_y
            temp_c = c + offset_x
            if check_edge(rows, cols, temp_r, temp_c):
                shp_mask.add((r, c))

            processed.add((r, c))

            r = int(r)
            c = int(c)

            # 扩展到邻域
            for dr, dc in directions:

                nr, nc = r + dr, c + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue

                if (nr, nc) not in processed:

                    temp_height = current_height + gradient * np.sqrt(dr * dr + dc * dc)  # 修正为相对坐标差值

                    if (nr, nc) not in visited and dem_array[nr][nc] >= temp_height:
                        continue

                    if (nr, nc) in visited:
                        if temp_height < dem_array[nr][nc]:
                            counter += 1
                            counter_map += (dem_array[nr][nc]<bench_mark)

                            total_volume += (temp_height - dem_array[nr][nc])
                            heapq.heappush(heap, (temp_height, nr, nc))
                            dem_array[nr][nc] = temp_height
                    else:
                        if temp_height > dem_array[nr][nc]:
                            counter += 1
                            counter_map += (dem_array[nr][nc] < bench_mark)

                            total_volume += (temp_height - dem_array[nr][nc])
                            heapq.heappush(heap, (temp_height, nr, nc))
                            dem_array[nr][nc] = temp_height

                    if counter % 1000 == 0:
                        if counter_map > 900:
                            current_judge = False
                            break
                        else:
                            counter_map = 0

                if not current_judge:
                    break

        if not current_judge:
            logger.info(f"第{idx + 1}个淤地坝处理方向似乎不对")
            dem_array = dem_array_copy
            result_judge.append(False)
            continue
        else:
            result_judge.append(True)

        actual_volume = total_volume * pixel_area
        total_volumes.append(actual_volume)
        logger.info(f"第{idx + 1}个淤地坝体积：{actual_volume}立方米")

        if shp_mask:  # 确保有处理过的像素

            start_idx = 0  # 默认起始序号
            if osp.exists(output_shp_file):
                # 若文件存在，读取现有最大dam_index并+1作为新序号起点
                existing_gdf = gpd.read_file(output_shp_file)
                max_existing_idx = existing_gdf['dam_index'].max()
                start_idx = max_existing_idx + 1
                logger.info(f"文件{output_shp_file}已存在，新序号从 {start_idx} 开始")
            else:
                logger.info(f"文件{output_shp_file}不存在，将创建新文件")

            # 步骤1：创建与DEM同尺寸的二值掩膜（1=处理区域，0=非处理区域）
            mask_array = np.zeros_like(dem_array, dtype=np.uint8)  # 初始化全0掩膜
            for (r, c) in shp_mask:
                mask_array[r, c] = 1  # 处理过的像素设为1

            # 步骤2：从掩膜提取矢量轮廓（像素坐标→地理坐标）
            # transform：DEM的地理变换，用于将像素坐标映射到实际空间坐标
            # mask=mask_array：只提取值为1的区域轮廓
            contour_shapes = shapes(
                source=mask_array,
                mask=mask_array,
                transform=transform  # 关键：确保轮廓坐标是地理坐标
            )

            # 步骤3：处理轮廓，合并为MultiPolygon（支持多连通区域/孔洞）
            polygons = []
            for geom_dict, value in contour_shapes:
                # 将rasterio的几何字典转为shapely的Polygon
                polygon = shape(geom_dict)
                polygons.append(polygon)

            # 合并所有多边形为一个MultiPolygon（单个区域也建议用MultiPolygon统一格式）
            if polygons:
                multi_polygon = MultiPolygon(polygons)

                # 步骤4：创建GeoDataFrame，绑定体积等属性
                gdf = gpd.GeoDataFrame({
                    'dam_index': [start_idx + idx],  # 淤地坝序号（区分不同坝）
                    'dam_height': [check_dam_height],  # 淤地坝高度
                    'volume': [actual_volume],  # 该区域的淤积体积（立方米）
                    'geometry': [multi_polygon]  # 轮廓几何（MultiPolygon）
                }, crs=dem_meta['crs'])  # 继承DEM的坐标系（确保坐标正确）

                all_gdfs.append(gdf)  # 加入总列表，后续合并

    # 合并所有区域为一个SHP文件（可选）
    if all_gdfs:
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=dem_meta['crs'])
        if osp.exists(output_shp_file):
            # 若文件存在，读取原有数据并合并新数据
            existing_gdf = gpd.read_file(output_shp_file)
            combined_gdf = pd.concat([existing_gdf, combined_gdf], ignore_index=True)
        combined_gdf.to_file(output_shp_file)

    with rasterio.open(output_dem_path, 'w', **dem_meta) as dst:
        dst.write(dem_array, 1)

    logger.info(f"已经完成地形修改")

    return result_judge

def modify_dem(small_google_files, small_mask_files, large_dem_file, flow_accumulation_file, output_dem_file, output_shp_file, repeat_file, logger, extend=100, offset_x=0, offset_y=0, gradient=0.0029, is_move_dem=False):

    start_point_masks, crest_point_masks, check_dam_heights = modify_large_dem(
        small_google_files=small_google_files,
        large_dem_file=large_dem_file,
        small_mask_files=small_mask_files,
        flow_accumulation_file=flow_accumulation_file,
        repeat_file=repeat_file,
        offset_x=offset_x,
        offset_y=offset_y,
        extend=extend,
        is_move_dem=is_move_dem,
        logger=logger,
        gradient=gradient,
        output_shp_file=output_shp_file,
        output_dem_file=output_dem_file
    )


import logging

if __name__ == '__main__':

    input_dir = r"C:\Users\Kevin\Desktop\results\375840_1103697"

    # 小DEM文件路径
    small_google_files = [r"375840_1103697_child_Google_CheckDam_remote_0.tif"]
    # 大范围DEM文件路径
    large_dem_file = r"375840_1103697_Dem_Filled - 副本.tif"
    # 掩膜文件路径
    small_mask_files = [r"375840_1103697_child_Google_CheckDam_remote_0.json"]
    # flow accumulation文件路径
    flow_accumulation_file = "375840_1103697_Spatial_Flow.tif"
    # repeat fill
    repeat_file = "375840_1103697_array.tif"

    # 配置基本logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 获取logger实例
    temp_logger = logging.getLogger('temp')

    # 修改大范围DEM
    start_point_masks, crest_point_masks, check_dam_heights = modify_large_dem(
        small_google_files=[osp.join(input_dir, small_google_file) for small_google_file in small_google_files],
        large_dem_file=osp.join(input_dir, large_dem_file),
        small_mask_files=[osp.join(input_dir, small_mask_file) for small_mask_file in small_mask_files],
        flow_accumulation_file=osp.join(input_dir, flow_accumulation_file),
        repeat_file=osp.join(input_dir, repeat_file),
        offset_x=-64,
        offset_y=32,
        extend=100,
        is_move_dem=False,
        logger=temp_logger,
        gradient=0.0029,
        output_dem_file=osp.join(input_dir, "final.tif"),
        output_shp_file=osp.join(input_dir, "final.shp"),
    )