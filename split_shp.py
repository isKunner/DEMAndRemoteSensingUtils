#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: split_shp
# @Time    : 2026/1/24 15:32
# @Author  : Kevin
# @Describe: 裁剪Shp文件，因为平台上传有大小限制

import os
import geopandas as gpd


def split_shp_by_us_regions(input_shp_path, states_shp_path, output_dir):
    """
    将SHP文件根据美国州区域分组进行拆分

    Args:
        input_shp_path (str): 输入SHP文件路径（例如水库数据）
        states_shp_path (str): 美国州边界SHP文件路径，包含REGION和DIVISION字段
        output_dir (str): 输出目录

    Returns:
        list: 生成的所有输出文件路径
    """
    # 读取输入SHP文件
    print(f"读取输入SHP文件: {os.path.basename(input_shp_path)}")
    input_gdf = gpd.read_file(input_shp_path)

    # 读取美国州边界文件
    print(f"读取美国州边界文件: {os.path.basename(states_shp_path)}")
    states_gdf = gpd.read_file(states_shp_path)

    # 检查是否包含必要的字段
    required_fields = ['REGION', 'DIVISION', 'NAME']
    if not all(field in states_gdf.columns for field in required_fields):
        raise ValueError(f"州边界文件必须包含以下字段: {required_fields}")

    print(f"输入数据包含 {len(input_gdf)} 个要素")
    print(f"州边界数据包含 {len(states_gdf)} 个州")

    # 检查坐标系，如果不一致则转换
    if not input_gdf.crs.equals(states_gdf.crs):
        print(f"坐标系不一致，将州边界数据转换为输入数据的坐标系: {input_gdf.crs}")
        states_gdf = states_gdf.to_crs(input_gdf.crs)

    # 为每个输入要素分配所属州
    print("执行空间连接，确定每个要素所属的州...")
    joined_gdf = gpd.sjoin(input_gdf, states_gdf[['REGION', 'DIVISION', 'NAME', 'geometry']],
                           how='inner', predicate='intersects')
    joined_gdf = joined_gdf.rename(columns={'NAME_right': 'NAME', 'REGION_right': 'REGION', 'DIVISION_right': 'DIVISION'})
    print(f"空间连接完成，{len(joined_gdf)} 个要素被分配到州")

    # 创建州分组策略
    # 1. 按REGION和DIVISION分组
    # 2. 优先将同一REGION的州放在同一组
    # 3. 每组3个州，其中两组4个州
    print("创建州分组策略...")

    # 获取每个州的要素数量，用于平衡分组
    state_counts = joined_gdf['NAME'].value_counts().reset_index()
    state_counts.columns = ['NAME', 'count']

    # 将计数合并回州数据
    states_with_counts = states_gdf[['NAME', 'REGION', 'DIVISION']].merge(
        state_counts, on='NAME', how='left'
    ).fillna(0)

    # 按REGION和DIVISION排序
    states_sorted = states_with_counts.sort_values(['REGION', 'DIVISION', 'NAME'])

    # 创建分组
    groups = {}
    group_id = 1

    # 首先按REGION分组
    region_groups = {}
    for region, region_df in states_sorted.groupby('REGION'):
        # 在每个REGION内按DIVISION分组
        for division, division_df in region_df.groupby('DIVISION'):
            # 为当前DIVISION内的州创建组
            states_in_division = division_df['NAME'].tolist()

            # 将州分配到组
            for i, state in enumerate(states_in_division):
                if i % 3 == 0 and i > 0:  # 每3个州创建一个新组
                    group_id += 1

                if group_id not in groups:
                    groups[group_id] = []

                groups[group_id].append(state)

    # 调整组大小，确保大部分组有3个州，只有两组有4个州
    # 首先，将所有组扁平化为州列表
    all_states = []
    for group_id, states in groups.items():
        all_states.extend(states)

    # 重新分组，确保每组3-4个州
    groups = {}
    group_id = 1
    current_group = []

    for state in all_states:
        current_group.append(state)

        # 除最后两组外，每组3个州
        if len(current_group) == 3 and group_id <= 14:
            groups[group_id] = current_group.copy()
            current_group = []
            group_id += 1
        # 最后两组每组4个州
        elif len(current_group) == 4 and group_id > 14:
            groups[group_id] = current_group.copy()
            current_group = []
            group_id += 1

    # 处理剩余的州
    if current_group:
        if group_id <= 14:
            # 将剩余州分配给前面的组
            for state in current_group:
                if group_id <= 14:
                    groups.setdefault(group_id, []).append(state)
                    group_id += 1
                else:
                    # 将其余州分配给最后一组
                    groups.setdefault(16, []).append(state)
        else:
            groups.setdefault(group_id, []).extend(current_group)

    # 确保有16个组
    while len(groups) < 16:
        groups[len(groups) + 1] = []

    # 打印分组信息
    print("\n州分组方案:")
    total_states = 0
    for gid, states in groups.items():
        total_states += len(states)
        region_info = []
        for state in states:
            region = states_sorted[states_sorted['NAME'] == state]['REGION'].values[0]
            region_info.append(f"{state}({region})")
        print(f"组 {gid} ({len(states)}个州): {', '.join(region_info)}")
    print(f"总共分组了 {total_states} 个州")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 为每个组创建输出文件
    output_files = []

    print("\n开始创建分组文件...")
    for group_id, states in groups.items():
        if not states:  # 跳过空组
            continue

        print(f"\n处理组 {group_id}，包含州: {', '.join(states)}")

        # 获取属于当前组的要素
        group_gdf = joined_gdf[joined_gdf['NAME'].isin(states)].copy()

        print(f"  该组包含 {len(group_gdf)} 个要素")

        if not group_gdf.empty:
            # 特殊处理：第15组需要拆分成两个文件
            if group_id == 13:
                print(f"  注意: 组 {group_id} 需要拆分成两个文件")

                # 将数据拆分成两半
                half_index = len(group_gdf) // 2
                group_gdf_1 = group_gdf.iloc[:half_index].copy()
                group_gdf_2 = group_gdf.iloc[half_index:].copy()

                # 创建组输出目录


                # 生成输出文件名 - 添加_1和_2后缀
                input_name = os.path.splitext(os.path.basename(input_shp_path))[0]

                # 第一个文件
                group_dir_1 = os.path.join(output_dir, f"group_{group_id}_1")
                os.makedirs(group_dir_1, exist_ok=True)
                output_filename_1 = f"{input_name}_group{group_id}_1.shp"
                output_path_1 = os.path.join(group_dir_1, output_filename_1)
                group_gdf_1.drop(columns=['index_right'], errors='ignore').to_file(output_path_1)
                print(f"  已创建 {output_filename_1}，包含 {len(group_gdf_1)} 个要素")
                output_files.append(output_path_1)

                # 第二个文件
                group_dir_2 = os.path.join(output_dir, f"group_{group_id}_2")
                os.makedirs(group_dir_2, exist_ok=True)
                output_filename_2 = f"{input_name}_group{group_id}_2.shp"
                output_path_2 = os.path.join(group_dir_2, output_filename_2)
                group_gdf_2.drop(columns=['index_right'], errors='ignore').to_file(output_path_2)
                print(f"  已创建 {output_filename_2}，包含 {len(group_gdf_2)} 个要素")
                output_files.append(output_path_2)

            else:
                # 创建组输出目录
                group_dir = os.path.join(output_dir, f"group_{group_id}")
                os.makedirs(group_dir, exist_ok=True)

                # 生成输出文件名
                input_name = os.path.splitext(os.path.basename(input_shp_path))[0]
                output_filename = f"{input_name}_group{group_id}.shp"
                output_path = os.path.join(group_dir, output_filename)

                # 保存到文件
                group_gdf.drop(columns=['index_right'], errors='ignore').to_file(output_path)
                print(f"  已创建 {output_filename}")
                output_files.append(output_path)
        else:
            print(f"  警告: 组 {group_id} 没有包含任何要素，跳过创建文件")

    print(f"\n处理完成，共生成 {len(output_files)} 个输出文件")
    return output_files


if __name__ == '__main__':
    input_shp = r"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\GeoDAR_v11_dams_of_USA\GeoDAR_v11_dams_of_USA.shp"
    states_shp = r"C:\Users\Kevin\Documents\ResearchData\RangeOfUSA\States.shp"
    output_dir = r"C:\Users\Kevin\Documents\ResearchData\GeoDAR_v10_v11\USA_DAM_ByRegion"

    split_shp_by_us_regions(input_shp, states_shp, output_dir)