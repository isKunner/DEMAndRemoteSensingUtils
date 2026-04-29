#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: USGS_DownLoad_with_Rectangle
# @Time    : 2026/4/28 16:39
# @Author  : Kevin
# @Describe:

import os
import json
import geopandas as gpd
from shapely.geometry import Point
import utm
from math import ceil


state_to_abbr = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    "american samoa": "AS", "district of columbia": "DC", "guam": "GU",
    "puerto rico": "PR", "commonwealth of the northern mariana islands": "MP",
    "united states virgin islands": "VI"
}

# 瓦片大小 10 km，步长 9 km
STEP_M = 9000

import os
import json
import geopandas as gpd
from shapely.geometry import Point
import utm

STEP_M = 9000  # 9km步长

import os
import json
import geopandas as gpd
from shapely.geometry import Point
import utm
from math import ceil

STEP_M = 9000  # 9km步长

def generate_usgs_links_by_bbox(min_lon, min_lat, max_lon, max_lat,
                                index_dir, usa_states_shp_path,
                                output_json_path,
                                group_name="default_group"):
    """
    根据矩形范围生成 USGS 1m DEM 下载链接
    - 每个瓦片名直接用 E/N 拼接生成
    - 保留原始下载索引匹配逻辑
    - JSON 结构保持 down_dict_info[group][file][link]=False
    - 找不到州/索引或匹配链接会记录 "__error__"
    """

    # 读取州边界
    usa_gdf = gpd.read_file(usa_states_shp_path)

    # 读取索引目录
    index_dict = {}
    for file_name in os.listdir(index_dir):
        key = file_name.split("_")[0]
        with open(os.path.join(index_dir, file_name), "r", encoding="UTF-8") as f:
            for line in f:
                index_dict.setdefault(key, []).append(line.strip())

    # 矩形四角转换 UTM，找最大最小
    min_E, min_N, zone_number, zone_letter = utm.from_latlon(min_lat, min_lon)
    max_E, max_N, _, _ = utm.from_latlon(max_lat, max_lon)

    easting_vals = list(range(int(min_E), int(max_E)+1, STEP_M))
    northing_vals = list(range(int(min_N), int(max_N)+1, STEP_M))

    down_dict_info = {group_name: {}}

    for E in easting_vals:
        for N in northing_vals:
            # 直接生成瓦片名
            x_idx = f"x{int(E//10000)}"
            y_idx = f"y{ceil(N/10000)}"
            file_name_part = f"{x_idx}{y_idx}"
            google_remote_file = file_name_part + ".tif"

            if google_remote_file not in down_dict_info[group_name]:
                down_dict_info[group_name][google_remote_file] = {}

            # 计算瓦片中心点找州
            lat_c, lon_c = utm.to_latlon(E, N, zone_number, zone_letter)
            point = Point(lon_c, lat_c)

            state_abbr = None
            for idx, row in usa_gdf.iterrows():
                if row.geometry.contains(point):
                    state_name = str(row['NAME']).lower()
                    if state_name in state_to_abbr:
                        state_abbr = state_to_abbr[state_name].upper()
                    break

            if not state_abbr or state_abbr not in index_dict:
                down_dict_info[group_name][google_remote_file]["__error__"] = "no_state_or_index"
                continue

            # print(f"州：{state_abbr}")
            # print(f"name: {file_name_part}")

            # 匹配索引链接
            link_found = False
            for link in index_dict[state_abbr]:
                if file_name_part in link:
                    down_dict_info[group_name][google_remote_file][link] = False
                    link_found = True
            if not link_found:
                down_dict_info[group_name][google_remote_file]["__error__"] = "no_link_found"

    # 保存 JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(down_dict_info, f, indent=2, ensure_ascii=False)

    print(f"✔ 完成，生成 {len(down_dict_info[group_name])} 个瓦片下载链接，保存到 {output_json_path}")
    return down_dict_info


if __name__ == '__main__':
    gdf = gpd.read_file(r"E:\Data\ResearchData\USA_ByState\RangeSHP\SuperResolution_Selected_Region.shp")
    index_dir = r"E:\Data\ResearchData\USA_ByState\USGSDEM\INDEX"
    usa_states_shp_path = r"C:\Users\Kevin\Documents\ResearchData\RangeOfUSA\States.shp"
    save_dir = r"E:\Data\ResearchData\USA_ByState\USGSDEM"
    for idx, row in gdf.iterrows():
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        name = row.get("Name", f"Feature_{idx}")

        generate_usgs_links_by_bbox(
            min_lon=minx, min_lat=miny, max_lon=maxx, max_lat=maxy,
            index_dir=index_dir,
            usa_states_shp_path=usa_states_shp_path,
            output_json_path=os.path.join(save_dir, f"{name}.json")
        )
