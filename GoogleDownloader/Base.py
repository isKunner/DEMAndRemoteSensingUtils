#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: google_tile_downloader.py
# @Time    : 2026/4/21
# @Author  : Kevin
# @Describe: 基于 GeoTIFF 范围的 Google 瓦片下载器

import math
import os
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import geopandas as gpd

try:
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
    from rasterio.transform import Affine, array_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from functools import lru_cache


@lru_cache(maxsize=2048)
def load_google_tile_cached(tile_path):
    """
    带 LRU 缓存的 Google tile 读取函数。

    作用：
        相邻 DEM tile 会反复用到同一批 Google JPG。
        用缓存后，同一个 JPG 不会被反复从硬盘读取。

    参数：
        tile_path: Google JPG 完整路径

    返回：
        PIL.Image，RGB 模式

    注意：
        必须使用 img.copy()，否则 Image.open 是懒加载，
        文件句柄可能一直保持打开状态。
    """
    with Image.open(tile_path) as img:
        return img.convert("RGB").copy()

class GoogleTileDownloader:
    """
    基于 GeoTIFF 地理范围下载 Google 卫星瓦片。
    支持原样保存 JPEG 或导出为带 WGS84 坐标的 GeoTIFF。
    """

    TILE_URL_TEMPLATE = "http://arcmap.googlecnapps.club/maps/vt?lyrs=s&x={x}&y={y}&z={z}&s=Ga"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0.36"
    }

    # ------------------------------------------------------------------ #
    # 静态工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def lonlat_to_tile(lon, lat, zoom):
        """经纬度 → 瓦片编号 (tx, ty)"""
        n = 2.0 ** zoom
        x = math.floor((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return int(x), int(y)

    @staticmethod
    def lonlat_to_global_pixel(lon, lat, zoom):
        sin_lat = math.sin(math.radians(lat))
        sin_lat = min(max(sin_lat, -0.9999), 0.9999)

        n = 2 ** zoom
        x = (lon + 180.0) / 360.0 * n * 256.0
        y = (0.5- math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * 256.0

        return x, y


    @staticmethod
    def tile_bounds(tx, ty, z):
        """瓦片编号 → 地理边界 (left, bottom, right, top)，WGS84"""
        n = 2.0 ** z
        left = tx / n * 360.0 - 180.0
        right = (tx + 1) / n * 360.0 - 180.0
        top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
        bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
        return left, bottom, right, top

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(self, output_dir, zoom_level, save_format='jpg',
                 max_workers=16, max_retry=3, timeout=15):
        """
        Args:
            output_dir: 输出目录（最终目录，不建子文件夹）
            zoom_level: Google Zoom Level (1-20)
            save_format: 'jpg' | 'tif'
            max_workers: 并发线程数
            max_retry: 单瓦片重试次数
            timeout: 请求超时秒数
        """
        assert save_format in ('jpg', 'tif'), "save_format 必须是 jpg 或 tif"
        if save_format == 'tif' and not HAS_RASTERIO:
            raise ImportError("保存为 TIF 需要 rasterio，请先安装")

        self.output_dir = output_dir
        self.zoom_level = zoom_level
        self.save_format = save_format
        self.max_workers = max_workers
        self.max_retry = max_retry
        self.timeout = timeout

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 核心：单瓦片下载（原样字节）
    # ------------------------------------------------------------------ #
    def _download_raw(self, tx, ty):
        """
        下载单张瓦片原始字节。
        Returns:
            (tx, ty, bytes)  失败返回 (tx, ty, None)
        """
        url = self.TILE_URL_TEMPLATE.format(x=tx, y=ty, z=self.zoom_level)
        for i in range(self.max_retry):
            try:
                resp = requests.get(url, headers=self.HEADERS,
                                    timeout=self.timeout)
                if resp.status_code == 200:
                    return tx, ty, resp.content
                time.sleep(0.3 * (i + 1))
            except Exception:
                if i == self.max_retry - 1:
                    return tx, ty, None
                time.sleep(0.3 * (i + 1))
        return tx, ty, None

    # ------------------------------------------------------------------ #
    # 核心：单瓦片保存（边下边存）
    # ------------------------------------------------------------------ #
    def _save_tile(self, tx, ty, content):
        """
        将原始字节保存为 jpg 或转换为 GeoTIFF。
        文件已存在则直接跳过（断点续传）。
        Returns:
            bool  是否成功（True=成功或已存在，False=失败）
        """
        ext = 'jpg' if self.save_format == 'jpg' else 'tif'
        out_path = os.path.join(self.output_dir,
                                f"{self.zoom_level}_{tx}_{ty}.{ext}")

        if os.path.exists(out_path):
            return True

        if self.save_format == 'jpg':
            with open(out_path, 'wb') as f:
                f.write(content)
            return True

        # --- tif 分支：JPEG → GeoTIFF（带 WGS84 坐标）---
        try:
            img = Image.open(BytesIO(content))
            arr = np.array(img)

            if len(arr.shape) == 3:
                arr = arr.transpose(2, 0, 1)
                count = arr.shape[0]
            else:
                arr = arr[np.newaxis, ...]
                count = 1

            left, bottom, right, top = self.tile_bounds(tx, ty, self.zoom_level)
            transform = Affine((right - left) / 256.0, 0, left,
                               0, (bottom - top) / 256.0, top)

            with rasterio.open(
                out_path, 'w',
                driver='GTiff',
                height=256,
                width=256,
                count=count,
                dtype=arr.dtype,
                crs='EPSG:4326',
                transform=transform,
            ) as dst:
                dst.write(arr)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # 对外接口：按 TIF 范围下载
    # ------------------------------------------------------------------ #
    def download_from_tif(self, tif_path):
        """
        读取 GeoTIFF 边界，下载范围内所有 Google 瓦片。
        边下边存，已存在则跳过。
        """
        if not HAS_RASTERIO:
            raise ImportError("读取 TIF 需要 rasterio")

        with rasterio.open(tif_path) as src:
            if src.crs and src.crs.to_epsg() != 4326:
                left, bottom, right, top = transform_bounds(
                    src.crs, 'EPSG:4326',
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top
                )
            else:
                left, bottom, right, top = (
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top
                )

        # 边界 → 瓦片编号范围
        tx_min, ty_max = self.lonlat_to_tile(left, bottom, self.zoom_level)
        tx_max, ty_min = self.lonlat_to_tile(right, top, self.zoom_level)

        n_x = tx_max - tx_min + 1
        n_y = ty_max - ty_min + 1
        total = n_x * n_y

        print(f"TIF: {os.path.basename(tif_path)}")
        print(f"边界: {left:.5f}, {bottom:.5f}, {right:.5f}, {top:.5f}")
        print(f"瓦片: x[{tx_min},{tx_max}] y[{ty_min},{ty_max}] = "
              f"{n_x}×{n_y}={total}张")

        # 统计已存在（断点续传）
        ext = 'jpg' if self.save_format == 'jpg' else 'tif'
        existing = 0
        pending = []
        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                if os.path.exists(
                    os.path.join(self.output_dir,
                                 f"{self.zoom_level}_{tx}_{ty}.{ext}")
                ):
                    existing += 1
                else:
                    pending.append((tx, ty))

        if existing:
            print(f"已存在 {existing} 张，跳过")
        if not pending:
            print("全部瓦片已存在，无需下载")
            return

        print(f"待下载: {len(pending)} 张")

        # 并发下载 & 立即保存
        fails = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_raw, tx, ty): (tx, ty)
                for tx, ty in pending
            }

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"ZL{self.zoom_level}"):
                tx, ty, content = future.result()
                if content is None:
                    fails.append((tx, ty))
                    continue

                ok = self._save_tile(tx, ty, content)
                if not ok:
                    fails.append((tx, ty))

        print(f"完成: {total - len(fails)}/{total}")
        if fails:
            print(f"失败 {len(fails)} 张: {fails[:10]}...")
        print(f"输出: {self.output_dir}")

    # ------------------------------------------------------------------ #
    # 对外接口：按自定义经纬度范围下载
    # ------------------------------------------------------------------ #
    def download_from_bounds(self, left, bottom, right, top):
        """
        直接给定 WGS84 经纬度范围下载。
        """
        tx_min, ty_max = self.lonlat_to_tile(left, bottom, self.zoom_level)
        tx_max, ty_min = self.lonlat_to_tile(right, top, self.zoom_level)

        n_x = tx_max - tx_min + 1
        n_y = ty_max - ty_min + 1
        total = n_x * n_y

        print(f"边界: {left:.5f}, {bottom:.5f}, {right:.5f}, {top:.5f}")
        print(f"瓦片: x[{tx_min},{tx_max}] y[{ty_min},{ty_max}] = "
              f"{n_x}×{n_y}={total}张")

        ext = 'jpg' if self.save_format == 'jpg' else 'tif'
        pending = []
        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                if not os.path.exists(
                    os.path.join(self.output_dir,
                                 f"{self.zoom_level}_{tx}_{ty}.{ext}")
                ):
                    pending.append((tx, ty))

        if not pending:
            print("全部瓦片已存在，无需下载")
            return

        print(f"待下载: {len(pending)} 张")

        fails = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_raw, tx, ty): (tx, ty)
                for tx, ty in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"ZL{self.zoom_level}"):
                tx, ty, content = future.result()
                if content is None:
                    fails.append((tx, ty))
                    continue
                if not self._save_tile(tx, ty, content):
                    fails.append((tx, ty))

        print(f"完成: {total - len(fails)}/{total}")
        if fails:
            print(f"失败 {len(fails)} 张: {fails[:10]}...")
        print(f"输出: {self.output_dir}")

    def download_from_shp(self, shp_path: str, key_field=None):
        """
        根据 SHP 文件下载瓦片。

        Args:
            shp_path: SHP 文件路径
            key_field: str or None，如果为 None，下载到原始 output_dir
                       如果有值，下载到 output_dir/row[key_field]
        """

        gdf = gpd.read_file(shp_path)

        for idx, row in gdf.iterrows():
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds

            # 输出目录处理
            if key_field is None:
                output_dir = self.output_dir
            else:
                subdir_name = str(row.get(key_field, key_field))
                output_dir = os.path.join(self.output_dir, subdir_name)
            os.makedirs(output_dir, exist_ok=True)

            # 临时修改 self.output_dir 以保持 download_from_bounds 不变
            original_dir = self.output_dir
            self.output_dir = output_dir

            # 调用原有下载函数
            self.download_from_bounds(minx, miny, maxx, maxy)

            # 恢复原来的输出目录
            self.output_dir = original_dir

    def build_google_tile_index(self, google_tile_dir, google_ext="jpg"):
        """
        一次性扫描 Google JPG 目录，建立内存索引。

        Google 瓦片命名格式必须是:
            {zoom}_{tx}_{ty}.{google_ext}

        例如:
            18_42921_100876.jpg

        返回:
            tile_index: dict
                key   = (tx, ty)
                value = JPG 完整路径

        用途:
            避免每个 DEM 反复在 50 万个 JPG 文件目录里查找，
            后续只需要 tile_index.get((tx, ty))，速度会快很多。
        """

        tile_index = {}

        suffix = f".{google_ext.lower()}"
        prefix = f"{self.zoom_level}_"

        for file_name in os.listdir(google_tile_dir):
            lower_name = file_name.lower()

            if not lower_name.endswith(suffix):
                continue

            if not file_name.startswith(prefix):
                continue

            name_no_ext = os.path.splitext(file_name)[0]
            parts = name_no_ext.split("_")

            if len(parts) != 3:
                continue

            try:
                z = int(parts[0])
                tx = int(parts[1])
                ty = int(parts[2])
            except ValueError:
                continue

            if z != self.zoom_level:
                continue

            tile_index[(tx, ty)] = os.path.join(google_tile_dir, file_name)

        print(f"Google JPG 索引建立完成: {len(tile_index)} 张")
        return tile_index

    def make_expanded_google_image_for_one_dem_warped(
            self,
            dem_path,
            google_tile_dir,
            output_path,
            target_size=1024,
            google_ext="jpg",
            save_with_geo=False,
            missing_policy="any",  # "any": 缺一张就跳过；"all": 全部缺失才跳过
            tile_index=None,
            fill_color=(0, 0, 0),
            resampling=Resampling.bilinear
    ):
        """
        根据一个 DEM tile，生成外扩到 target_size×target_size 的 Google 影像。

        这个版本不再用 WGS84 bbox 直接裁剪后 resize，
        而是：
            1. 以 DEM 中心为准，计算 target_size×target_size 的扩展范围；
            2. 将扩展范围转为 WGS84，只用于判断需要哪些 Google tile；
            3. 拼接这些 Google tile，得到 Web Mercator mosaic；
            4. 将 Web Mercator mosaic 重投影到 DEM 的 CRS + expanded_transform；
            5. 输出与 DEM 扩展网格严格对齐的 1024×1024 影像。

        参数：
            dem_path:
                单个 DEM tile 路径，例如 448×448 的 USGS DEM。

            google_tile_dir:
                已下载的 Google 瓦片目录。
                文件名格式应为：
                    {zoom}_{tx}_{ty}.{google_ext}
                例如：
                    18_42921_100876.jpg

            output_path:
                输出路径。
                save_with_geo=False 时，可以是 .jpg / .png。
                save_with_geo=True 时，建议是 .tif。

            target_size:
                输出尺寸，例如 1024。

            save_with_geo:
                False：保存普通图像，不带坐标。
                True ：保存 GeoTIFF，带 DEM CRS 和 expanded_transform。

            missing_policy:
                "any":
                    只要缺一张 Google tile，就跳过保存。

                "all":
                    只有所有 Google tile 都缺失时才跳过。
                    部分缺失时，缺失区域用 fill_color 填充。

            tile_index:
                Google tile 索引，建议提前用 build_google_tile_index() 建好。
                key=(tx, ty), value=tile_path。

        返回：
            dict，记录处理状态。
        """

        if missing_policy not in ("any", "all"):
            raise ValueError("missing_policy 只能是 'any' 或 'all'")

        WEB_MERCATOR_CRS = "EPSG:3857"
        R = 6378137.0
        ORIGIN_SHIFT = math.pi * R
        resolution = 2 * math.pi * R / (256 * (2 ** self.zoom_level))

        if tile_index is None:
            tile_index = self.build_google_tile_index(
                google_tile_dir=google_tile_dir,
                google_ext=google_ext
            )

        with rasterio.open(dem_path) as src:
            src_width = src.width
            src_height = src.height
            src_transform = src.transform
            src_crs = src.crs

            if src_crs is None:
                raise ValueError(f"DEM 没有 CRS: {dem_path}")

            if target_size < src_width or target_size < src_height:
                raise ValueError(
                    f"target_size={target_size} 小于 DEM 尺寸 {src_width}×{src_height}"
                )

            # 以 DEM 中心为准扩展，例如 448 -> 1024，各方向扩展 288 像素
            pad_x = (target_size - src_width) / 2.0
            pad_y = (target_size - src_height) / 2.0

            # 保持 DEM 分辨率不变，只移动左上角
            expanded_transform = src_transform * Affine.translation(-pad_x, -pad_y)

            # 扩展后区域在 DEM 原始 CRS 下的范围
            exp_left, exp_bottom, exp_right, exp_top = array_bounds(
                target_size,
                target_size,
                expanded_transform
            )

            # 转 WGS84，只用于确定需要哪些 Google tile
            left_wgs84, bottom_wgs84, right_wgs84, top_wgs84 = transform_bounds(
                src_crs,
                "EPSG:4326",
                exp_left,
                exp_bottom,
                exp_right,
                exp_top,
                densify_pts=21
            )

        # 计算需要的 Google tile 范围
        tx_min, ty_max = self.lonlat_to_tile(left_wgs84, bottom_wgs84, self.zoom_level)
        tx_max, ty_min = self.lonlat_to_tile(right_wgs84, top_wgs84, self.zoom_level)

        tx_min -= 1
        tx_max += 1
        ty_min -= 1
        ty_max += 1

        required_tiles = []
        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                tile_name = f"{self.zoom_level}_{tx}_{ty}.{google_ext}"
                tile_path = tile_index.get((tx, ty))
                required_tiles.append((tx, ty, tile_name, tile_path))

        missing_tiles = [
            tile_name
            for _, _, tile_name, tile_path in required_tiles
            if tile_path is None
        ]

        if missing_policy == "any" and len(missing_tiles) > 0:
            # print(f"⚠️ 跳过保存: {os.path.basename(dem_path)}")
            # print(f"原因: 缺失 Google tile {len(missing_tiles)} / {len(required_tiles)} 张")
            # print(f"示例缺失: {missing_tiles[:10]}")
            return {
                "status": "skipped_any",
                "dem": dem_path,
                "output": output_path,
                "required_tile_count": len(required_tiles),
                "missing_tile_count": len(missing_tiles),
                "missing_tiles": missing_tiles
            }

        if missing_policy == "all" and len(missing_tiles) == len(required_tiles):
            # print(f"⚠️ 跳过保存: {os.path.basename(dem_path)}")
            # print(f"原因: 所需 Google tile 全部缺失，共 {len(required_tiles)} 张")
            return {
                "status": "skipped_all",
                "dem": dem_path,
                "output": output_path,
                "required_tile_count": len(required_tiles),
                "missing_tile_count": len(missing_tiles),
                "missing_tiles": missing_tiles
            }

        # 拼接 Web Mercator mosaic
        mosaic_width = (tx_max - tx_min + 1) * 256
        mosaic_height = (ty_max - ty_min + 1) * 256

        mosaic = Image.new("RGB", (mosaic_width, mosaic_height), fill_color)

        for tx, ty, tile_name, tile_path in required_tiles:
            if tile_path is None:
                continue

            paste_x = (tx - tx_min) * 256
            paste_y = (ty - ty_min) * 256

            try:
                tile_img = load_google_tile_cached(tile_path)
                mosaic.paste(tile_img, (paste_x, paste_y))
            except Exception:
                missing_tiles.append(tile_name)

        src_arr = np.array(mosaic).astype(np.uint8)

        # HWC -> CHW
        src_arr = src_arr.transpose(2, 0, 1)

        # mosaic 左上角在 EPSG:3857 下的坐标
        mosaic_left = tx_min * 256 * resolution - ORIGIN_SHIFT
        mosaic_top = ORIGIN_SHIFT - ty_min * 256 * resolution

        mosaic_transform = Affine(
            resolution, 0.0, mosaic_left,
            0.0, -resolution, mosaic_top
        )

        dst_arr = np.zeros(
            (3, target_size, target_size),
            dtype=np.uint8
        )

        # 关键：把 Google Web Mercator mosaic 重投影到 DEM 的扩展网格
        for band_idx in range(3):
            reproject(
                source=src_arr[band_idx],
                destination=dst_arr[band_idx],
                src_transform=mosaic_transform,
                src_crs=WEB_MERCATOR_CRS,
                dst_transform=expanded_transform,
                dst_crs=src_crs,
                resampling=resampling,
                src_nodata=None,
                dst_nodata=0
            )

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if save_with_geo:
            with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=target_size,
                    width=target_size,
                    count=3,
                    dtype=np.uint8,
                    crs=src_crs,
                    transform=expanded_transform
            ) as dst:
                dst.write(dst_arr)
        else:
            out_img = Image.fromarray(dst_arr.transpose(1, 2, 0))
            out_img.save(output_path)

        # print(f"✅ 保存完成: {output_path}")

        return {
            "status": "saved",
            "dem": dem_path,
            "output": output_path,
            "original_size": (src_width, src_height),
            "target_size": target_size,
            "required_tile_count": len(required_tiles),
            "missing_tile_count": len(missing_tiles),
            "missing_tiles": missing_tiles,
            "expanded_bounds_dem_crs": (
                exp_left,
                exp_bottom,
                exp_right,
                exp_top
            ),
            "expanded_bounds_wgs84": (
                left_wgs84,
                bottom_wgs84,
                right_wgs84,
                top_wgs84
            )
        }

    def batch_make_expanded_google_images_parallel(
            self,
            dem_dir,
            google_tile_dir,
            output_dir,
            target_size=1024,
            google_ext="jpg",
            output_ext="jpg",
            save_with_geo=False,
            missing_policy="any",
            tile_index=None,
            max_workers=6
    ):
        """
        类内批量并行处理 DEM tile，生成对应的扩展 Google 影像。

        依赖类中已有函数：
            self.build_google_tile_index(...)
            self.make_expanded_google_image_for_one_dem_warped(...)

        参数:
            dem_dir:
                已切分好的 DEM tile 文件夹。

            google_tile_dir:
                已下载好的 Google tile 文件夹。

            output_dir:
                输出 Google 影像文件夹。

            target_size:
                输出图像大小，例如 1024。

            google_ext:
                已下载 Google tile 的后缀，例如 "jpg"。

            output_ext:
                输出普通图像的后缀，例如 "jpg"。
                当 save_with_geo=True 时，输出会强制为 tif。

            save_with_geo:
                False: 输出普通 jpg/png，不带坐标。
                True : 输出 GeoTIFF，带 DEM 扩展后的 transform 和 CRS。

            missing_policy:
                "any": 只要缺一张 Google tile 就跳过保存。
                "all": 只有全部 Google tile 缺失才跳过保存。

            tile_index:
                Google tile 索引。
                如果已经提前 build，可以传入，避免重复扫描目录。
                如果为 None，则函数内部自动建立一次。

            max_workers:
                并行线程数。
                建议 4-8，太大会增加内存和磁盘 IO 压力。

        返回:
            results: list[dict]
                每个 DEM 的处理结果。
        """

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        os.makedirs(output_dir, exist_ok=True)

        if tile_index is None:
            tile_index = self.build_google_tile_index(
                google_tile_dir=google_tile_dir,
                google_ext=google_ext
            )

        dem_files = [
            f for f in os.listdir(dem_dir)
            if f.lower().endswith(".tif")
        ]
        dem_files.sort()

        print(f"DEM 数量: {len(dem_files)}")
        print(f"并行线程数: {max_workers}")

        def process_one(file_name):
            dem_path = os.path.join(dem_dir, file_name)
            base_name = os.path.splitext(file_name)[0]

            if save_with_geo:
                output_path = os.path.join(output_dir, f"{base_name}.tif")
            else:
                output_path = os.path.join(output_dir, f"{base_name}.{output_ext}")

            # 断点续跑：输出已存在则跳过
            if os.path.exists(output_path):
                return {
                    "status": "exists",
                    "dem": dem_path,
                    "output": output_path,
                    "missing_tile_count": 0
                }

            try:
                result = self.make_expanded_google_image_for_one_dem_warped(
                    dem_path=dem_path,
                    google_tile_dir=google_tile_dir,
                    output_path=output_path,
                    target_size=target_size,
                    google_ext=google_ext,
                    save_with_geo=save_with_geo,
                    missing_policy=missing_policy,
                    tile_index=tile_index
                )
                return result

            except Exception as e:
                return {
                    "status": "error",
                    "dem": dem_path,
                    "output": output_path,
                    "error": str(e)
                }

        results = []
        status_count = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one, file_name): file_name
                for file_name in dem_files
            }

            for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Matching Google"
            ):
                result = future.result()
                results.append(result)

                status = result.get("status", "unknown")
                status_count[status] = status_count.get(status, 0) + 1

        print("\n处理完成:")
        for status, count in status_count.items():
            print(f"  {status}: {count}")

        return results


# ---------------------------------------------------------------------- #
# 使用示例（直接运行此文件时）
# ---------------------------------------------------------------------- #
if __name__ == "__main__":

    dl = GoogleTileDownloader(
        output_dir=r"E:\Data\ResearchData\USA_ByState\GoogleRemoteSensing\Test2",
        zoom_level=18,
        save_format="jpg",
        max_workers=8
    )

    # dl.download_from_shp(shp_path=r"E:\Data\ResearchData\USA_ByState\RangeSHP\SuperResolution_Selected_Region.shp",
    #                      key_field="Name")

    google_tile_dir = r"E:\Data\ResearchData\USA_ByState\GoogleRemoteSensing\Test2\Arable Land"
    dem_dir = r"E:\Data\ResearchData\USA_ByState\USGSDEM\ArableLand"
    output_dir = r"E:\Data\ResearchData\USA_ByState\GoogleRemoteSensing\ArableLand"

    tile_index = dl.build_google_tile_index(
        google_tile_dir=google_tile_dir,
        google_ext="jpg"
    )

    dl.batch_make_expanded_google_images_parallel(dem_dir=dem_dir, google_tile_dir=google_tile_dir, output_dir=output_dir, target_size=1024, tile_index=tile_index, max_workers=8)


    # 示例 1：从 TIF 读取范围，保存为原图 jpg
    # dl = GoogleTileDownloader(
    #     output_dir=r"C:\Users\Kevin\Documents\ResearchData\LongQuan\Google",
    #     zoom_level=18,
    #     save_format='jpg',
    #     max_workers=16
    # )
    # dl.download_from_tif(
    #     r"C:\Users\Kevin\Documents\ResearchData\LongQuan\longQuan_50cm.tif"
    # )

    # 示例 2：从 TIF 读取范围，保存为带坐标的 GeoTIFF
    # dl = GoogleTileDownloader(
    #     output_dir=r"C:\Users\Kevin\Documents\ResearchData\RelevantDocument\RemoteSensing\RemoteSAM-master\Test\WangMao",
    #     zoom_level=17,
    #     save_format='tif',
    #     max_workers=16
    # )
    # dl.download_from_tif(
    #     r"C:\Users\Kevin\Documents\ResearchData\WangMao\cleaned_dem.tif"
    # )

    # # 示例 3：直接给定经纬度范围下载
    # dl = GoogleTileDownloader(
    #     output_dir=r"C:\Users\Kevin\Desktop\test",
    #     zoom_level=19,
    #     save_format='jpg'
    # )
    # dl.download_from_bounds(
    #     left=-118.607, bottom=37.1818, right=-118.606, top=37.1825
    # )