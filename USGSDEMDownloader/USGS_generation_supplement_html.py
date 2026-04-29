#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: USGS_generation_supplement_html
# @Time    : 2026/3/20 10:10
# @Author  : Kevin
# @Describe:

import json
import os
from datetime import datetime
from pathlib import Path

try:
    import rasterio
    import numpy as np
except ImportError:
    print("请先安装 rasterio: pip install rasterio")
    raise

# ===================== 配置项（与原代码保持一致） =====================
# JSON文件路径
JSON_FILE_PATH = r"D:\Data\ResearchData\USA_ByState\USGSDEM\DownloadLink.json"
# HTML文件输出目录（与JSON同目录）
HTML_OUTPUT_DIR = r"D:\Data\ResearchData\USA_ByState\USGSDEM"
# TIF文件基础目录（与JSON同目录，即 USGSDEM 文件夹内按group分子文件夹）
TIF_BASE_DIR = r"D:\Data\ResearchData\USA_ByState\USGSDEM"
# 输出HTML文件名
OUTPUT_HTML_NAME = "Supplement.html"

# ===================== HTML模板 =====================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>USGS DEM 补充下载清单 - Supplement</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
        }}
        body {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #2c3e50;
            line-height: 1.6;
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 2rem;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 2rem;
            font-size: 1rem;
        }}

        /* 统计面板 */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}
        .stat-box {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border-top: 4px solid;
        }}
        .stat-box.valid {{ border-color: #27ae60; }}
        .stat-box.missing {{ border-color: #e74c3c; }}
        .stat-box.nodata {{ border-color: #f39c12; }}

        .stat-num {{
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }}
        .stat-box.valid .stat-num {{ color: #27ae60; }}
        .stat-box.missing .stat-num {{ color: #e74c3c; }}
        .stat-box.nodata .stat-num {{ color: #f39c12; }}
        .stat-label {{ color: #555; font-weight: 600; }}

        /* 主要内容区 */
        .main-content {{
            margin-top: 2rem;
        }}
        .section-title {{
            font-size: 1.3rem;
            color: #34495e;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #3498db;
        }}

        /* Group 样式 */
        .group-section {{
            margin-bottom: 2rem;
            background: #fff;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            overflow: hidden;
        }}
        .group-header {{
            background: #34495e;
            color: white;
            padding: 1rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .group-count {{
            background: rgba(255,255,255,0.2);
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
        }}

        /* File 样式 */
        .file-list {{
            padding: 1rem;
        }}
        .file-item {{
            background: #f8f9fa;
            border-left: 4px solid;
            border-radius: 0 8px 8px 0;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }}
        .file-item.missing {{ border-color: #e74c3c; }}
        .file-item.nodata {{ border-color: #f39c12; }}

        .file-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .file-name {{
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.1rem;
        }}
        .status-tag {{
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            color: white;
        }}
        .status-tag.missing {{ background: #e74c3c; }}
        .status-tag.nodata {{ background: #f39c12; }}

        .file-path {{
            color: #7f8c8d;
            font-size: 0.85rem;
            margin-bottom: 0.8rem;
            font-family: 'Courier New', monospace;
            word-break: break-all;
        }}
        .nodata-info {{
            color: #e67e22;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
            font-style: italic;
        }}

        /* 链接区域 */
        .links-box {{
            background: white;
            border-radius: 6px;
            padding: 0.8rem;
            border: 1px solid #e1e8ed;
        }}
        .link-row {{
            margin: 0.4rem 0;
            padding: 0.4rem;
            background: #f1f2f6;
            border-radius: 4px;
            font-size: 0.9rem;
            word-break: break-all;
        }}
        a {{
            color: #2980b9;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
            color: #e74c3c;
        }}

        .empty-notice {{
            text-align: center;
            color: #27ae60;
            font-size: 1.2rem;
            padding: 3rem;
            background: #d4edda;
            border-radius: 8px;
            margin: 2rem 0;
        }}

        .footer {{
            margin-top: 3rem;
            text-align: center;
            color: #95a5a6;
            font-size: 0.9rem;
            padding-top: 2rem;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📥 USGS DEM 补充下载清单</h1>
        <p class="subtitle">仅包含需要重新下载的文件（缺失或存在Nodata值）</p>

        <!-- 统计面板 -->
        <div class="stats-grid">
            <div class="stat-box valid">
                <div class="stat-label">✅ 已有效下载</div>
                <div class="stat-num">{valid_count}</div>
                <div style="font-size:0.85rem;color:#666">无需处理</div>
            </div>
            <div class="stat-box missing">
                <div class="stat-label">❌ 文件缺失</div>
                <div class="stat-num">{missing_count}</div>
                <div style="font-size:0.85rem;color:#666">需重新下载</div>
            </div>
            <div class="stat-box nodata">
                <div class="stat-label">⚠️ 存在Nodata</div>
                <div class="stat-num">{nodata_count}</div>
                <div style="font-size:0.85rem;color:#666">需重新下载</div>
            </div>
        </div>

        <!-- 详细内容（仅有问题文件） -->
        <div class="main-content">
            <div class="section-title">🎯 待下载文件清单（共 {problem_count} 个）</div>
            {content}
        </div>

        <div class="footer">
            <p>生成时间: {generation_time} | 总计检查: {total_count} 个文件</p>
            <p>说明：本清单仅列出本地缺失或数据存在空洞（Nodata）的文件，有效文件已隐藏</p>
        </div>
    </div>
</body>
</html>
"""


def check_tif_status(tif_path):
    """
    检查TIF文件状态
    返回: (status, detail_msg)
    status: 'valid', 'missing', 'nodata'
    """
    if not os.path.exists(tif_path):
        return 'missing', "本地文件不存在"

    try:
        with rasterio.open(tif_path) as src:
            band1 = src.read(1)
            nodata_val = src.nodata

            # 检查定义的nodata
            if nodata_val is not None and nodata_val in band1:
                count = (band1 == nodata_val).sum()
                return 'nodata', f"包含Nodata值({nodata_val})，共{count}个像素"

            # 检查常见无效值
            for val in [-9999, -32768, -32767]:
                if val in band1:
                    count = (band1 == val).sum()
                    return 'nodata', f"包含疑似Nodata值({val})，共{count}个像素"

            # 检查NaN
            if np.isnan(band1).any():
                count = np.isnan(band1).sum()
                return 'nodata', f"包含NaN值，共{count}个像素"

            return 'valid', "数据完整无缺失"

    except Exception as e:
        return 'missing', f"文件读取失败: {str(e)}"


def is_error_entry(links):
    """检查是否为错误条目"""
    return any(key.startswith('__') for key in links.keys())


def generate_supplement():
    """生成补充下载清单"""

    # 确保输出目录存在
    if not os.path.exists(HTML_OUTPUT_DIR):
        os.makedirs(HTML_OUTPUT_DIR)

    # 读取JSON
    if not os.path.exists(JSON_FILE_PATH):
        print(f"错误: 未找到JSON文件 {JSON_FILE_PATH}")
        return False

    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 读取JSON失败 - {e}")
        return False

    # 统计数据
    stats = {'valid': 0, 'missing': 0, 'nodata': 0, 'total': 0}
    problem_files = []  # 存储有问题的文件信息

    # 遍历检查
    for group_name, files in data.items():
        if not files:
            continue

        print(f"Current processing: {group_name}")

        for file_name, links in files.items():
            if is_error_entry(links):
                continue

            stats['total'] += 1

            # 构建TIF路径：与JSON同目录，按group分子文件夹
            tif_path = os.path.join(TIF_BASE_DIR, group_name, file_name)

            status, msg = check_tif_status(tif_path)
            stats[status] += 1

            # 只记录有问题的文件
            if status != 'valid':
                problem_files.append({
                    'group': group_name,
                    'file': file_name,
                    'status': status,
                    'msg': msg,
                    'links': list(links.keys()),
                    'path': tif_path
                })

    # 构建HTML内容（仅有问题文件）
    content_html = ""

    if not problem_files:
        content_html = '<div class="empty-notice">🎉 恭喜！所有文件均已有效下载，无需补充下载</div>'
    else:
        # 按Group分组
        from itertools import groupby
        problem_files.sort(key=lambda x: x['group'])

        for group_name, items in groupby(problem_files, key=lambda x: x['group']):
            items_list = list(items)

            group_section = f"""
            <div class="group-section">
                <div class="group-header">
                    <span>📁 {group_name}</span>
                    <span class="group-count">{len(items_list)} 个文件</span>
                </div>
                <div class="file-list">
            """

            for item in items_list:
                # 构建链接HTML
                links_html = ""
                for link in item['links']:
                    links_html += f'<div class="link-row"><a href="{link}" target="_blank">{link}</a></div>'

                if not links_html:
                    links_html = '<div class="link-row" style="color:#e74c3c">⚠️ 无可用下载链接</div>'

                status_class = item['status']
                status_text = "❌ 文件缺失" if item['status'] == 'missing' else "⚠️ 存在Nodata"

                file_html = f"""
                    <div class="file-item {status_class}">
                        <div class="file-header">
                            <span class="file-name">{item['file']}</span>
                            <span class="status-tag {status_class}">{status_text}</span>
                        </div>
                        <div class="file-path">{item['path']}</div>
                        <div class="nodata-info">{item['msg']}</div>
                        <div class="links-box">
                            <div style="font-size:0.8rem;color:#666;margin-bottom:0.5rem">下载链接：</div>
                            {links_html}
                        </div>
                    </div>
                """
                group_section += file_html

            group_section += "</div></div>"
            content_html += group_section

    # 填充模板
    problem_count = len(problem_files)
    final_html = HTML_TEMPLATE.format(
        valid_count=stats['valid'],
        missing_count=stats['missing'],
        nodata_count=stats['nodata'],
        total_count=stats['total'],
        problem_count=problem_count,
        content=content_html,
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # 写入文件
    output_path = os.path.join(HTML_OUTPUT_DIR, OUTPUT_HTML_NAME)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)

        print(f"✅ 成功生成: {output_path}")
        print(f"\n📊 统计结果:")
        print(f"   ✅ 有效（已隐藏）: {stats['valid']} 个")
        print(f"   ❌ 文件缺失: {stats['missing']} 个")
        print(f"   ⚠️  存在Nodata: {stats['nodata']} 个")
        print(f"   📁 总计检查: {stats['total']} 个")
        print(f"   🎯 需补充下载: {problem_count} 个")
        return True
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return False


if __name__ == '__main__':
    generate_supplement()