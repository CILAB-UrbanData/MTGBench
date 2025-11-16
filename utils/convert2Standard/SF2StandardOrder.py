#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将原始轨迹 CSV 转换为：每条轨迹一行，只保留
- 起点时间
- 起点经纬度
- 终点经纬度

不使用 offset 字段中的任何信息。
点序列在第 3 列（下标 3），格式为：
[(lon, lat, timestamp, speed), ...]
"""

import csv
import ast
import argparse
import geopandas as gpd
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

sf_tz = ZoneInfo("America/Los_Angeles")

def build_edge_region_map(shp_path, m, n, fid_col="fid"):
    """根据路网构造 edge_id -> region 映射，并返回 bbox。"""
    gdf = gpd.read_file(shp_path)

    # 自动兼容 fid / FID
    if fid_col not in gdf.columns:
        if "FID" in gdf.columns:
            fid_col = "FID"
        else:
            raise ValueError(f"shp 中找不到 fid 列（既没有 '{fid_col}' 也没有 'FID'"")")

    # bbox: minx, miny, maxx, maxy  (lon_min, lat_min, lon_max, lat_max)
    minx, miny, maxx, maxy = gdf.total_bounds

    dx = (maxx - minx) / n
    dy = (maxy - miny) / m
    if dx == 0 or dy == 0:
        raise ValueError("路网外包矩形大小为 0，检查坐标系 / 数据是否正确。")

    def point_to_region(lon, lat):
        """把一个经纬度点映射到 (row, col, region_id)。"""
        col = int((lon - minx) / dx)
        row = int((lat - miny) / dy)

        # 处理落在最右 / 最上侧边界上的点
        if col == n:
            col = n - 1
        if row == m:
            row = m - 1

        # 夹在合法范围内
        col = max(0, min(col, n - 1))
        row = max(0, min(row, m - 1))

        region_id = row * n + col
        return row, col, region_id

    edge2region = {}

    # 对每一条 edge，用几何中心（centroid）来决定 region
    for _, r in gdf.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        row, col, region_id = point_to_region(c.x, c.y)
        edge_id = r[fid_col]
        edge2region[edge_id] = {
            "row": row,
            "col": col,
            "region_id": region_id,
        }

    return edge2region, (minx, miny, maxx, maxy)


def add_region_to_trips(df, edge2region, m, n, out_csv):
    """给轨迹 csv 增加 start/end 的 region 信息并输出。"""

    def map_edge(edge_id, key):
        info = edge2region.get(edge_id)
        if info is None:
            return pd.NA
        return info[key]

    df["start_region_id"] = df["start_edge"].apply(
        lambda x: map_edge(x, "region_id")
    )
    df["end_region_id"] = df["end_edge"].apply(
        lambda x: map_edge(x, "region_id")
    )

    df.to_csv(out_csv, index=False)
    print(f"保存结果到: {out_csv}")

def convert_traj_file_df(input_path: str) -> pd.DataFrame:
    """
    读取原始 CSV，返回 DataFrame：
    driver, traj_id, start_time, start_edge, end_edge
    """

    data = {
        "driver": [],
        "traj_id": [],
        "start_time": [],
        "start_edge": [],
        "end_edge": [],
    }

    with open(input_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in)

        for row in reader:
            if not row:
                continue

            if len(row) < 4:
                continue

            driver = row[0].strip()
            traj_id = row[1].strip()
            point_seq_str = row[3].strip()

            if not point_seq_str:
                continue

            # 解析点序列
            try:
                points = ast.literal_eval(point_seq_str)
            except Exception:
                continue

            if not points:
                continue

            start_point = points[0]
            end_point = points[-1]

            # 解包： (edge_id/lon?, lat?, timestamp?, speed)
            try:
                start_edge, start_time, _, _ = start_point
                end_edge, end_time, _, _ = end_point
            except ValueError:
                continue

            data["driver"].append(driver)
            data["traj_id"].append(traj_id)
            data["start_time"].append(start_time)
            data["start_edge"].append(start_edge)
            data["end_edge"].append(end_edge)

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    df["start_time_sf"] = df["start_time"].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=sf_tz).strftime("%Y-%m-%d %H:%M:%S")
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="从原始轨迹 CSV 中提取起点时间和起终点经纬度（每条轨迹一行）。"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="原始轨迹 CSV 文件路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--shp",
        "-s",
        required=True,
        help="路网 shapefile 文件路径",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=20,
        help="将区域划分为 m 行",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="将区域划分为 n 列",
    )
    parser.add_argument(
        "--fid_col",
        type=str,
        default="fid",
        help="shapefile 中表示 edge id 的列名（默认 'fid'）",
    )

    args = parser.parse_args()

    df = convert_traj_file_df(args.input)

    edge2region, bbox = build_edge_region_map(
        args.shp, args.m, args.n, fid_col=args.fid_col
    )

    add_region_to_trips(df, edge2region, args.m, args.n, out_csv=args.output)


if __name__ == "__main__":
    main()
