#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Porto taxi CSV → 起终点经纬度 → 栅格 region_id（Porto shapefile bbox）
并带有轻量级进度条（不依赖 tqdm）
"""

import csv
import ast
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import geopandas as gpd
import pandas as pd
import os
import sys

PORTO_TZ = ZoneInfo("Europe/Lisbon")


# ---------------- Helper：轻量级进度条 ---------------- #

def print_progress(prefix, current, total, bar_len=40):
    """
    简易进度条，不依赖 tqdm
    """
    if total <= 0:
        return
    rate = current / total
    filled = int(bar_len * rate)
    bar = "=" * filled + "." * (bar_len - filled)
    percent = int(rate * 100)
    sys.stdout.write(f"\r{prefix} [{bar}] {percent}% ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


# ---------------- 区域划分 ---------------- #

def build_region_bbox(shp_path: str):
    gdf = gpd.read_file(shp_path)
    minx, miny, maxx, maxy = gdf.total_bounds
    if minx == maxx or miny == maxy:
        raise ValueError("shapefile bbox 无效")
    return minx, miny, maxx, maxy


def lonlat_to_region(lon, lat, bbox, m, n):
    minx, miny, maxx, maxy = bbox
    dx = (maxx - minx) / n
    dy = (maxy - miny) / m

    col = int((lon - minx) / dx)
    row = int((lat - miny) / dy)

    # 边界处理
    if col == n:
        col = n - 1
    if row == m:
        row = m - 1

    col = max(0, min(col, n - 1))
    row = max(0, min(row, m - 1))

    region_id = row * n + col
    return row, col, region_id


# ---------------- CSV 转 DataFrame ---------------- #

def count_lines(file_path):
    """
    统计总行数用于进度条
    """
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f)


def convert_traj_file_df(input_path: str) -> pd.DataFrame:
    """
    读取 Porto CSV，提取：
    - trip_id
    - taxi_id
    - start_time / local
    - start_lon, start_lat
    - end_lon, end_lat
    并带进度条
    """
    total_lines = count_lines(input_path)
    processed = 0

    data = {
        "trip_id": [],
        "taxi_id": [],
        "start_time": [],
        "start_time_local": [],
        "start_lon": [],
        "start_lat": [],
        "end_lon": [],
        "end_lat": [],
    }

    with open(input_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)

        for row in reader:
            processed += 1
            if processed % 10000 == 0:
                print_progress("解析 CSV 中", processed, total_lines)

            poly_str = (row.get("POLYLINE") or "").strip()
            if not poly_str or poly_str == "[]":
                continue

            try:
                points = ast.literal_eval(poly_str)
            except Exception:
                continue

            if not points:
                continue

            try:
                start_lon, start_lat = points[0]
                end_lon, end_lat = points[-1]
            except Exception:
                continue

            trip_id = (row.get("TRIP_ID") or "").strip()
            taxi_id = (row.get("TAXI_ID") or "").strip()
            ts_str = (row.get("TIMESTAMP") or "").strip()

            try:
                start_ts = int(float(ts_str))
            except Exception:
                continue

            dt_local = datetime.fromtimestamp(start_ts, tz=PORTO_TZ)
            dt_local_str = dt_local.strftime("%Y-%m-%d %H:%M:%S")

            data["trip_id"].append(trip_id)
            data["taxi_id"].append(taxi_id)
            data["start_time"].append(start_ts)
            data["start_time_local"].append(dt_local_str)
            data["start_lon"].append(start_lon)
            data["start_lat"].append(start_lat)
            data["end_lon"].append(end_lon)
            data["end_lat"].append(end_lat)

    print_progress("解析 CSV 中", total_lines, total_lines)
    print("CSV 解析完成.")

    return pd.DataFrame(data)


# ---------------- 添加 region_id & 输出 ---------------- #

def add_region_to_trips(df, bbox, m, n, out_csv):
    total = len(df)
    processed = 0

    start_region = []
    end_region = []

    for idx, row in df.iterrows():
        processed += 1
        if processed % 10000 == 0:
            print_progress("映射区域中", processed, total)

        _, _, r1 = lonlat_to_region(row["start_lon"], row["start_lat"], bbox, m, n)
        _, _, r2 = lonlat_to_region(row["end_lon"], row["end_lat"], bbox, m, n)

        start_region.append(r1)
        end_region.append(r2)

    print_progress("映射区域中", total, total)
    print("区域映射完成。")

    df["start_region_id"] = start_region
    df["end_region_id"] = end_region

    df.to_csv(out_csv, index=False)
    print(f"已保存输出: {out_csv}")


# ---------------- main ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Porto taxi 轨迹转 region_id（带简易进度条，无 tqdm）"
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--shp", "-s", required=True)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    df = convert_traj_file_df(args.input)
    bbox = build_region_bbox(args.shp)
    add_region_to_trips(df, bbox, args.m, args.n, args.output)


if __name__ == "__main__":
    main()
