#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将匹配好的 Porto 轨迹 pickle 转换为 DRPK 风格的 CSV。

输入 pickle 中每条记录格式为：
    (traj_id, [edge_id1, edge_id2, ...], [t1, t2, ...])

其中 t_k 为进入第 k 条边的时间戳（单位：秒）。

输出 CSV 中每条记录格式为：
    driver, traj_id, offsets, segment_sequence

- driver：直接用 traj_id 填充
- traj_id：沿用输入中的 traj_id
- offsets：不关心，但需占位，统一写成空元组 "()"
- segment_sequence：Python 风格的列表字符串：
    [
        [segment_id, enter_time, avg_speed, duration],
        ...
    ]

  这里：
    - segment_id = edge_id
    - enter_time = 进入该边的时间戳（来自输入）
    - avg_speed  = None  （无实际速度信息，占位用）
    - duration   = 下一条边时间戳 - 当前时间戳（最后一条边用上一段的时长，
                   若上一段不存在则为 0.0）
"""

import csv
import argparse
import pickle
from typing import Any, Iterable, List, Sequence, Tuple


def _build_segment_sequence(
    edges: Sequence[int],
    times: Sequence[float],
) -> List[List[Any]]:
    """
    根据 edge 序列和时间序列构造 segment_sequence 列表。

    每个元素为：
        [segment_id, enter_time, avg_speed(None), duration]
    """
    if not edges or not times or len(edges) != len(times):
        return []

    segments: List[List[Any]] = []

    prev_duration: float = 0.0
    n = len(edges)
    for i, edge_id in enumerate(edges):
        enter_time = float(times[i])

        if i < n - 1:
            # 与下一条边的时间差作为当前边的 duration
            duration = float(times[i + 1]) - float(times[i])
            if duration < 0:
                # 极端情况下时间不递增，至少保证非负
                duration = 0.0
            prev_duration = duration
        else:
            # 最后一条边：使用上一条 duration，如果没有，则为 0.0
            duration = prev_duration

        seg_item = [
            int(edge_id),   # segment_id
            enter_time,     # enter_time
            None,           # avg_speed 占位
            duration,       # duration
        ]
        segments.append(seg_item)

    return segments


def convert_traj_pkl_to_csv(input_pkl: str, output_csv: str) -> None:
    """
    读取匹配好的 Porto 轨迹 pickle，生成 DRPK 风格的 CSV。

    输入：
        input_pkl:  pickle 文件路径
        output_csv: 输出 CSV 文件路径
    """
    # 载入 pickle：应为列表，每个元素为 (traj_id, edges, times)
    with open(input_pkl, "rb") as f_in:
        trips: Iterable[Tuple[Any, Sequence[int], Sequence[float]]] = pickle.load(f_in)

    with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)

        # 输出表头
        writer.writerow([
            "driver",
            "traj_id",
            "offsets",
            "segment_sequence",
        ])

        for trip_idx, trip in enumerate(trips):
            if not trip or len(trip) != 3:
                continue

            traj_id, edges, times = trip

            # 基础检查
            if not edges or not times or len(edges) != len(times):
                print(f"跳过无效轨迹 trip_idx={trip_idx}, traj_id={traj_id}")
                continue

            # 要求 3：driver_id 直接用 traj_id 填充
            driver = str(traj_id)
            traj_id_str = str(traj_id)

            # 要求 1：offset 不关心，用空元组占位
            offsets_str = "()"  # Python 风格的空元组字符串

            # 构造 segment_sequence
            segments = _build_segment_sequence(edges, times)
            if not segments:
                continue

            # 用 repr 输出成 Python 风格列表字符串，方便后续 ast.literal_eval 解析
            segment_seq_str = repr(segments)

            writer.writerow([
                driver,
                traj_id_str,
                offsets_str,
                segment_seq_str,
            ])


def main():
    parser = argparse.ArgumentParser(
        description="将匹配好的 Porto 轨迹 pickle 转换为 DRPK 风格 CSV（含空 offsets 占位，无 points list）。"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入 pickle 文件路径（形如 preprocessed_porto_trips_all_clean.pkl）",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="输出 CSV 文件路径",
    )
    args = parser.parse_args()

    convert_traj_pkl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
