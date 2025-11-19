import pandas as pd
import numpy as np
import os
import ast
from tqdm import tqdm

def convert_traj2flow(traj_file, N, idx2seg=None, bin_minutes: int = 10):
    """
    将轨迹文件按给定时间粒度统计为流量张量。

    参数
    ----
    traj_file : str
        轨迹 csv 路径
    N : int
        路段总数（如果没有 idx2seg，就认为 segment_id 为 0..N-1）
    idx2seg : list 或 array, 可选
        长度为 N 的路段 id 列表，用于固定列顺序
    bin_minutes : int, 默认 10
        时间桶长度（分钟），例如 5/10/15/30 等
    """
    records = []
    traj_df = pd.read_csv(traj_file, header=None)
    traj_df.columns = ['driver', 'traj_id', 'start_end', 'traj']

    # pandas 的 freq/floor 需要类似 '10min' 这种字符串
    freq_str = f"{bin_minutes}min"

    for raw_traj in tqdm(traj_df['traj'], total=len(traj_df), desc='parse trajs'):
        try:
            traj_list = ast.literal_eval(raw_traj)
        except Exception:
            continue
        if not traj_list:
            continue

        seg_ids = [seg[0] for seg in traj_list]
        ts_vals = [seg[1] for seg in traj_list]

        ts_numeric = pd.to_numeric(pd.Series(ts_vals), errors='coerce')
        if ts_numeric.notna().any():
            ts_arr = ts_numeric.values.astype('float64')
            # 粗略判断是否是毫秒级时间戳（> 1e12）
            if (ts_arr > 1e12).sum() > 0:
                ts_arr = ts_arr / 1000.0
            try:
                dt_index = pd.to_datetime(ts_arr.astype('int64'), unit='s', errors='coerce')
                dt_floored = dt_index.floor(freq_str)
            except Exception:
                dt_index = pd.to_datetime(pd.Series(ts_vals).astype(str), errors='coerce')
                dt_floored = dt_index.dt.floor(freq_str)
        else:
            dt_index = pd.to_datetime(pd.Series(ts_vals).astype(str), errors='coerce')
            dt_floored = dt_index.dt.floor(freq_str)

        for sid, dt_val in zip(seg_ids, dt_floored):
            if pd.isna(dt_val):
                continue
            records.append((sid, dt_val))

    stat_df = pd.DataFrame(records, columns=['segment_id', 'time_bin'])
    if stat_df.empty:
        raise ValueError("No valid records parsed from traj_file.")

    result = stat_df.groupby(['segment_id', 'time_bin']).size().reset_index(name='car_count')

    if idx2seg is not None:
        all_segments = list(idx2seg)
    else:
        all_segments = list(np.arange(0, N))

    # 生成完整时间索引，步长为 bin_minutes
    full_time_index = pd.date_range(
        stat_df['time_bin'].min(),
        stat_df['time_bin'].max(),
        freq=freq_str
    )

    pivot = result.pivot(index='time_bin', columns='segment_id', values='car_count')
    pivot = pivot.reindex(index=full_time_index, columns=all_segments, fill_value=0)
    pivot = pivot.fillna(0)
    pivot_matrix = pivot[all_segments]

    pivot_array = pivot_matrix.to_numpy(dtype=np.int32)
    # 增加最后一维 C=1
    pivot_array = pivot_array[..., np.newaxis]

    out_path = os.path.join(os.path.dirname(traj_file), f'flow_{bin_minutes}min.npy')
    np.save(out_path, pivot_array)
    return pivot_array
