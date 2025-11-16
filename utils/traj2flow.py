import pandas as pd
import numpy as np
import os
import ast
from tqdm import tqdm

def convert_traj2flow(traj_file, N, idx2seg=None):
    records = []
    traj_df = pd.read_csv(traj_file, header=None)
    traj_df.columns = ['driver', 'traj_id', 'start_end', 'traj']

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
            if (ts_arr > 1e12).sum() > 0:
                ts_arr = ts_arr / 1000.0
            try:
                dt_index = pd.to_datetime(ts_arr.astype('int64'), unit='s', errors='coerce')
                dt_floored = dt_index.floor('10min')
            except Exception:
                dt_index = pd.to_datetime(pd.Series(ts_vals).astype(str), errors='coerce')
                dt_floored = dt_index.dt.floor('10min')
        else:
            dt_index = pd.to_datetime(pd.Series(ts_vals).astype(str), errors='coerce')
            dt_floored = dt_index.dt.floor('10min')

        for sid, dt_val in zip(seg_ids, dt_floored):
            if pd.isna(dt_val):
                continue
            records.append((sid, dt_val))

    stat_df = pd.DataFrame(records, columns=['segment_id', 'time_bin'])
    result = stat_df.groupby(['segment_id', 'time_bin']).size().reset_index(name='car_count')

    if idx2seg is not None:
        all_segments = list(idx2seg)
    else:
        all_segments = list(np.arange(0, N))

    full_time_index = pd.date_range(stat_df['time_bin'].min(), stat_df['time_bin'].max(), freq='10min')

    pivot = result.pivot(index='time_bin', columns='segment_id', values='car_count')
    pivot = pivot.reindex(index=full_time_index, columns=all_segments, fill_value=0)
    pivot = pivot.fillna(0)
    pivot_matrix = pivot[all_segments]

    pivot_array = pivot_matrix.to_numpy(dtype=np.int32)
    pivot_array = pivot_array[..., np.newaxis]
    np.save(os.path.join(os.path.dirname(traj_file), 'flow.npy'), pivot_array)
    return pivot_array
