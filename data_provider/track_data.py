# dataset/segment_road_dataset.py
"""
SegmentRoadDataset

严格把 "node = road segment"（segment-level graph）。
适用于你的轨迹文件已经是基于 road_id（每项为路段 id）的情况。

主要功能：
- 构建 segment2idx / idx2segment vocabulary
- 从轨迹（或 edge_list）构造 segment->segment 全局边 full_edge_index
- 若轨迹带时间戳，估计相邻 segment 的 travel time 并生成 P_time (time-bin x E_full)
- 构造 K-minute 可达子图 edge_index_kmin（若没有 travel_time 则退回 k-hop 或 distance-based 近似）
- 返回 item（可被训练代码直接使用）:
    {
      'full_edge_index': 2 x E_full LongTensor,
      'P_edge_full': E_full FloatTensor (for the time bin),
      'edge_index': 2 x E_k (kmin) LongTensor,
      'static': N_seg x C_static FloatTensor,
      'S_hist': T_hist x N_seg x C_state FloatTensor,
      'weekly_idx': LongTensor(T_hist) or None,
      'daily_idx': LongTensor(T_hist) or None,
      'trajs': list of (nodes_list, bins_list) # nodes_list are segment idx
      'time_idx': int
    }

注意：
- 此代码假设你的轨迹文件中每条访问记录的 "node" 字段就是 road_id（即 segment id）。
- 如果你的 traffic_timeseries 已经按 road_id 排序并与你的 vocab 对齐，可直接作为 input；否则需要预先把 traffic 按 segment2idx reorder。
"""

import ast
import os
import sys
import numpy as np
import pandas as pd
from shapely import Point
import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm
import geopandas as gpd
import re

try:
    import networkx as nx
except Exception:
    nx = None

# -------------------- 辅助函数 --------------------
def build_segment_vocab_from_trajs_and_edge(raw_map, roadid_col=None, out_vocab_file=None):
    """
    从 raw_map文件构造 segment vocabulary（road_id -> idx）。
    输入格式期望：
      - raw_map: 每行 "fid, length" 的 csv 或 shapefile（shapefile 中 fid 列为 road_id）
    实际上只要能从raw_map 文件里抽出所有 road_id 即可。
    返回: segment2idx (dict), idx2segment (list)
    """
    segs = set()
    if os.path.exists(raw_map):
        cols_to_keep = [roadid_col,"length"]  
        cols_to_copy = [roadid_col]  # 只保留的列名称
        new_col_name = "weight"    # 新增列名称
        speed = 583                # 速度，单位：m/min 约等于 35km/h
        gdf = gpd.read_file(raw_map)
        # 检查并只保留指定列 + geometry
        missing_cols = [c for c in cols_to_keep if c not in gdf.columns]
        if missing_cols:
            raise ValueError(f"以下字段在源文件中不存在: {missing_cols}")

        new_gdf = gdf[cols_to_copy].copy()
        new_gdf.rename(columns={roadid_col:"road_id"}, inplace=True)

        original_length = gdf["length"]
        # 计算时间（分钟）
        travel_time = original_length / speed

        # 添加新字段并赋值
        new_gdf[new_col_name] = travel_time
        for rid in new_gdf["road_id"]:
            try:
                segs.add(int(rid))
            except:
                continue
    else:
        raise ValueError(f"raw_map file {raw_map} not found")

    segs = sorted(list(segs))
    seg2idx = {int(s): i for i, s in enumerate(segs)}
    idx2seg = segs
    if out_vocab_file:
        with open(out_vocab_file, 'wb') as fh:
            pickle.dump({'seg2idx': seg2idx, 'idx2seg': idx2seg}, fh)
    return seg2idx, idx2seg

def build_static_features_for_segments(idx2seg, raw_map, feature_cols, static_file, roadid_col=None):
    """
    从 raw_map 文件中提取静态特征（按 segment id 顺序排列）。
    输入格式期望：
      - raw_map: 每行 "fid, length, feature1, feature2, ..." 的 csv 或 shapefile
      - feature_cols: 需要提取的列名列表（必须在 raw_map 中存在）
    返回: static_features (N_seg x C_static) np.ndarray
    """
    segs = set()
    if os.path.exists(raw_map):
        cols_to_keep = [roadid_col] + feature_cols  
        gdf = gpd.read_file(raw_map)
        # 检查并只保留指定列 + geometry
        missing_cols = [c for c in cols_to_keep if c not in gdf.columns]
        if missing_cols:
            raise ValueError(f"以下字段在源文件中不存在: {missing_cols}")

        new_gdf = gdf[cols_to_keep].copy()
        new_gdf.rename(columns={roadid_col:"road_id"}, inplace=True)

        # 先对 feature_cols 做批量数值化处理：
        # 1) 尝试 pd.to_numeric(errors='coerce')
        # 2) 对仍为 NaN 的字符串尝试 ast.literal_eval（若为 list/tuple 则取最大数值），
        #    或用正则从字符串中抽取数字
        # 3) 用列均值填充剩下的 NaN
        proc_df = new_gdf[feature_cols].copy()

        def try_extract_numeric_from_str(s):
            # None/NaN
            if s is None:
                return None
            # 如果已经是数字类型
            if isinstance(s, (int, float, np.number)):
                try:
                    return float(s)
                except Exception:
                    return None
            # 尝试 literal_eval（处理 ['2','3'] 之类）
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    nums = []
                    for x in val:
                        try:
                            nums.append(float(x))
                        except Exception:
                            continue
                    if len(nums) > 0:
                        return float(max(nums))
                if isinstance(val, (int, float, np.number)):
                    return float(val)
            except Exception:
                pass
            # 最后尝试正则抽取数字
            try:
                parts = re.findall(r"[-+]?\d*\.\d+|\d+", str(s))
                if parts:
                    # 取最后一个数字（通常代表最大或关键词处的数）
                    return float(parts[-1])
            except Exception:
                pass
            return None

        for col in feature_cols:
            # first pass: numeric coercion
            coerced = pd.to_numeric(proc_df[col], errors='coerce')
            proc_df[col] = coerced
            # second pass: try parse remaining non-numeric strings
            na_mask = proc_df[col].isna() & new_gdf[col].notna()
            if na_mask.any():
                # apply parser only on the problematic entries
                parsed_vals = proc_df.loc[na_mask, col].copy()
                for idx_row in proc_df.loc[na_mask].index:
                    raw = new_gdf.at[idx_row, col]
                    parsed = try_extract_numeric_from_str(raw)
                    if parsed is not None:
                        proc_df.at[idx_row, col] = parsed
            # fill remaining NaN with column mean (or 0 if mean is NaN)
            col_mean = proc_df[col].dropna().astype(float).mean()
            if np.isnan(col_mean):
                col_mean = 0.0
            proc_df[col] = proc_df[col].fillna(col_mean).astype(np.float32)

        # 构建 road_id -> features 映射
        roadid_to_features = {}
        for i, row in new_gdf.iterrows():
            rid = row["road_id"]
            try:
                rid_key = int(rid)
            except Exception:
                rid_key = rid
            feats = proc_df.loc[i, feature_cols].to_numpy(dtype=np.float32)
            roadid_to_features[rid_key] = feats
    else:
        raise ValueError(f"raw_map file {raw_map} not found")

    N = len(idx2seg)
    C = len(feature_cols)
    static_features = np.zeros((N, C), dtype=np.float32)
    for i, seg in enumerate(idx2seg):
        if seg in roadid_to_features:
            static_features[i] = roadid_to_features[seg]
        else:
            # 若某些 segment 在 raw_map 中没有对应的特征，则保持为零向量
            pass
    np.save(static_file, static_features)
    return static_features

def convert_traj2flow(traj_file, N, idx2seg=None):
    records = []
    traj_df = pd.read_csv(traj_file, header=None)
    traj_df.columns = ['driver','traj_id','start_end', 'traj']

    # 先安全解析所有轨迹列（使用 ast.literal_eval），并对每条轨迹内部的时间戳进行向量化转换
    for raw_traj in tqdm(traj_df['traj'], total=len(traj_df), desc='parse trajs'):
        try:
            traj_list = ast.literal_eval(raw_traj)
        except Exception:
            # 无法解析则跳过该轨迹
            continue
        if not traj_list:
            continue

        # 提取 segment id 和时间戳列表
        seg_ids = [seg[0] for seg in traj_list]
        ts_vals = [seg[1] for seg in traj_list]

        # 尝试数值化（秒或毫秒）
        ts_numeric = pd.to_numeric(pd.Series(ts_vals), errors='coerce')
        if ts_numeric.notna().any():
            # 对能解析为数值的项，统一处理；对 NaN 项在后续回退为字符串解析
            ts_arr = ts_numeric.values.astype('float64')
            # 判断是否存在毫秒数值（非常大），若存在则将其缩放到秒
            if (ts_arr > 1e12).sum() > 0:
                ts_arr = ts_arr / 1000.0
            try:
                dt_index = pd.to_datetime(ts_arr.astype('int64'), unit='s', errors='coerce')
                dt_floored = dt_index.floor('10min')
            except Exception:
                dt_index = pd.to_datetime(pd.Series(ts_vals).astype(str), errors='coerce')
                dt_floored = dt_index.dt.floor('10min')
        else:
            # 全部不能解析为数值，回退为字符串解析
            dt_index = pd.to_datetime(pd.Series(ts_vals).astype(str), errors='coerce')
            dt_floored = dt_index.dt.floor('10min')

        # 批量加入 records，跳过无法解析的时间点
        for sid, dt_val in zip(seg_ids, dt_floored):
            if pd.isna(dt_val):
                continue
            records.append((sid, dt_val))

    stat_df = pd.DataFrame(records, columns=['segment_id', 'time_bin'])

    # 统计每条segment每10分钟的车辆数
    result = stat_df.groupby(['segment_id', 'time_bin']).size().reset_index(name='car_count')

    # 生成完整的时间索引和所有 segment（保持 time_bin 为 index，不把它变成列）
    # 如果提供了 idx2seg，则使用 idx2seg 的顺序保证导出的列顺序与 vocabulary 对齐
    if idx2seg is not None:
        all_segments = list(idx2seg)
    else:
        all_segments = list(np.arange(0, N))  # sf_100 segment_id从0到26658，若不想全部输出，则可以不填self.N

    full_time_index = pd.date_range(stat_df['time_bin'].min(), stat_df['time_bin'].max(), freq='10min')

    # 透视表并补全缺失（不要 reset_index，这样 time_bin 保持为 index）
    pivot = result.pivot(index='time_bin', columns='segment_id', values='car_count')
    # 使用 all_segments 做 reindex，保证列完整且按 idx2seg 顺序排列（若 idx2seg 提供）
    pivot = pivot.reindex(index=full_time_index, columns=all_segments, fill_value=0)
    pivot = pivot.fillna(0)

    # 直接按 all_segments 的顺序取列，保证与 idx2seg 对齐
    pivot_matrix = pivot[all_segments]

    # 将 pivot_matrix 转为 numpy（整型）并在末尾增加一个长度为1的维度，便于后续与模型期望的形状对齐
    # 结果形状: (T, N, 1)
    # 注意：这里我们显式使用整型（int32），因为 car_count 是计数值
    pivot_array = pivot_matrix.to_numpy(dtype=np.int32)
    pivot_array = pivot_array[..., np.newaxis]
    np.save(os.path.join(os.path.dirname(traj_file), 'flow.npy'), pivot_array)
    return pivot_array

def parse_points_from_field(field_value: str):
    """
    把类似字符串解析为 [(road_id, timestamp(int), lon, lat), ...]
    使用 ast.literal_eval 做安全解析。
    """
    if pd.isna(field_value):
        return []
    s = field_value
    # 若字段被额外双引号包裹（例如 CSV 导出时），去掉外层双引号再试
    if isinstance(s, str):
        s_strip = s.strip()
        if s_strip.startswith('"') and s_strip.endswith('"'):
            s = s_strip[1:-1]
    try:
        pts = ast.literal_eval(s)
    except Exception as e:
        # 抛出具体错误以便定位问题
        raise ValueError(f"无法解析轨迹点字符串: {e} -- 原始值片段: {str(s)[:200]!r}")
    norm = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        road_id = p[0]
        # timestamp 在示例中是第二个元素
        try:
            timestamp = int(float(p[1]))
        except Exception:
            # 不能解析 timestamp 时跳过该点
            continue
        # 有些情况下经纬出现在 p[2],p[3]，但我们这里不需要 lon/lat
        norm.append((road_id, timestamp))
    return norm

def build_full_edges_from_shp_using_vocab(shp_path,
                                         seg2idx,
                                         roadid_col='fid',
                                         from_col='u',
                                         to_col='v',
                                         geom_col='geometry',
                                         assume_uv_present=True,
                                         coord_round=6,
                                         verbose=True,
                                         skip_missing=True):
    """
    使用已有 seg2idx (roadid -> seg_idx) 从 shapefile 构建 full_edge_index（segment-level adjacency）。
    参数:
      - shp_path: path to edge.shp
      - seg2idx: dict {roadid -> seg_idx} 已存在的 vocabulary
      - roadid_col: shapefile 中表示 road id 的列名
      - from_col, to_col: 可选列（若存在并且 assume_uv_present=True 则优先使用）
      - geom_col: geometry 列名（通常 'geometry'）
      - assume_uv_present: 如果 True 且 from_col/to_col 存在，则会用这些 node id 辅助判断相邻关系
      - coord_round: 若使用 geometry 推断端点，则坐标四舍五入精度
      - skip_missing: 如果 shp 中出现的 roadid 不在 seg2idx 中，True -> 跳过并记录, False -> 抛错
    返回:
      - full_edge_index: torch.LongTensor (2 x E) (src_seg_idx, dst_seg_idx)
      - info: dict 包含统计信息（num_features, missing_roadids, num_edges）
    """
    gdf = gpd.read_file(shp_path)
    if verbose:
        print(f"[shp->full_edges] load {len(gdf)} features from {shp_path}")

    # 检查字段
    if roadid_col not in gdf.columns:
        raise ValueError(f"roadid_col '{roadid_col}' not found in shapefile columns: {gdf.columns.tolist()}")

    # 遍历 shapefile，收集每条 segment 的 (seg_idx, start_node_key, end_node_key)
    # 如果 shapefile 包含 explicit from_col/to_col 使用它们（但仍以 seg2idx 的 roadid 映射为主）
    seg_to_endpoints = {}   # seg_idx -> (start_key, end_key)
    missing_roadids = set()
    coord_map = {}  # rounded coord -> node_key  (only used if using geometry)
    next_coord_node = 0

    use_uv = assume_uv_present and (from_col in gdf.columns) and (to_col in gdf.columns)
    if use_uv and verbose:
        print("[shp->full_edges] using explicit from/to columns for endpoints")

    for i, row in gdf.iterrows():
        raw_rid = row[roadid_col]
        # try normalize numeric if stored as float/int-like strings
        try:
            rid_key = int(raw_rid)
        except Exception:
            rid_key = raw_rid

        if rid_key not in seg2idx:
            missing_roadids.add(rid_key)
            if skip_missing:
                continue
            else:
                raise KeyError(f"roadid {rid_key} from shapefile not found in provided seg2idx vocab")

        seg_idx = seg2idx[rid_key]

        if use_uv:
            u_raw = row[from_col]; v_raw = row[to_col]
            # normalize if numeric-like
            try:
                u_key = int(u_raw)
            except:
                u_key = u_raw
            try:
                v_key = int(v_raw)
            except:
                v_key = v_raw
            seg_to_endpoints[seg_idx] = (("nid", u_key), ("nid", v_key))
        else:
            geom = row[geom_col]
            if geom is None:
                # skip or raise
                if verbose:
                    print(f"[shp->full_edges] warning: seg {rid_key} has no geometry, skipping")
                continue
            # handle LineString or MultiLineString
            try:
                coords = list(geom.coords)
            except Exception:
                # MultiLineString -> take first part
                if geom.geom_type == 'MultiLineString':
                    coords = list(list(geom)[0].coords)
                else:
                    raise
            start = coords[0]; end = coords[-1]
            # round coords to reduce tiny differences
            start_k = (round(float(start[0]), coord_round), round(float(start[1]), coord_round))
            end_k = (round(float(end[0]), coord_round), round(float(end[1]), coord_round))
            # map to synthetic node keys (stringify to avoid colliding with numeric node ids)
            if start_k not in coord_map:
                coord_map[start_k] = f"c{next_coord_node}"; next_coord_node += 1
            if end_k not in coord_map:
                coord_map[end_k] = f"c{next_coord_node}"; next_coord_node += 1
            seg_to_endpoints[seg_idx] = (( "coord", coord_map[start_k] ), ( "coord", coord_map[end_k] ))

    if verbose:
        print(f"[shp->full_edges] found {len(seg_to_endpoints)} segments matched to vocab; {len(missing_roadids)} missing roadids")

    # Build reverse index: start_key -> list of seg_idx, end_key -> seg_idx
    start_to_segs = defaultdict(list)
    end_of_seg = {}
    for seg_idx, (start_key, end_key) in seg_to_endpoints.items():
        start_to_segs[start_key].append(seg_idx)
        end_of_seg[seg_idx] = end_key

    # For each segment A, find B s.t. end(A) == start(B)
    src_list = []
    dst_list = []
    for A, end_key in end_of_seg.items():
        # find segments whose start_key equals this end_key
        b_list = start_to_segs.get(end_key, [])
        for B in b_list:
            src_list.append(A)
            dst_list.append(B)

    if len(src_list) == 0:
        fe = np.zeros((2,0), dtype=np.int64)
    else:
        fe = np.stack([np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)], axis=0)

    info = {
        'num_features_loaded': len(gdf),
        'num_segments_matched': len(seg_to_endpoints),
        'num_missing_roadids': len(missing_roadids),
        'missing_roadids': list(missing_roadids),
        'num_edges': fe.shape[1]
    }
    if verbose:
        print(f"[shp->full_edges] built full_edge_index with {fe.shape[1]} edges")

    return torch.LongTensor(fe), info

def compute_P_time_from_transitions(full_edge_index, transitions_dict, num_time_bins):
    """
    从 transitions_dict {(i,j): [dt_samples]} 估计 per-edge log-prob P_time（按时间 bin）
    简化实现：只统计转移频次以得到 P(edge|src) 的概率（按时间 bin 暂用同一分布，若没有时间维度只返回 single）
    因为 transitions_dict 包含 dt（时间差）并非发生时刻；如果轨迹带有发生时刻并按 bin 统计更好。
    这里我们做较简单的频次统计：P_time[tb, e] = log( count_tb(src->dst)/sum_dst count_tb(src->*) )
    但要做到 "time-dependent"，我们需要轨迹中带发生时刻并按时间 bin 统计源-目的对的出现次数。
    若轨迹中没有明确发生时刻信息（或不做 bin），返回 single-row P (1 x E_full)。
    """
    # simple fallback: if transitions_dict provided but no per-bin info, build a single time bin
    E = full_edge_index.shape[1]
    # build mapping from pair -> idx
    pair_to_idx = {(int(full_edge_index[0,i]), int(full_edge_index[1,i])): i for i in range(E)}
    # counts per src
    src_counts = defaultdict(float)
    pair_count_arr = np.zeros((E,), dtype=np.float32)
    for (i,j), samples in transitions_dict.items():
        if (i,j) in pair_to_idx:
            idx = pair_to_idx[(i,j)]
            pair_count_arr[idx] = float(len(samples))
            src_counts[i] += float(len(samples))
    # normalize per src
    probs = np.zeros_like(pair_count_arr, dtype=np.float32)
    for e_idx in range(E):
        s = int(full_edge_index[0, e_idx])
        denom = src_counts.get(s, 1.0)
        probs[e_idx] = pair_count_arr[e_idx] / (denom + 1e-12)
    logp = np.log(probs + 1e-9)
    P_time = logp.reshape(1, E)  # 1 x E (no temporal bins)
    # Note: if you have time-of-day info per transition, you can make it num_time_bins x E
    return P_time

def compute_travel_time_matrix_from_transitions(full_edge_index, transitions_dict, N):
    """
    基于 transitions_dict 中的 dt 样本估计边权（单条边权），并用 Dijkstra 得到 shortest-path travel time matrix。
    如果 transitions_dict 没有覆盖某些直接边，边权设为 large。
    返回 T (N x N) matrix with estimated shortest travel-time (minutes), np.inf if unreachable.
    """
    if nx is None:
        # fallback: build trivial matrix where direct transitions time = median(dt) if exists, else inf
        T = np.full((N, N), np.inf, dtype=np.float32)
        for (i,j), samples in transitions_dict.items():
            T[int(i), int(j)] = float(np.median(samples))
        return T
    # build graph
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    # assign weight for edges present
    for (i,j), samples in transitions_dict.items():
        median_t = float(np.median(samples))
        G.add_edge(int(i), int(j), time=median_t)
    T = np.full((N, N), np.inf, dtype=np.float32)
    for i in range(N):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='time')
        for j, tt in lengths.items():
            T[i, int(j)] = tt
    return T

def build_kmin_reachable_edges_from_travel_time_matrix(travel_time_matrix, K_min):
    tt = np.array(travel_time_matrix)
    N = tt.shape[0]
    src_list = []
    dst_list = []
    for i in range(N):
        row = tt[i]
        mask = (row > 0) & (row <= K_min) & np.isfinite(row)
        idxs = np.nonzero(mask)[0]
        if idxs.size:
            src_list.extend([i]*idxs.size)
            dst_list.extend(idxs.tolist())
    if len(src_list) == 0:
        return np.zeros((2,0), dtype=np.int64)
    return np.stack([np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)], axis=0)

def build_khop_edges_from_full_edges(full_edge_index, k):
    """
    基于 full_edge_index（2 x E_full，src,dst）构造 k-hop reachability edges。
    返回 edge_index_khop (2 x E_k)
    """
    src = full_edge_index[0]; dst = full_edge_index[1]
    N = int(max(src.max(), dst.max()) + 1)
    # build adjacency
    adj = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        adj[int(u)].append(int(v))
    srcs, dsts = [], []
    from collections import deque
    for node in range(N):
        seen = set([node])
        q = deque([(node, 0)])
        while q:
            cur, dist = q.popleft()
            if dist >= 1:
                srcs.append(node)
                dsts.append(cur)
            if dist == k:
                continue
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, dist+1))
    if len(srcs) == 0:
        return np.zeros((2,0), dtype=np.int64)
    return np.stack([np.array(srcs, dtype=np.int64), np.array(dsts, dtype=np.int64)], axis=0)

# ------------------ Dataset 类 ------------------
class SegmentRoadDataset(Dataset):
    """
    Dataset where each node is a road segment (road_id).
    初始化参数:
      data_root: 数据目录
      edge_list_file: optional, 边表（若有）
      traj_file: 必需，轨迹文件 (traj_id, timestamp, road_id)
      static_file: optional, static features per segment (N_seg x C_static) (按 segment id 顺序或后续映射)
      traffic_ts_file: optional, T_total x N_seg x C_state (若无可用在训练时填零或抛错)
      num_time_bins: 时间片数量 (默认 24)
      T_hist: traffic history length
      K_min: K-minute 阈值（若无 travel_time 可选用 k_hop fallback）
      cache_dir: 缓存路径
      force_recompute: 是否强制重算缓存
      khop_fallback: int or None, 若缺 travel_time 用多少 hop 作为替代
    """
    def __init__(self, data_root,
                 static_file=None,
                 traffic_ts_file=None,
                 road_shp_file=None,
                 traj_file=None,
                 roadid_col=None,
                 feature_cols=None,
                 num_time_bins=24,
                 T_hist=12,
                 K_min=15,
                 cache_dir='./cache',
                 force_recompute=False,
                 khop_fallback=2):
        super().__init__()
        os.makedirs(cache_dir, exist_ok=True)

        if static_file is None:
            static_file = os.path.join(data_root, 'static_features.npy')
        if traffic_ts_file is None:
            traffic_ts_file = os.path.join(data_root, 'flow.npy')
        if road_shp_file is None:
            road_shp_file = os.path.join(data_root, 'map/edges.shp')
        if traj_file is None:
            traj_file = os.path.join(data_root, 'traj_train_100.csv')
        if roadid_col is None:
            roadid_col = 'fid'  # default road id column in shapefile
        if feature_cols is None:
            feature_cols = ['length', 'lanes', 'oneway']

        self.data_root = data_root
        self.road_shp_file = road_shp_file
        self.traj_file = traj_file
        self.static_file = static_file
        self.traffic_ts_file = traffic_ts_file
        self.roadid_col = roadid_col
        self.feature_cols = feature_cols
        self.num_time_bins = num_time_bins
        self.T_hist = T_hist
        self.K_min = K_min
        self.cache_dir = cache_dir
        self.force_recompute = force_recompute
        self.khop_fallback = khop_fallback

        # 1) build segment vocab
        vocab_cache = os.path.join(cache_dir, 'segment_vocab.pkl')
        if os.path.exists(vocab_cache) and not force_recompute:
            with open(vocab_cache, 'rb') as fh:
                tmp = pickle.load(fh)
                self.seg2idx = tmp['seg2idx']
                self.idx2seg = tmp['idx2seg']
        else:
            self.seg2idx, self.idx2seg = build_segment_vocab_from_trajs_and_edge(self.road_shp_file, self.roadid_col, out_vocab_file=vocab_cache)
        self.N = len(self.idx2seg) # TODO 清洗频率较低的road 加入 unk pad

        # 2) load static features (per segment)
        if static_file is not None and os.path.exists(static_file):
            self.static = torch.FloatTensor(np.load(static_file))  # expect N x C_static aligned to idx2seg order
        else:
            # fallback zero features
            self.static = torch.FloatTensor(build_static_features_for_segments(self.idx2seg, self.road_shp_file,
                                                              feature_cols=self.feature_cols, static_file=static_file, roadid_col=self.roadid_col))

        # 3) traffic timeseries (optional)
        if traffic_ts_file is not None and os.path.exists(traffic_ts_file):
            self.traffic_ts = np.load(traffic_ts_file)  # shape T_total x N_seg x C_state
            self.T_total, _, _ = self.traffic_ts.shape
        else:
            print(f"[Dataset] traffic_ts_file {traffic_ts_file} not found, will convert from traj_file")
            # 传入 idx2seg 以保证生成的 timeseries 列顺序与 vocabulary 对齐
            self.traffic_ts = convert_traj2flow(self.traj_file, self.N, idx2seg=self.idx2seg)
            self.T_total, _, _ = self.traffic_ts.shape
            
        # 4) parse trajectories into segment idx sequences and collect transitions
        self.trajs_by_timebin = defaultdict(list)
        self.global_traj_pool = []
        # We'll parse trajectories, and also collect transitions with dt if timestamps present
        transitions_samples = defaultdict(list)  # (i,j) -> list of dt (minutes)
        transitions_counts = Counter() 

        traj_df = pd.read_csv(self.traj_file, header=None)
        traj_df.columns = ['driver','traj_id','start_end', 'traj']
        points_col = 'traj'
        traj_col = 'traj_id'

        out_rows = []
        for idx, row in traj_df.iterrows():
            points_field = row.get(points_col)
            if pd.isna(points_field):
                continue
            try:
                pts = parse_points_from_field(points_field)
            except Exception as e:
                print(f"Warning: cannot parse points in CSV row {idx}: {e}", file=sys.stderr)
                continue
            traj_id = str(row.get(traj_col, idx))
            for (road_id, ts) in pts:
                out_rows.append((traj_id, int(ts), road_id))

        out_rows.sort(key=lambda x: (x[0], x[1]))
        traj_map = defaultdict(list)
        for traj_id, ts, raw in out_rows:
            if raw not in self.seg2idx:
                continue
            seg = self.seg2idx[raw]
            traj_map[traj_id].append((ts, seg))
        for traj_id, recs in traj_map.items():
            recs.sort()
            times = [r[0] for r in recs]
            nodes = [r[1] for r in recs]
            # map times to bins
            bins = [int((t // 3600) % self.num_time_bins) for t in times]
            self.global_traj_pool.append((nodes, bins))
            if len(bins) > 0:
                bin_idx = bins[-1]
                self.trajs_by_timebin[bin_idx].append((nodes, bins))
            # collect transitions
            for (t0, s0), (t1, s1) in zip(recs[:-1], recs[1:]):
                transitions_counts[(s0, s1)] += 1
                dt_min = (t1 - t0) / 60.0
                transitions_samples[(s0, s1)].append(dt_min)

        # 5) build full_edge_index (from observed transitions or edge_list if provided)
        full_edge_cache = os.path.join(cache_dir, 'full_edge_index.npy')
        if os.path.exists(full_edge_cache) and not force_recompute:
            arr = np.load(full_edge_cache)
            self.full_edge_index = torch.LongTensor(arr)
        else:
            fe_tensor, info = build_full_edges_from_shp_using_vocab(self.road_shp_file, self.seg2idx,
                                                                        roadid_col=self.roadid_col,
                                                                        from_col='u',
                                                                        to_col='v',
                                                                        assume_uv_present=True,
                                                                        coord_round=6,
                                                                        verbose=True,
                                                                        skip_missing=True)
            self.full_edge_index = fe_tensor
            if info['num_missing_roadids'] > 0:
                # 记录缺失并提示（但不阻塞）
                print("[Dataset] Warning: some roadids in shp not in vocab (count=%d). Missing sample: %s" %
                    (info['num_missing_roadids'], info['missing_roadids'][:10]))
            np.save(full_edge_cache, self.full_edge_index.cpu().numpy())

        # 6) compute P_time (per-edge log-prob). If transitions provided, compute single-bin P_time
        ptime_cache = os.path.join(cache_dir, f'P_time_{self.num_time_bins}.npy')
        if os.path.exists(ptime_cache) and not force_recompute:
            self.P_time = np.load(ptime_cache)
        else:
            # if we have transitions_counts -> compute P per src
            if self.full_edge_index.numel() == 0:
                self.P_time = np.zeros((1, 0), dtype=np.float32)
            else:
                # compute pair->count mapping
                E_full = self.full_edge_index.size(1)
                srcs = self.full_edge_index[0].cpu().numpy()
                dsts = self.full_edge_index[1].cpu().numpy()
                counts = np.zeros((E_full,), dtype=np.float32)
                src_sum = defaultdict(float)
                for idx in range(E_full):
                    s = int(srcs[idx]); d = int(dsts[idx])
                    c = float(transitions_counts.get((s, d), 0.0))
                    counts[idx] = c
                    src_sum[s] += c
                probs = np.zeros_like(counts)
                for i_e in range(E_full):
                    s = int(srcs[i_e])
                    denom = src_sum.get(s, 1.0)
                    probs[i_e] = counts[i_e] / (denom + 1e-12)
                logp = np.log(probs + 1e-9)
                self.P_time = logp.reshape(1, E_full)
                np.save(ptime_cache, self.P_time)

        # 7) build travel_time_matrix if dt samples available
        tt_cache = os.path.join(cache_dir, 'travel_time.npy')
        self.travel_time = None
        # build transitions_samples mapping to pass to travel time estimator
        # if len(transitions_samples) > 0:
        #     # if cache exists use it
        #     if os.path.exists(tt_cache) and not force_recompute:
        #         self.travel_time = np.load(tt_cache)
        #     else:
        #         self.travel_time = compute_travel_time_matrix_from_transitions(self.full_edge_index.cpu().numpy(), transitions_samples, self.N)
        #         np.save(tt_cache, self.travel_time)
        # else:
        #     self.travel_time = None

        # 8) build edge_index_kmin (K-minute neighborhood) - prefer travel_time if available, else fall back to khop
        kmin_cache = os.path.join(cache_dir, f'edge_index_kmin_{self.K_min}.npy')
        if os.path.exists(kmin_cache) and not force_recompute:
            arr = np.load(kmin_cache, allow_pickle=False)
            self.edge_index_kmin = torch.LongTensor(arr)
        else:
            if self.travel_time is not None:
                kmin = build_kmin_reachable_edges_from_travel_time_matrix(self.travel_time, self.K_min)
                self.edge_index_kmin = torch.LongTensor(kmin)
            else:
                # fallback: use khop over full_edge_index
                if self.full_edge_index.numel() == 0:
                    self.edge_index_kmin = torch.LongTensor(np.zeros((2,0), dtype=np.int64))
                else:
                    khop = build_khop_edges_from_full_edges(self.full_edge_index.cpu().numpy(), self.khop_fallback)
                    self.edge_index_kmin = torch.LongTensor(khop)
            if self.edge_index_kmin.numel() > 0:
                np.save(kmin_cache, self.edge_index_kmin.cpu().numpy())

        # 9) prepare sample time idxs
        if self.traffic_ts is None or self.T_total == 0:
            # we will sample only based on traj bins if traffic not available; create a fake time axis from traj bins
            # produce sample_time_idxs as available time bins present
            self.sample_time_idxs = sorted(list(self.trajs_by_timebin.keys()))
            if len(self.sample_time_idxs) == 0:
                # fallback to 0
                self.sample_time_idxs = [0]
        else:
            self.sample_time_idxs = list(range(self.T_hist, self.T_total))

    def __len__(self):
        return len(self.sample_time_idxs)

    def __getitem__(self, idx):
        """
        返回 item:
        - full_edge_index: 2 x E_full LongTensor
        - P_edge_full: E_full FloatTensor (for sample's time_bin) OR B x E_full handled in batch collate
        - edge_index: kmin edges 2 x E_k LongTensor
        - static: N x C_static FloatTensor
        - S_hist: T_hist x N x C_state FloatTensor (if traffic_ts available)
        - weekly_idx, daily_idx: LongTensor(T_hist) optional
        - trajs: list of (nodes_list, bins_list) - nodes_list are segment idx integers
        - time_idx: integer (global time index or bin)
        """
        t_sample = self.sample_time_idxs[idx]
        # decide time bin
        if self.traffic_ts is None or self.T_total == 0:
            time_bin = int(t_sample % self.num_time_bins)
        else:
            time_bin = int(t_sample % self.num_time_bins)

        full_edge_index = self.full_edge_index  # 2 x E_full
        # P_edge_full: choose row from P_time (if available)
        if hasattr(self, 'P_time') and self.P_time is not None and self.P_time.shape[1] == full_edge_index.size(1):
            P_edge_full = torch.FloatTensor(self.P_time[time_bin % self.P_time.shape[0]])
        else:
            P_edge_full = torch.FloatTensor(np.zeros((full_edge_index.size(1),), dtype=np.float32))

        edge_index_kmin = self.edge_index_kmin
        static = self.static

        if self.traffic_ts is not None and self.T_total > 0:
            start = int(t_sample - self.T_hist)
            if start < 0:
                # pad with zeros at beginning
                pad_len = -start
                hist = np.zeros((self.T_hist, self.N, self.traffic_ts.shape[2]), dtype=np.float32)
                hist[pad_len:] = self.traffic_ts[0:t_sample]
                S_hist = torch.FloatTensor(hist)
            else:
                S_hist = torch.FloatTensor(self.traffic_ts[start:t_sample])
        else:
            # fallback zeros
            S_hist = torch.zeros((self.T_hist, self.N, 2), dtype=torch.float32)

        weekly_idx = torch.LongTensor([ (t_sample // (24*3600)) % 7 for _ in range(self.T_hist) ])
        daily_idx = torch.LongTensor([ (t_sample // 3600) % 24 for _ in range(self.T_hist) ])

        trajs_for_bin = self.trajs_by_timebin.get(time_bin, [])
        if len(trajs_for_bin) == 0:
            trajs_for_bin = self.global_traj_pool[:100] if len(self.global_traj_pool) > 100 else self.global_traj_pool

        item = {
            'full_edge_index': full_edge_index,
            'P_edge_full': P_edge_full,
            'edge_index': edge_index_kmin,
            'static': static,
            'S_hist': S_hist,
            'weekly_idx': weekly_idx,
            'daily_idx': daily_idx,
            'trajs': trajs_for_bin,
            'time_idx': t_sample
        }
        return item

# ------------------ collate ------------------
def batch_collate_fn(batch):
    """
    batch: list of items
    返回 dict:
       'full_edge_index': 2 x E_full (shared)
       'edge_index': 2 x E_k (shared)
       'P_edge_full': B x E_full
       'static': B x N x C
       'S_hist': B x T_hist x N x C_state
       'weekly_idx': B x T_hist
       'daily_idx': B x T_hist
       'trajs': list len B
       'time_idx': list len B
    """
    B = len(batch)
    full_edge_index = batch[0]['full_edge_index']
    edge_index = batch[0]['edge_index']
    P_edges = torch.stack([it['P_edge_full'] for it in batch], dim=0) if full_edge_index.numel() > 0 else torch.zeros((B,0), dtype=torch.float32)
    static = torch.stack([it['static'] for it in batch], dim=0)
    S_hist = torch.stack([it['S_hist'] for it in batch], dim=0)
    weekly = torch.stack([it['weekly_idx'] for it in batch], dim=0)
    daily = torch.stack([it['daily_idx'] for it in batch], dim=0)
    trajs = [it['trajs'] for it in batch]
    time_idxs = [it['time_idx'] for it in batch]
    return {
        'full_edge_index': full_edge_index,
        'edge_index': edge_index,
        'P_edge_full': P_edges,
        'static': static,
        'S_hist': S_hist,
        'weekly_idx': weekly,
        'daily_idx': daily,
        'trajs': trajs,
        'time_idx': time_idxs
    }
