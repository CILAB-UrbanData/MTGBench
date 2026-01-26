import ast
import os
import sys
import numpy as np
import pandas as pd
from shapely import Point
import torch
from torch.utils.data import Dataset
from utils.traj2flow import convert_traj2flow
from collections import defaultdict, Counter
import pickle
import time
from tqdm import tqdm
import geopandas as gpd
import re

try:
    import networkx as nx
except Exception:
    nx = None

# -------------------- 辅助函数 --------------------
def build_full_segment_vocab_from_trajs_and_edge(raw_map, roadid_col=None, out_vocab_file=None):
    """
    从 raw_map 文件构造 segment vocabulary（road_id -> idx）。
    """
    segs = set()
    if os.path.exists(raw_map):
        cols_to_keep = [roadid_col]
        cols_to_copy = [roadid_col]
        new_col_name = "weight"
        speed = 583  # m/min ~ 35km/h
        gdf = gpd.read_file(raw_map)
        missing_cols = [c for c in cols_to_keep if c not in gdf.columns]  
        if missing_cols:
            raise ValueError(f"以下字段在源文件中不存在: {missing_cols}")
        new_gdf = gdf[cols_to_copy].copy()
        new_gdf.rename(columns={roadid_col: "road_id"}, inplace=True)
        # 可选：travel_time = gdf["length"] / speed  # 未实际使用
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
    返回: static_features (N_seg x C_static) np.ndarray
    """
    if not os.path.exists(raw_map):
        raise ValueError(f"raw_map file {raw_map} not found")

    # 读完整 gdf，保留 geometry 用来算坐标
    gdf = gpd.read_file(raw_map)

    if roadid_col is None:
        raise ValueError("roadid_col must be specified")

    cols_to_keep = [roadid_col] + feature_cols
    missing_cols = [c for c in cols_to_keep if c not in gdf.columns]
    if missing_cols:
        raise ValueError(f"以下字段在源文件中不存在: {missing_cols}")

    # 仅用于 feature 解析的 DataFrame（不包含 geometry）
    new_gdf = gdf[cols_to_keep].copy()
    new_gdf.rename(columns={roadid_col: "road_id"}, inplace=True)
    proc_df = new_gdf[feature_cols].copy()

    # ---------- 1) 先把 feature_cols 统一成 float ----------
    def try_extract_numeric_from_str(s):
        if s is None:
            return None
        if isinstance(s, (int, float, np.number)):
            try:
                return float(s)
            except Exception:
                return None
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
        try:
            parts = re.findall(r"[-+]?\d*\.\d+|\d+", str(s))
            if parts:
                return float(parts[-1])
        except Exception:
            pass
        return None

    for col in feature_cols:
        coerced = pd.to_numeric(proc_df[col], errors='coerce')
        proc_df[col] = coerced
        na_mask = proc_df[col].isna() & new_gdf[col].notna()
        if na_mask.any():
            for idx_row in proc_df.loc[na_mask].index:
                raw = new_gdf.at[idx_row, col]
                parsed = try_extract_numeric_from_str(raw)
                if parsed is not None:
                    proc_df.at[idx_row, col] = parsed
        col_mean = proc_df[col].dropna().astype(float).mean()
        if np.isnan(col_mean):
            col_mean = 0.0
        proc_df[col] = proc_df[col].fillna(col_mean).astype(np.float32)

    # ---------- 2) 从几何信息里抽一个 (x, y) ----------
    if "geometry" not in gdf.columns:
        raise ValueError("shapefile 中缺少 geometry 列，无法提取坐标")

    # 这里用中心点，你也可以改成起点/终点坐标
    centroids = gdf.geometry.centroid
    xs = centroids.x.astype(np.float32)
    ys = centroids.y.astype(np.float32)

    # ---------- 3) 按 road_id 映射到 feature 向量 ----------
    roadid_to_features = {}
    for i, row in new_gdf.iterrows():
        rid = row["road_id"]
        try:
            rid_key = int(rid)
        except Exception:
            rid_key = rid

        base_feats = proc_df.loc[i, feature_cols].to_numpy(dtype=np.float32)
        x = float(xs.iloc[i])
        y = float(ys.iloc[i])
        full_feats = np.concatenate([[x, y], base_feats], axis=0)
        roadid_to_features[rid_key] = full_feats

    # ---------- 4) 根据 idx2seg 的顺序生成 static_features ----------
    N = len(idx2seg)
    C = len(feature_cols) + 2  # 多了 x,y 两列
    static_features = np.zeros((N, C), dtype=np.float32)
    for i, seg in enumerate(idx2seg):
        if seg in roadid_to_features:
            static_features[i] = roadid_to_features[seg]
        else:
            # 没有的就保持全 0（包括坐标）
            pass

    np.save(static_file, static_features)
    return static_features

def parse_points_from_field(field_value: str):
    """
    把类似字符串解析为 [(road_id, timestamp(int), ...), ...]
    """
    if pd.isna(field_value):
        return []
    s = field_value
    if isinstance(s, str):
        s_strip = s.strip()
        if s_strip.startswith('"') and s_strip.endswith('"'):
            s = s_strip[1:-1]
    try:
        pts = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"无法解析轨迹点字符串: {e} -- 原始值片段: {str(s)[:200]!r}")
    norm = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        road_id = p[0]
        try:
            timestamp = int(float(p[1]))
        except Exception:
            continue
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
    使用已有 seg2idx 从 shapefile 构建 full_edge_index（segment-level adjacency）。
    """
    gdf = gpd.read_file(shp_path)
    if verbose:
        print(f"[shp->full_edges] load {len(gdf)} features from {shp_path}")

    if roadid_col not in gdf.columns:
        raise ValueError(f"roadid_col '{roadid_col}' not found in shapefile columns: {gdf.columns.tolist()}")

    seg_to_endpoints = {}   # seg_idx -> (start_key, end_key)
    missing_roadids = set()
    coord_map = {}
    next_coord_node = 0

    use_uv = assume_uv_present and (from_col in gdf.columns) and (to_col in gdf.columns)
    if use_uv and verbose:
        print("[shp->full_edges] using explicit from/to columns for endpoints")

    for i, row in gdf.iterrows():
        raw_rid = row[roadid_col]
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
                if verbose:
                    print(f"[shp->full_edges] warning: seg {rid_key} has no geometry, skipping")
                continue
            try:
                coords = list(geom.coords)
            except Exception:
                if geom.geom_type == 'MultiLineString':
                    coords = list(list(geom)[0].coords)
                else:
                    raise
            start = coords[0]; end = coords[-1]
            start_k = (round(float(start[0]), coord_round), round(float(start[1]), coord_round))
            end_k = (round(float(end[0]), coord_round), round(float(end[1]), coord_round))
            if start_k not in coord_map:
                coord_map[start_k] = f"c{next_coord_node}"; next_coord_node += 1
            if end_k not in coord_map:
                coord_map[end_k] = f"c{next_coord_node}"; next_coord_node += 1
            seg_to_endpoints[seg_idx] = (("coord", coord_map[start_k]), ("coord", coord_map[end_k]))

    if verbose:
        print(f"[shp->full_edges] found {len(seg_to_endpoints)} segments matched to vocab; {len(missing_roadids)} missing roadids")

    start_to_segs = defaultdict(list)
    end_of_seg = {}
    for seg_idx, (start_key, end_key) in seg_to_endpoints.items():
        start_to_segs[start_key].append(seg_idx)
        end_of_seg[seg_idx] = end_key

    src_list, dst_list = [], []
    for A, end_key in end_of_seg.items():
        b_list = start_to_segs.get(end_key, [])
        for B in b_list:
            src_list.append(A)
            dst_list.append(B)

    if len(src_list) == 0:
        fe = np.zeros((2, 0), dtype=np.int64)
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
    E = full_edge_index.shape[1]
    pair_to_idx = {(int(full_edge_index[0, i]), int(full_edge_index[1, i])): i for i in range(E)}
    src_counts = defaultdict(float)
    pair_count_arr = np.zeros((E,), dtype=np.float32)
    for (i, j), samples in transitions_dict.items():
        if (i, j) in pair_to_idx:
            idx = pair_to_idx[(i, j)]
            pair_count_arr[idx] = float(len(samples))
            src_counts[i] += float(len(samples))
    probs = np.zeros_like(pair_count_arr, dtype=np.float32)
    for e_idx in range(E):
        s = int(full_edge_index[0, e_idx])
        denom = src_counts.get(s, 1.0)
        probs[e_idx] = pair_count_arr[e_idx] / (denom + 1e-12)
    logp = np.log(probs + 1e-9)
    P_time = logp.reshape(1, E)
    return P_time


def compute_travel_time_matrix_from_transitions(full_edge_index, transitions_dict, N):
    if nx is None:
        T = np.full((N, N), np.inf, dtype=np.float32)
        for (i, j), samples in transitions_dict.items():
            T[int(i), int(j)] = float(np.median(samples))
        return T
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for (i, j), samples in transitions_dict.items():
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
    src_list, dst_list = [], []
    for i in range(N):
        row = tt[i]
        mask = (row > 0) & (row <= K_min) & np.isfinite(row)
        idxs = np.nonzero(mask)[0]
        if idxs.size:
            src_list.extend([i] * idxs.size)
            dst_list.extend(idxs.tolist())
    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)], axis=0)


def build_khop_edges_from_full_edges(full_edge_index, k):
    src = full_edge_index[0]; dst = full_edge_index[1]
    if src.numel() == 0:
        return np.zeros((2, 0), dtype=np.int64)
    N = int(max(src.max().item(), dst.max().item()) + 1)
    adj = [[] for _ in range(N)]
    for u, v in zip(src.tolist(), dst.tolist()):
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
                    q.append((nb, dist + 1))
    if len(srcs) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.array(srcs, dtype=np.int64), np.array(dsts, dtype=np.int64)], axis=0)


class TRACKDataset(Dataset):
    """
    Dataset where each node is a road segment (road_id).
    """
    def __init__(self, args, flag,
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
        self.args = args
        data_root = args.root_path
        
        if static_file is None:
            static_file = os.path.join(data_root, 'static_features.npy')
        if traffic_ts_file is None:
            traffic_ts_file = os.path.join(data_root, f'flow_{args.time_interval}min.npy')
        if road_shp_file is None:
            #road_shp_file = os.path.join(data_root, 'map/edges.shp')  # sf porto
            road_shp_file = os.path.join(data_root, self.args.shp_file)  
        if traj_file is None:
            # traj_file = os.path.join(data_root, 'traj_train_100.csv')  #sf
            traj_file = os.path.join(data_root, self.args.traj_file)  #chengdu
            #traj_file = os.path.join(data_root, 'traj_porto.csv')  #porto
        if self.args.data == 'chengdu':
            roadid_col = 'edge_id'
        else:
            roadid_col = 'fid'
        # feature_cols = ['length', 'lanes', 'oneway']  sf
        # feature_cols = self.args.feat_col.split(",")  # cd
        feature_cols = ['length', 'oneway'] # porto

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
        self.min_flow_count = int(self.args.min_flow_count)

        # 1) 初始 vocab
        vocab_cache = os.path.join(cache_dir, f"{self.args.model}_{self.args.data}_segment_vocab.pkl")
        if os.path.exists(vocab_cache) and not force_recompute:
            with open(vocab_cache, 'rb') as fh:
                tmp = pickle.load(fh)
                self.seg2idx = tmp['seg2idx']
                self.idx2seg = tmp['idx2seg']
        else:
            self.seg2idx, self.idx2seg = build_full_segment_vocab_from_trajs_and_edge(
                self.road_shp_file, self.roadid_col, out_vocab_file=vocab_cache
            )

        # 2) traffic timeseries
        if traffic_ts_file is not None and os.path.exists(traffic_ts_file):
            self.traffic_ts = np.load(traffic_ts_file)  # T x N x C_state
            self.T_total, _, _ = self.traffic_ts.shape
        else:
            print(f"[Dataset] traffic_ts_file {traffic_ts_file} not found, will convert from traj_file")
            self.traffic_ts = convert_traj2flow(self.traj_file, len(self.idx2seg), idx2seg=self.idx2seg,bin_minutes=self.args.time_interval) #TODO: flow normalize
            self.T_total, _, _ = self.traffic_ts.shape

        # 2.5) 低频过滤 -> UNK（仅当 min_flow_count > 0）
        self.unk_idx = None
        if self.min_flow_count > 0:
            self._apply_min_freq_filter_and_add_unk()

        # 计算 N_total / N_graph，并保留 self.N 兼容用法
        self.N_total = len(self.idx2seg)                  # 含 UNK
        self.N_graph = self.N_total - 1 if (self.unk_idx is not None) else self.N_total  # 图中有效节点数（不含 UNK）
        self.N = self.N_total  # 兼容旧代码
        print(f"[Dataset] N_total={self.N_total}, N_graph={self.N_graph}, T_total={self.T_total}")
        
        assert flag in ['train', 'val', 'test']
        N_len = self.T_total - self.T_hist
        train_n = int(N_len * 0.6)
        val_n   = int(N_len * 0.2)
        test_n  = N_len - train_n - val_n
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = self.T_hist
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        self.idx_map = idx_map[flag]

        # 3) static（对齐过滤后的 idx2seg；若有 UNK 末尾补 0）
        if self.static_file is not None and os.path.exists(self.static_file):
            static_loaded = np.load(self.static_file)
            if static_loaded.shape[0] != len(self.idx2seg):
                static_loaded = build_static_features_for_segments(
                    self.idx2seg, self.road_shp_file,
                    feature_cols=self.feature_cols,
                    static_file=self.static_file, roadid_col=self.roadid_col
                )
            self.static = torch.FloatTensor(static_loaded)
        else:
            self.static = torch.FloatTensor(build_static_features_for_segments(
                self.idx2seg, self.road_shp_file,
                feature_cols=self.feature_cols, static_file=self.static_file, roadid_col=self.roadid_col
            ))
        if self.unk_idx is not None and self.static.shape[0] == (len(self.idx2seg) - 1):
            pad = torch.zeros((1, self.static.shape[1]), dtype=self.static.dtype)
            self.static = torch.vstack([self.static, pad])

        # 4) 解析轨迹，并收集转移统计
        # --------------------------------------------------
        # 先初始化这几个成员 / 变量
        self.trajs_by_timebin = defaultdict(list)
        self.global_traj_pool = []
        transitions_samples = defaultdict(list)
        transitions_counts = Counter()

        # ---- (1) 为 out_rows 做磁盘缓存 ----
        traj_cache = os.path.join(
            self.cache_dir,
            f"traj_out_rows_{self.args.model}_{self.args.data}_{self.args.min_flow_count}.pkl"
        )

        if os.path.exists(traj_cache) and not self.force_recompute:
            # 直接从缓存加载 out_rows
            with open(traj_cache, "rb") as f:
                out_rows = pickle.load(f)
            print(f"[Dataset] load cached out_rows from {traj_cache}, size={len(out_rows)}")
        else:
            # 首次或强制重算：从 CSV 解析
            traj_df = pd.read_csv(self.traj_file, header=None)
            traj_df.columns = ['driver', 'traj_id', 'start_end', 'traj']
            points_col = 'traj'
            traj_col = 'traj_id'

            out_rows = []
            total_rows = len(traj_df)
            if total_rows == 0:
                raise ValueError("traj_df is empty.")

            print(f"[Points] Start parsing traj_df, total rows: {total_rows}")
            start_time = time.time()

            report_every = 50000  # 每多少行打印一次进度，可以按数据规模调

            for ridx, row in traj_df.iterrows():
                # 进度统计
                if (ridx + 1) % report_every == 0 or (ridx + 1) == total_rows:
                    frac = (ridx + 1) / total_rows
                    elapsed = time.time() - start_time
                    if frac > 0:
                        est_total = elapsed / frac
                        remaining = est_total - elapsed
                    else:
                        remaining = 0.0

                    print(
                        f"[Points] {frac*100:5.1f}% "
                        f"({ridx+1:,}/{total_rows:,})  "
                        f"elapsed: {elapsed/60.0:6.2f} min  "
                        f"ETA: {remaining/60.0:6.2f} min"
                    )

                points_field = row.get(points_col)
                if pd.isna(points_field):
                    continue

                try:
                    pts = parse_points_from_field(points_field)
                except Exception as e:
                    print(f"Warning: cannot parse points in CSV row {ridx}: {e}", file=sys.stderr)
                    continue

                traj_id = str(row.get(traj_col, ridx))
                for (road_id, ts) in pts:
                    out_rows.append((traj_id, int(ts), road_id))

            elapsed_total = time.time() - start_time
            print(
                f"[Points] DONE. Parsed {total_rows:,} rows, "
                f"time: {elapsed_total/60.0:.2f} min"
            )

            out_rows.sort(key=lambda x: (x[0], x[1]))

            # 写入缓存
            with open(traj_cache, "wb") as f:
                pickle.dump(out_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[Dataset] save out_rows cache to {traj_cache}, size={len(out_rows)}")

        # ---- (2) 为转移统计 + 轨迹池也做磁盘缓存 ----
        stats_cache = os.path.join(
            self.cache_dir,
            f"traj_stats_{self.args.model}_{self.args.data}_{self.args.min_flow_count}.pkl"
        )

        if os.path.exists(stats_cache) and not self.force_recompute:
            # 命中缓存：直接读取转移统计 & trajs_by_timebin & global_traj_pool
            with open(stats_cache, "rb") as f:
                stats = pickle.load(f)

            # 可能存成普通 dict，这里统一成 defaultdict/list
            tb = stats.get("trajs_by_timebin", {})
            self.trajs_by_timebin = defaultdict(list, tb)
            self.global_traj_pool = stats.get("global_traj_pool", [])

            transitions_counts = stats.get("transitions_counts", Counter())
            # 注意：可能存成普通 dict，这里统一成 defaultdict(list)
            tsmp = stats.get("transitions_samples", {})
            transitions_samples = defaultdict(list, tsmp)

            print(
                f"[Dataset] load cached traj stats from {stats_cache}; "
                f"timebins={len(self.trajs_by_timebin)}, "
                f"global_trajs={len(self.global_traj_pool)}, "
                f"num_transitions={len(transitions_counts)}"
            )
        else:
            # 没有缓存：根据 out_rows 重新构建所有统计量
            traj_map = defaultdict(list)
            for traj_id, ts, raw in out_rows:
                if raw in self.seg2idx:
                    seg = self.seg2idx[raw]
                else:
                    if self.unk_idx is not None:
                        seg = self.unk_idx
                    else:
                        continue
                traj_map[traj_id].append((ts, seg))

            for traj_id, recs in traj_map.items():
                # 先按时间排序
                recs.sort()
                times = [r[0] for r in recs]
                nodes = [r[1] for r in recs]
                # 按小时 -> 时间 bin
                bins = [int((t // 3600) % self.num_time_bins) for t in times]

                # 全局轨迹池
                self.global_traj_pool.append((nodes, bins))

                # 按「最后一个时间点」所属的 timebin 分类
                if len(bins) > 0:
                    bin_idx = bins[-1]
                    self.trajs_by_timebin[bin_idx].append((nodes, bins))

                # 收集转移统计
                for (t0, s0), (t1, s1) in zip(recs[:-1], recs[1:]):
                    transitions_counts[(s0, s1)] += 1
                    dt_min = (t1 - t0) / 60.0
                    transitions_samples[(s0, s1)].append(dt_min)

            # 写入缓存（全部打包在一起）
            stats = {
                "trajs_by_timebin": dict(self.trajs_by_timebin),  # defaultdict -> dict
                "global_traj_pool": list(self.global_traj_pool),
                "transitions_counts": transitions_counts,
                "transitions_samples": dict(transitions_samples),  # defaultdict -> dict
            }
            with open(stats_cache, "wb") as f:
                pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(
                f"[Dataset] save traj stats cache to {stats_cache}; "
                f"timebins={len(self.trajs_by_timebin)}, "
                f"global_trajs={len(self.global_traj_pool)}, "
                f"num_transitions={len(transitions_counts)}"
            )

        # transitions_counts / transitions_samples 后面还会被 P_time / travel_time 使用
        # ====== 辅助: 缓存名签名 + 校验/清洗 ======
        # 这三个方法定义为成员，供后续使用
        # （为了让整文件自包含，直接写在类里，PyCharm/VSCode 可跳转）
        # 已在本类下方实现：_edge_cache_name / _assert_edge_index_ok / _sanitize_edges

        # 5) full_edge_index（从 shp，以当前 seg2idx 构建；缓存带签名）
        full_edge_cache = self._edge_cache_name('full_edge_index')
        need_build = True
        if os.path.exists(full_edge_cache) and not self.force_recompute:
            arr = np.load(full_edge_cache)
            self.full_edge_index = torch.as_tensor(arr, dtype=torch.long)
            if self._assert_edge_index_ok(self.full_edge_index, "full_edge_index(cache)"):
                need_build = False
            else:
                print("[Dataset] full_edge_index cache invalid for current vocab; will rebuild.")
        if need_build:
            fe_tensor, info = build_full_edges_from_shp_using_vocab(
                self.road_shp_file, self.seg2idx,
                roadid_col=self.roadid_col, from_col='u', to_col='v',
                assume_uv_present=True, coord_round=6, verbose=True, skip_missing=True
            )
            self.full_edge_index = self._sanitize_edges(fe_tensor, "full_edge_index(built)")
            if info['num_missing_roadids'] > 0:
                print("[Dataset] Warning: some roadids in shp not in vocab (count=%d). Missing sample: %s" %
                      (info['num_missing_roadids'], info['missing_roadids'][:10]))
            np.save(full_edge_cache, self.full_edge_index.cpu().numpy())
        # 终态自检
        assert self._assert_edge_index_ok(self.full_edge_index, "full_edge_index(final)"), "full_edge_index still invalid"

        # 6) P_time（按当前 E_full 缓存带签名）
        ptime_cache = self._edge_cache_name(f'P_time_{self.num_time_bins}')
        if os.path.exists(ptime_cache) and not self.force_recompute:
            self.P_time = np.load(ptime_cache)
        else:
            if self.full_edge_index.numel() == 0:
                self.P_time = np.zeros((1, 0), dtype=np.float32)
            else:
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

        # 7) travel_time（如需，可释放注释）
        self.travel_time = None
        # if len(transitions_samples) > 0:
        #     tt_cache = self._edge_cache_name('travel_time')
        #     if os.path.exists(tt_cache) and not self.force_recompute:
        #         self.travel_time = np.load(tt_cache)
        #     else:
        #         self.travel_time = compute_travel_time_matrix_from_transitions(
        #             self.full_edge_index.cpu().numpy(), transitions_samples, self.N_graph
        #         )
        #         np.save(tt_cache, self.travel_time)

        # 8) edge_index_kmin（缓存带签名 + 校验）
        kmin_cache = self._edge_cache_name(f'edge_index_kmin_{self.K_min}')
        need_build_k = True
        if os.path.exists(kmin_cache) and not self.force_recompute:
            arr = np.load(kmin_cache, allow_pickle=False)
            self.edge_index_kmin = torch.as_tensor(arr, dtype=torch.long)
            if self._assert_edge_index_ok(self.edge_index_kmin, "edge_index_kmin(cache)"):
                need_build_k = False
            else:
                print("[Dataset] edge_index_kmin cache invalid for current vocab; will rebuild.")
        if need_build_k:
            if self.travel_time is not None:
                kmin = build_kmin_reachable_edges_from_travel_time_matrix(self.travel_time, self.K_min)
                self.edge_index_kmin = torch.as_tensor(kmin, dtype=torch.long)
            else:
                if self.full_edge_index.numel() == 0:
                    self.edge_index_kmin = torch.zeros((2, 0), dtype=torch.long)
                else:
                    khop = build_khop_edges_from_full_edges(self.full_edge_index, self.khop_fallback)
                    self.edge_index_kmin = torch.as_tensor(khop, dtype=torch.long)
            self.edge_index_kmin = self._sanitize_edges(self.edge_index_kmin, "edge_index_kmin(built)")
            if self.edge_index_kmin.numel() > 0:
                np.save(kmin_cache, self.edge_index_kmin.cpu().numpy())
        assert self._assert_edge_index_ok(self.edge_index_kmin, "edge_index_kmin(final)"), "edge_index_kmin still invalid"

    # ============ 内部工具：缓存名/校验/清洗 ============
    def _edge_cache_name(self, prefix):
        # 签名：minfreq + N_graph，确保不同过滤结果不会撞缓存
        return os.path.join(self.cache_dir, f"{prefix}_minfreq{self.min_flow_count}_N{self.N_graph}_{self.args.model}_{self.args.data}.npy")

    def _assert_edge_index_ok(self, edge_index, name):
        if edge_index is None or edge_index.numel() == 0:
            return True
        edge_index = edge_index.long()
        src = edge_index[0]; dst = edge_index[1]
        max_idx = int(torch.max(torch.stack([src.max(), dst.max()])))
        min_idx = int(torch.min(torch.stack([src.min(), dst.min()])))
        if min_idx < 0 or max_idx >= self.N_graph:
            print(f"[Dataset][WARN] {name} out-of-range: min={min_idx}, max={max_idx}, N_graph={self.N_graph}")
            return False
        return True

    def _sanitize_edges(self, edge_index, name):
        if edge_index is None or edge_index.numel() == 0:
            return edge_index
        src, dst = edge_index[0].long(), edge_index[1].long()
        mask = (src >= 0) & (dst >= 0) & (src < self.N_graph) & (dst < self.N_graph)
        if mask.sum() != edge_index.size(1):
            bad = (~mask).nonzero(as_tuple=False).view(-1)
            examples = [(int(src[i]), int(dst[i])) for i in bad[:5]]
            print(f"[Dataset][WARN] sanitize {name}: drop {bad.numel()} / {edge_index.size(1)} edges, examples={examples}")
            edge_index = edge_index[:, mask]
        return edge_index.contiguous().long()

    # ============ 低频过滤：保留高频 + 末尾追加 UNK（不进图） ============
    def _apply_min_freq_filter_and_add_unk(self):
        """
        基于 self.traffic_ts 的列总和筛除低频路段；重建 idx2seg/seg2idx；
        末尾追加 UNK 索引；并在 traffic_ts 末尾追加 UNK 列（全 0）。
        """
        T, N, C = self.traffic_ts.shape
        assert N == len(self.idx2seg), "traffic_ts 列数必须与 vocab 对齐。"

        counts = self.traffic_ts.sum(axis=(0, 2))  # shape (N,)
        keep_mask = counts >= self.min_flow_count
        keep_idx = np.where(keep_mask)[0].tolist()

        if len(keep_idx) == N:
            return

        removed = N - len(keep_idx)
        print(f"[Dataset] min_flow_count={self.min_flow_count}: keep {len(keep_idx)}/{N} segments, removed {removed}, append UNK.")

        idx2seg_new = [self.idx2seg[i] for i in keep_idx]
        self.unk_idx = len(idx2seg_new)
        idx2seg_new.append('UNK')

        seg2idx_new = {rid: i for i, rid in enumerate(idx2seg_new[:-1])}

        ts_kept = self.traffic_ts[:, keep_mask, :]
        zeros_col = np.zeros((ts_kept.shape[0], 1, ts_kept.shape[2]), dtype=ts_kept.dtype)
        self.traffic_ts = np.concatenate([ts_kept, zeros_col], axis=1)
        self.T_total = self.traffic_ts.shape[0]

        self.idx2seg = idx2seg_new
        self.seg2idx = seg2idx_new

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        """
        返回 item:
        - full_edge_index: 2 x E_full LongTensor
        - P_edge_full: E_full FloatTensor
        - edge_index: kmin edges 2 x E_k LongTensor
        - static: N x C_static FloatTensor
        - S_hist: T_hist x N x C_state FloatTensor
        - weekly_idx, daily_idx: LongTensor(T_hist)
        - trajs: list of (nodes_list, bins_list)
        - time_idx: integer
        """
        t_sample = self.idx_map[idx]
        time_bin = int(t_sample % self.num_time_bins)

        full_edge_index = self.full_edge_index  # 2 x E_full
        if hasattr(self, 'P_time') and self.P_time is not None and self.P_time.shape[1] == full_edge_index.size(1):
            P_edge_full = torch.FloatTensor(self.P_time[time_bin % self.P_time.shape[0]])
        else:
            P_edge_full = torch.FloatTensor(np.zeros((full_edge_index.size(1),), dtype=np.float32))

        edge_index_kmin = self.edge_index_kmin
        static = self.static

        if self.traffic_ts is not None and self.T_total > 0:
            start = int(t_sample - self.T_hist)
            if start < 0:
                pad_len = -start
                hist = np.zeros((self.T_hist, self.N_total, self.traffic_ts.shape[2]), dtype=np.float32)
                hist[pad_len:] = self.traffic_ts[0:t_sample]
                S_hist = torch.FloatTensor(hist)
            else:
                S_hist = torch.FloatTensor(self.traffic_ts[start:t_sample])
        else:
            S_hist = torch.zeros((self.T_hist, self.N_total, 2), dtype=torch.float32)

        weekly_idx = torch.LongTensor([ (t_sample // 144) % 7 for _ in range(self.T_hist) ])
        daily_idx = torch.LongTensor([ (t_sample // 6) % 24 for _ in range(self.T_hist) ])

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
    def collate_fn(self, batch):
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
        P_edges = torch.stack([it['P_edge_full'] for it in batch], dim=0) if full_edge_index.numel() > 0 else torch.zeros((B, 0), dtype=torch.float32)
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
        }, S_hist[:,-self.args.pre_steps].reshape(B,self.args.NumofRoads,-1)

if __name__ == "__main__":
    # 示例用法
    data_root = 'data/GaiyaData/TRACK'
    dataset = TRACKDataset(data_root, flag='train', args={},road_shp_file="data/GaiyaData/TRACK/roads_chengdu.shp",
                           traj_file="data/GaiyaData/TRACK/traj_converted.csv",roadid_col='edge_id',feature_cols=['bridge','tunnel','oneway'],cache_dir='./cache',force_recompute=True)
    print(f"NumofRoads: {int(dataset.static.shape[0])}")
