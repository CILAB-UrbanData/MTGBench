import ast
import csv
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle as pkl
import networkx as nx
import torch
import pickle as pkl
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt
from datetime import date, timedelta
from utils.tools import date_range, to_sparse_tensor
from utils.build_seg import build_full_segment_vocab_from_trajs_and_edge
from utils.traj2flow import convert_traj2flow
import warnings

warnings.filterwarnings('ignore')

class SF_forTrGNN_Dataset(Dataset):
    """
    SF for TrGNN Dataset
    """
    def __init__(self, args, flag, traffic_ts_file=None,
                 cache_dir = './cache',road_shp_file=None,preprocess_path=None, force_recompute=False,traj_file=None,
                 roadid_col=None, u_col = None, v_col = None, length_col = None):
        os.makedirs(cache_dir, exist_ok=True)
        self.args = args
        self.data_root = self.args.root_path
        
        if traffic_ts_file is None:
            traffic_ts_file = os.path.join(self.data_root,  f'flow_{self.args.time_interval}min.npy')
        else:
            traffic_ts_file = os.path.join(self.data_root, traffic_ts_file)
            
        if road_shp_file is None:
            road_shp_file = os.path.join(self.data_root, 'map/edges.shp')  
        else:
            road_shp_file = os.path.join(self.data_root, road_shp_file)
            
        if traj_file is None:
            traj_file = os.path.join(self.data_root, 'traj_train_100.csv')
        else:
            traj_file = os.path.join(self.data_root, traj_file)
            
        if roadid_col is None:
            roadid_col = 'fid'
        if u_col is None:
            u_col = 'u'
        if v_col is None:
            v_col = 'v'
        if length_col is None:
            length_col = 'length'
            

        self.road_shp_file = road_shp_file
        self.traj_file = traj_file
        self.roadid_col = roadid_col
        self.u_col = u_col
        self.v_col = v_col
        self.length_col = length_col
        self.min_flow_count = int(self.args.min_flow_count)

        if preprocess_path is None:
            preprocess_path = 'cache/preprocess_TrGNNsf.pkl'
        else:
            preprocess_path = os.path.join(cache_dir, preprocess_path)
        
        # 1) 初始 vocab
        vocab_cache = os.path.join(cache_dir, f"{os.path.basename(self.traj_file)}_segment_vocab.pkl")
        if os.path.exists(vocab_cache) and not force_recompute:
            with open(vocab_cache, 'rb') as fh:
                tmp = pkl.load(fh)
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
            self.traffic_ts = convert_traj2flow(self.traj_file, len(self.idx2seg), idx2seg=self.idx2seg, bin_minutes=10) #(T, N, C)
            self.T_total, _, _ = self.traffic_ts.shape 
        
        # 2.5) 低频过滤 -> UNK（仅当 min_flow_count > 0）
        self.unk_idx = None
        if self.min_flow_count > 0:
            self._apply_min_freq_filter_and_add_unk()   
        self.traffic_ts = self.traffic_ts.reshape(self.T_total, -1)  # TrGNN只有flow信息，像TRACK还要考虑 speed/status等，并且为了适配下面的scaler索性直接reshape成 (T,N)
  
        # 计算 N_total / N_graph，并保留 self.N 兼容用法
        self.N_total = len(self.idx2seg)                  # 含 UNK
        self.N_graph = self.N_total - 1 if (self.unk_idx is not None) else self.N_total  # 图中有效节点数（不含 UNK）
        self.N = self.N_total  # 兼容旧代码
        
        print(f"[Dataset] traffic_ts shape: {self.traffic_ts.shape}, N_graph: {self.N_graph}, N_total: {self.N_total}")

        assert flag in ['train', 'val', 'test']
        N_len = int(len(self.traffic_ts) * 23 / 24)  # 只保留23小时的数据
        train_n = int(N_len * 0.7)
        val_n   = int(N_len * 0.1)
        test_n  = int(N_len  - train_n - val_n)
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        self.idx_map = idx_map[flag]
        scaler = StandardScaler().fit(self.traffic_ts) # normalize flow
        self.scaler = scaler
        self.normalized_flows = torch.from_numpy(scaler.transform(self.traffic_ts)).float()
        
        #3) build graph and cal transition probability matrix
        if os.path.exists(preprocess_path) and not force_recompute:
            print('Loading preprocessed data...')
            with open(preprocess_path, 'rb') as f:
                transitions_ToD, W, W_norm = pkl.load(f)
        else:
            print("preprocessing data...")
            G = self.road_graph()
                
            road_adj = np.zeros((len(G.nodes), len(G.nodes)), dtype=np.float32)
                # masked exponential kernel. Set lambda = 1.
            lambda_ = 1
            for O in list(G.nodes):
                for D in list(G.successors(O)):
                    road_adj[self.seg_id_to_idx(O), self.seg_id_to_idx(D)] = np.exp(-lambda_ * G.edges[O, D]['weight'])
                
            trajectory_transition = self.extract_trajectory_transition()
            road_adj_mask = np.zeros(road_adj.shape)
            road_adj_mask[road_adj > 0] = 1
            np.fill_diagonal(road_adj_mask, 0)
            print("Applying road adjacency mask to trajectory transition matrices...")
            for i in range(len(trajectory_transition)):
                trajectory_transition[i] = trajectory_transition[i] + road_adj_mask
            print("Normalizing trajectory transition matrices...")
            transitions_ToD = [to_sparse_tensor(normalize_adj(trajectory_transition[i])) 
                                for i in range(len(trajectory_transition))]
            W = torch.from_numpy(road_adj)
            W_norm = torch.from_numpy(normalize_adj(road_adj, mode='aggregation'))
            print("Saving preprocessed data...")
            with open(preprocess_path, 'wb') as f:
                pkl.dump([transitions_ToD, W, W_norm], f)      
        
        self.transitions_ToD = transitions_ToD
        self.W_norm = W_norm
        self.W = W
        
        self.start_date = dt.strptime(self.args.start_date, '%Y%m%d')
        self.end_date = dt.strptime(self.args.end_date, '%Y%m%d')

        date_list = [self.start_date + timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        # 找出所有 weekday 的索引（周一到周五，weekday() 返回0~4）
        self.weekdays = np.array([i for i, d in enumerate(date_list) if d.weekday() < 5])
 
    def __len__(self):
        return len(self.idx_map)
    
    def __getitem__(self, idx):
        i = self.idx_map[idx]
        d = i // 92
        t = i % 92       

        X = self.normalized_flows[d*96+t: d*96+t+self.args.seq_len]
        T = tuple(self.transitions_ToD[t: t+self.args.seq_len])
        y_true = self.normalized_flows[d*96+t+self.args.seq_len]

        ToD = torch.from_numpy(np.eye(24)[np.full((self.traffic_ts.shape[1]), ((t+4) * 15 // 60) % 24)]).float() # one-hot encoding: hour of day. (n_road, 24)
        DoW = torch.from_numpy(np.full((self.traffic_ts.shape[1], 1), int(d in self.weekdays))).float() # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        return X, T, ToD, DoW, y_true
    
    def collate_fn(self, batch):
        """
        batch: list of samples, 每个 sample=(X, T, ToD, DoW, y_true)
        - X:      Tensor (seq_len, n_road)
        - T:      tuple of length H of sparse (n_road,n_road)
        - ToD:    Tensor (n_road, 24)
        - DoW:    Tensor (n_road, 1)
        - y_true: Tensor (n_road,)
        """
        # unzip
        Xs, Ts, ToDs, DoWs, ys = zip(*batch)
        
        # 1) stack X, ToD, DoW, y_true
        X_batch   = torch.stack(Xs,   dim=0)  # (B, H, n_road)
        ToD_batch = torch.stack(ToDs, dim=0)  # (B, n_road, 24)
        DoW_batch = torch.stack(DoWs, dim=0)  # (B, n_road, 1)
        y_batch   = torch.stack(ys,   dim=0)  # (B, n_road)
        
        # 2) 处理 T: 直接转换为 dense tensor
        H = len(Ts[0])
        B = len(Ts)
        
        # 假设所有稀疏矩阵都有相同的形状 (n_road, n_road)
        n_road = Ts[0][0].shape[0]
        
        T_batch = torch.zeros(B, H, n_road, n_road)
        for b in range(B):
            for t in range(H):
                T_batch[b, t] = Ts[b][t].to_dense()
        
        # 3) 打包 input
        inp = {
            'X':   X_batch,
            'T':   T_batch, #(H, B, n_road, n_road)
            'ToD': ToD_batch,
            'DoW': DoW_batch,
            'W_norm': self.W_norm
        }
        return inp, y_batch   
    
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

    def seg_id_to_idx(self, seg_id: int) -> int:
        return self.seg2idx.get(int(seg_id), self.unk_idx)

    def road_graph(self):        
        raw_map = self.road_shp_file
        print(f"Generating new graph from shp: {raw_map}")

        # ==== 1. 读 shp ====
        if not os.path.exists(raw_map):
            raise ValueError(f"raw_map file {raw_map} not found")

        gdf = gpd.read_file(raw_map)

        missing_cols = [c for c in [self.roadid_col, self.u_col, self.v_col, self.length_col] if c not in gdf.columns]
        if missing_cols:
            raise ValueError(f"以下字段在源 shp 中不存在: {missing_cols}")

        # 只保留需要的列
        gdf = gdf[[self.roadid_col, self.u_col, self.v_col, self.length_col]].copy()

        # road_id 转成 int，和 seg2idx 的 key 对齐
        def _to_int_safe(x):
            try:
                return int(x)
            except Exception:
                return None

        gdf["road_id_int"] = gdf[self.roadid_col].apply(_to_int_safe)
        gdf = gdf.dropna(subset=["road_id_int"])
        gdf["road_id_int"] = gdf["road_id_int"].astype(int)

        # 只保留在 vocab 里的 road_id
        vocab_road_ids = set(self.seg2idx.keys())   # 原始 road_id 集合
        gdf = gdf[gdf["road_id_int"].isin(vocab_road_ids)].copy()

        # 映射成 vocab 的索引 seg_idx
        gdf["seg_idx"] = gdf["road_id_int"].map(self.seg2idx)
        # ==== 2. 建图：节点 = vocab 索引 ====
        G = nx.DiGraph()

        N = len(self.idx2seg)

        # 节点列表：直接用 0..N-1（包括 UNK），保证和 traffic_ts / transition_tensor 完全对齐
        node_list = list(range(N))
        G.add_nodes_from(node_list)

        # 为每个“真实路段”的节点设置 length 属性
        lengths = dict(zip(gdf["seg_idx"], gdf[self.length_col]))
        # UNK 节点如果存在，给一个 0 长度
        if self.unk_idx is not None and self.unk_idx not in lengths:
            lengths[self.unk_idx] = 0.0
        nx.set_node_attributes(G, lengths, "length")

        # ==== 3. 利用 u/v 拓扑建立路段之间的邻接 ====
        # 只保留 seg_idx, u, v, length 这些列
        road_df = gdf[["seg_idx", self.u_col, self.v_col, self.length_col]].copy()

        # 左表：上一个路段 A，右表：后继路段 B
        adj_df = pd.merge(
            road_df,
            road_df,
            left_on=self.v_col,   # A.v
            right_on=self.u_col,  # B.u
            suffixes=("_x", "_y"),
        )

        # 去掉 seg_idx 一样的自环（后面会手动加 self-loop）
        adj_df = adj_df[adj_df["seg_idx_x"] != adj_df["seg_idx_y"]]

        # 计算边权重
        adj_df["distance"] = (adj_df[f"{self.length_col}_x"] + adj_df[f"{self.length_col}_y"]) / 2.0

        adj_df["edge"] = adj_df.apply(
            lambda row: (
                int(row["seg_idx_x"]),            # from vocab idx
                int(row["seg_idx_y"]),            # to vocab idx
                {"weight": float(row["distance"])},
            ),
            axis=1,
        )
        edge_list = list(adj_df["edge"])

        # ==== 4. 为所有“真实路段”添加 self-loop ====
        self_loops = [
            (int(row["seg_idx"]), int(row["seg_idx"]), {"weight": 0.0})
            for _, row in road_df.iterrows()
        ]
        edge_list.extend(self_loops)

        # UNK 节点（如果有的话）也加一个 self-loop
        if self.unk_idx is not None:
            edge_list.append((self.unk_idx, self.unk_idx, {"weight": 0.0}))

        G.add_edges_from(edge_list)
        
        return G

    def extract_trajectory_transition(self, interval=10):
        """
        只统计转移次数，不做归一化。
        输出形状 (T, N, N)，其中 N = len(idx2seg)，最后一个索引为 UNK。
        """
        if os.path.exists(os.path.join(os.path.dirname(self.traj_file),"transition.npy")):
            counts = np.load(os.path.join(os.path.dirname(self.traj_file),"transition.npy"))
            return counts
        
        N = len(self.idx2seg)

        T = int(24 * 60 / interval)
        counts = np.zeros((T, N, N), dtype=np.float64)
        
        with open(self.traj_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        # 防止空文件导致除零
        if total_lines == 0:
            raise ValueError(f"traj_file {self.traj_file} is empty.")

        print(f"[Transition] Processing traj file: {self.traj_file}")
        print(f"[Transition] Total lines: {total_lines}")

        with open(self.traj_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            line_cnt = 0          # 已读行数（包含可能的表头）
            report_every = 50000  # 每多少行汇报一次进度，可根据数据量调整
            start_time = time.time()

            for row in reader:
                line_cnt += 1

                # —— 进度 & ETA —— #
                if line_cnt % report_every == 0 or line_cnt == total_lines:
                    frac = line_cnt / total_lines
                    elapsed = time.time() - start_time  # 秒
                    if frac > 0:
                        est_total = elapsed / frac
                        remaining = est_total - elapsed
                    else:
                        remaining = 0.0

                    # 转成分钟显示（也可以换成秒、小时自己调）
                    elapsed_min = elapsed / 60.0
                    remaining_min = remaining / 60.0

                    print(
                        f"[Transition] {frac*100:5.1f}% "
                        f"({line_cnt:,}/{total_lines:,})  "
                        f"elapsed: {elapsed_min:6.2f} min  "
                        f"ETA: {remaining_min:6.2f} min"
                    )

                # —— 下面保持你原来的逻辑不变 —— #
                if not row or len(row) <= 3:
                    continue

                seg_seq_str = row[3]
                try:
                    seg_seq = ast.literal_eval(seg_seq_str)
                except Exception:
                    continue

                if not isinstance(seg_seq, list) or len(seg_seq) < 2:
                    continue

                seg_seq = sorted(seg_seq, key=lambda x: float(x[1]))

                for curr, nxt in zip(seg_seq[:-1], seg_seq[1:]):
                    try:
                        seg_id_from = int(curr[0])
                        time_from = float(curr[1])
                        seg_id_to = int(nxt[0])
                    except (IndexError, ValueError, TypeError):
                        continue

                    idx_from = self.seg_id_to_idx(seg_id_from)
                    idx_to   = self.seg_id_to_idx(seg_id_to)

                    seconds_in_day = time_from % (24 * 3600)
                    minute_in_day = int(seconds_in_day // 60)
                    t_slot = minute_in_day // interval

                    if 0 <= t_slot < T:
                        counts[t_slot, idx_from, idx_to] += 1.0

        elapsed_total = time.time() - start_time
        print(
            f"[Transition] DONE. Total processed lines: {line_cnt:,}, "
            f"time: {elapsed_total/60.0:.2f} min"
        )

        np.save(os.path.join(os.path.dirname(self.traj_file),"transition.npy"),counts)
        return counts

class DiDi_forTrGNN_Dataset(Dataset):

    def __init__(self, args, flag, start_date='20161101', end_date='20161130', root_path='data/GaiyaData/TrGNN/processed'):
        self.args = args
        self.root_path = root_path
        self.dates = date_range(start_date, end_date)
        preprocess_path = os.path.join(self.root_path, 'cache/preprocess_DiDiTrGNN.pkl')

        # weekdays scaler都要有 
        flow_df = pd.concat([pd.read_csv(os.path.join(root_path, 'tmp/flow_matched_%s_dedup_with_dwell.csv'%(date)), index_col=0) for date in self.dates])
        flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)
        self.start_date = dt.strptime(start_date, '%Y%m%d')
        self.end_date = dt.strptime(end_date, '%Y%m%d')

        date_list = [self.start_date + timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        # 找出所有 weekday 的索引（周一到周五，weekday() 返回0~4）
        self.weekdays = np.array([i for i, d in enumerate(date_list) if d.weekday() < 5])

        assert flag in ['train', 'val', 'test']
        N_len = int(len(flow_df) * 23 / 24)  # 只保留23小时的数据
        train_n = int(N_len * 0.7)
        val_n   = int(N_len * 0.1)
        test_n  = N_len - train_n - val_n
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        scaler = StandardScaler().fit(
            flow_df.iloc[idx_map['train'] + idx_map['val']].values
            ) # normalize flow
        self.scaler = scaler
        
        try:
            print('Loading preprocessed data...')
            with open(preprocess_path, 'rb') as f:
                normalized_flows, transitions_ToD, W, W_norm = pkl.load(f)
        except FileNotFoundError:
            print("文件名有错或者还未进行预处理！")      
            
        self.normalized_flows = normalized_flows
        self.transitions_ToD = transitions_ToD
        self.idx_map = idx_map[flag]
        self.flow = flow_df
        print(f"N_len: {len(self.idx_map)}")
    
    def __len__(self):
        return len(self.idx_map)
    
    def __getitem__(self, idx):
        i = self.idx_map[idx]
        d = i // 92
        t = i % 92       

        X = self.normalized_flows[d*96+t: d*96+t+self.args.seq_len]
        T = tuple(self.transitions_ToD[t: t+self.args.seq_len])
        y_true = self.normalized_flows[d*96+t+self.args.seq_len]

        ToD = torch.from_numpy(np.eye(24)[np.full((self.flow.shape[1]), ((t+4) * 15 // 60) % 24)]).float() # one-hot encoding: hour of day. (n_road, 24)
        DoW = torch.from_numpy(np.full((self.flow.shape[1], 1), int(d in self.weekdays))).float() # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        return X, T, ToD, DoW, y_true
    
    def collate_fn(self, batch):
        """
        batch: list of samples, 每个 sample=(X, T, ToD, DoW, y_true)
        - X:      Tensor (seq_len, n_road)
        - T:      tuple of length H of sparse (n_road,n_road)
        - ToD:    Tensor (n_road, 24)
        - DoW:    Tensor (n_road, 1)
        - y_true: Tensor (n_road,)
        """
        # unzip
        Xs, Ts, ToDs, DoWs, ys = zip(*batch)
        
        # 1) stack X, ToD, DoW, y_true
        X_batch   = torch.stack(Xs,   dim=0)  # (B, H, n_road)
        ToD_batch = torch.stack(ToDs, dim=0)  # (B, n_road, 24)
        DoW_batch = torch.stack(DoWs, dim=0)  # (B, n_road, 1)
        y_batch   = torch.stack(ys,   dim=0)  # (B, n_road)
        
        # 2) 处理 T: 直接转换为 dense tensor
        H = len(Ts[0])
        B = len(Ts)
        
        # 假设所有稀疏矩阵都有相同的形状 (n_road, n_road)
        n_road = Ts[0][0].shape[0]
        
        T_batch = torch.zeros(B, H, n_road, n_road)
        for b in range(B):
            for t in range(H):
                T_batch[b, t] = Ts[b][t].to_dense()
        
        # 3) 打包 input
        inp = {
            'X':   X_batch,
            'T':   T_batch, #(H, B, n_road, n_road)
            'ToD': ToD_batch,
            'DoW': DoW_batch,
        }
        return inp, y_batch

def normalize_adj(adj, mode='random walk'):
    # mode: 'random walk', 'aggregation'
    if mode == 'random walk': # for T. avg weight for sending node
        deg = np.sum(adj, axis=1).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(D_inv, adj)
    if mode == 'aggregation': # for W. avg weight for receiving node
        deg = np.sum(adj, axis=0).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(adj, D_inv)
    return normalized_adj