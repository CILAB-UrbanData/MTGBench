# dataset/real_road_dataset.py
"""
RealRoadDataset — 返回 full_edge_index + P_edge_full 以及 edge_index_kmin（K-minute neighbor）
已删除之前把 P_time 映射到 kmin 的逻辑（full->kmin 映射不再自动生成）。

输出 item 示例：
{
  'full_edge_index': 2 x E_full LongTensor,
  'P_edge_full': E_full FloatTensor,
  'edge_index': 2 x E_k LongTensor,      # 或 'edge_index_kmin'
  'static': N x C_static FloatTensor,
  'S_hist': T_hist x N x C_state FloatTensor,
  'weekly_idx': LongTensor(T_hist) or None,
  'daily_idx': LongTensor(T_hist) or None,
  'trajs': list of (nodes_list, bins_list),
  'time_idx': int
}
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
try:
    import networkx as nx
except Exception:
    nx = None

# ------------------ 辅助函数 ------------------
def build_vocab_from_edge_and_trajectories(edge_list_file, traj_file, out_vocab_file=None):
    nodes = set()
    with open(edge_list_file, 'r') as f:
        for line in f:
            sp = line.strip().split(',')
            if len(sp) < 2:
                continue
            u, v = sp[0], sp[1]
            nodes.add(int(u)); nodes.add(int(v))
    with open(traj_file, 'r') as f:
        for line in f:
            sp = line.strip().split(',')
            if len(sp) < 3: continue
            node = int(sp[2])
            nodes.add(node)
    nodes = sorted(list(nodes))
    node2idx = {int(n): i for i,n in enumerate(nodes)}
    idx2node = nodes
    if out_vocab_file:
        with open(out_vocab_file, 'wb') as fh:
            pickle.dump({'node2idx':node2idx, 'idx2node':idx2node}, fh)
    return node2idx, idx2node

def compute_travel_time_matrix_from_edge_list(edge_list_file, num_nodes, cache_path=None):
    if nx is None:
        raise RuntimeError("需要 networkx 来计算最短路径，请安装 networkx 或提供 travel_time.npy 文件")
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    with open(edge_list_file, 'r') as f:
        for line in f:
            sp = line.strip().split(',')
            if len(sp) < 3:
                continue
            u, v, w = int(sp[0]), int(sp[1]), float(sp[2])
            G.add_edge(u, v, time=w)
    T = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    for i in range(num_nodes):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='time')
        for j, t in lengths.items():
            T[i, int(j)] = float(t)
    if cache_path:
        np.save(cache_path, T)
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
    if len(src_list)==0:
        return torch.zeros((2,0), dtype=torch.long)
    edge_index = torch.LongTensor([src_list, dst_list])
    return edge_index

def build_time_dependent_P(full_edge_index, trajectories_with_times, num_time_bins):
    """
    full_edge_index: 2 x E_full (numpy or torch)
    trajectories_with_times: list of (nodes_list, times_list) with times already mapped to bins (0..num_time_bins-1)
    返回 P_time: num_time_bins x E_full (numpy array, log-probs)
    """
    if isinstance(full_edge_index, torch.Tensor):
        src = full_edge_index[0].cpu().numpy()
        dst = full_edge_index[1].cpu().numpy()
    else:
        src = np.array(full_edge_index[0])
        dst = np.array(full_edge_index[1])
    E = len(src)
    N = max(src.max(), dst.max())+1
    counts = np.ones((num_time_bins, N, N), dtype=np.float32) * 1e-6
    for nodes, times in trajectories_with_times:
        for a,b,t in zip(nodes[:-1], nodes[1:], times[:-1]):
            bin_idx = int(t % num_time_bins)
            counts[bin_idx, int(a), int(b)] += 1.0
    P_time = np.zeros((num_time_bins, E), dtype=np.float32)
    for tb in range(num_time_bins):
        row = counts[tb]
        denom = row.sum(axis=1, keepdims=True) + 1e-12
        probs = row / denom
        logp = np.log(probs + 1e-9)
        P_time[tb, :] = logp[src, dst]
    return P_time

# ------------------ Dataset 类 ------------------
class RealRoadDataset(Dataset):
    def __init__(self, data_root,
                 edge_list_file=None,
                 travel_time_file=None,
                 traj_file=None,
                 static_file=None,
                 traffic_ts_file=None,
                 num_time_bins=24,
                 T_hist=12,
                 K_min=15,
                 cache_dir='./cache',
                 force_recompute=False):
        super().__init__()
        os.makedirs(cache_dir, exist_ok=True)
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.T_hist = T_hist
        self.K_min = K_min
        self.num_time_bins = num_time_bins

        # default paths
        if edge_list_file is None:
            edge_list_file = os.path.join(data_root, 'edge_list.csv')
        if traj_file is None:
            traj_file = os.path.join(data_root, 'trajectories.csv')
        if static_file is None:
            static_file = os.path.join(data_root, 'static_nodes.npy')
        if traffic_ts_file is None:
            traffic_ts_file = os.path.join(data_root, 'traffic_timeseries.npy')

        # 1) vocab
        vocab_cache = os.path.join(cache_dir, 'vocab.pkl')
        if os.path.exists(vocab_cache) and not force_recompute:
            with open(vocab_cache, 'rb') as fh:
                tmp = pickle.load(fh)
                self.node2idx = tmp['node2idx']
                self.idx2node = tmp['idx2node']
        else:
            self.node2idx, self.idx2node = build_vocab_from_edge_and_trajectories(edge_list_file, traj_file, out_vocab_file=vocab_cache)
        self.N = len(self.idx2node)

        # 2) static nodes
        if os.path.exists(static_file):
            self.static = torch.FloatTensor(np.load(static_file))
        else:
            self.static = torch.zeros((self.N, 4), dtype=torch.float32)

        # 3) travel_time matrix
        tt_cache = os.path.join(cache_dir, 'travel_time.npy')
        if travel_time_file is not None and os.path.exists(travel_time_file):
            self.travel_time = np.load(travel_time_file)
        elif os.path.exists(tt_cache) and not force_recompute:
            self.travel_time = np.load(tt_cache)
        else:
            if not os.path.exists(edge_list_file):
                raise RuntimeError("edge_list_file missing and no travel_time cache")
            self.travel_time = compute_travel_time_matrix_from_edge_list(edge_list_file, num_nodes=self.N, cache_path=tt_cache)

        # 4) build K-minute neighbor edges (edge_index_kmin)
        edge_index_cache = os.path.join(cache_dir, f'edge_index_kmin_{self.K_min}.npy')
        if os.path.exists(edge_index_cache) and not force_recompute:
            arr = np.load(edge_index_cache, allow_pickle=False)
            self.edge_index_kmin = torch.LongTensor(arr)
        else:
            self.edge_index_kmin = build_kmin_reachable_edges_from_travel_time_matrix(self.travel_time, self.K_min)
            if self.edge_index_kmin.numel() > 0:
                np.save(edge_index_cache, self.edge_index_kmin.cpu().numpy())

        # 5) full_edge_index (所有有路径的 i->j，用于 P_time 对齐)
        srcs, dsts = np.nonzero(np.isfinite(self.travel_time) & (self.travel_time > 0))
        full_edge_index = np.stack([srcs, dsts], axis=0)
        self.full_edge_index = torch.LongTensor(full_edge_index)  # 2 x E_full

        # 6) load traffic time series
        if not os.path.exists(traffic_ts_file):
            raise RuntimeError("traffic timeseries file missing: " + traffic_ts_file)
        self.traffic_ts = np.load(traffic_ts_file)
        self.T_total, _, _ = self.traffic_ts.shape

        # 7) load/parse trajectories (map node ids -> idx) and build global traj pool
        self.trajs_by_timebin = defaultdict(list)
        self.global_traj_pool = []
        with open(traj_file, 'r') as f:
            traj_map = defaultdict(list)
            for line in f:
                sp = line.strip().split(',')
                if len(sp) < 3:
                    continue
                traj_id = sp[0]
                ts = float(sp[1])
                node_raw = int(sp[2])
                if node_raw not in self.node2idx:
                    continue
                node = self.node2idx[node_raw]
                traj_map[traj_id].append((ts, node))
            for tid, records in traj_map.items():
                records.sort()
                times = [r[0] for r in records]
                nodes = [r[1] for r in records]
                bins = [int((t // 3600) % self.num_time_bins) for t in times]
                self.global_traj_pool.append((nodes, bins))
                if len(bins) > 0:
                    bin_idx = bins[-1]
                    self.trajs_by_timebin[bin_idx].append((nodes, bins))

        # 8) build P_time aligned to full_edge_index (num_time_bins x E_full)
        ptime_cache = os.path.join(cache_dir, f'P_time_{self.num_time_bins}.npy')
        if os.path.exists(ptime_cache) and not force_recompute:
            self.P_time = np.load(ptime_cache)
        else:
            self.P_time = build_time_dependent_P(self.full_edge_index, self.global_traj_pool, num_time_bins=self.num_time_bins)
            np.save(ptime_cache, self.P_time)

        # 9) sampleable time indices (we need history of length T_hist)
        self.sample_time_idxs = list(range(self.T_hist, self.T_total))

    def __len__(self):
        return len(self.sample_time_idxs)

    def __getitem__(self, idx):
        t_global = self.sample_time_idxs[idx]
        time_bin = int(t_global % self.num_time_bins)

        # full edge index & P_edge_full (aligned to full_edge_index)
        full_edge_index = self.full_edge_index  # 2 x E_full
        P_edge_full = torch.FloatTensor(self.P_time[time_bin])  # E_full

        # K-minute neighbor edges (for co-att)
        edge_index_kmin = self.edge_index_kmin  # 2 x E_k

        static = self.static  # N x C_static

        # S_hist: recent T_hist ending at t_global
        start = t_global - self.T_hist
        S_hist = torch.FloatTensor(self.traffic_ts[start:t_global])  # T_hist x N x C_state

        weekly_idx = torch.LongTensor([ (t_global//(24*3600)) % 7 for _ in range(self.T_hist) ])
        daily_idx = torch.LongTensor([ (t_global//3600) % 24 for _ in range(self.T_hist) ])

        trajs_for_bin = self.trajs_by_timebin.get(time_bin, [])
        if len(trajs_for_bin) == 0:
            trajs_for_bin = self.global_traj_pool[:100] if len(self.global_traj_pool)>100 else self.global_traj_pool

        item = {
            'full_edge_index': full_edge_index,
            'P_edge_full': P_edge_full,
            'edge_index': edge_index_kmin,  # also accessible as 'edge_index_kmin'
            'static': static,
            'S_hist': S_hist,
            'weekly_idx': weekly_idx,
            'daily_idx': daily_idx,
            'trajs': trajs_for_bin,
            'time_idx': t_global
        }
        return item

# ------------------ collate 函数 ------------------
def list_collate_fn(batch):
    return batch

def batch_collate_fn(batch):
    """
    把多个 item 合并为 batch（更高效的 GPU 训练时使用）
    返回 dict，注意 full_edge_index 与 edge_index 假定相同（dataset 保证）
    """
    B = len(batch)
    full_edge_index = batch[0]['full_edge_index']
    edge_index = batch[0]['edge_index']
    # P_edge_full: stack (B x E_full)
    P_edges = torch.stack([it['P_edge_full'] for it in batch], dim=0)  # B x E_full
    static = torch.stack([it['static'] for it in batch], dim=0)       # B x N x C
    S_hist = torch.stack([it['S_hist'] for it in batch], dim=0)       # B x T_hist x N x C_state

    weekly = None
    daily = None
    if 'weekly_idx' in batch[0] and batch[0]['weekly_idx'] is not None:
        weekly = torch.stack([it['weekly_idx'] for it in batch], dim=0)
    if 'daily_idx' in batch[0] and batch[0]['daily_idx'] is not None:
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
