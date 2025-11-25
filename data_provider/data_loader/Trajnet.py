#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import ast
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.traj2flow import convert_traj2flow
from utils.build_seg import build_full_segment_vocab_from_trajs_and_edge
from utils.Adjacency import build_full_edges_from_shp_using_vocab

# =============================== PREFIX FOREST ==================================

class TrieNode:
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False


def build_prefix_forest(truncated_paths: List[List[int]]) -> List[TrieNode]:
    forest: Dict[int, TrieNode] = {}
    for path in truncated_paths:
        if not path:
            continue
        root = forest.setdefault(path[0], TrieNode())
        node = root
        for seg in path:
            node = node.children.setdefault(seg, TrieNode())
        node.is_end = True
    return list(forest.values())


# =============================== 轨迹滑动窗口 ==================================

def build_seg2frags_with_sliding_window(
    remapped_trajs: List[List[int]],
    trunc_length: int,
) -> Dict[int, List[List[int]]]:
    """
    按 TrajNet 原文，用滑动窗口从轨迹中抽取长度为 trunc_length 的子轨迹，
    并按「末端 segment」分桶。
    """
    seg2frags: Dict[int, List[List[int]]] = defaultdict(list)

    for traj in remapped_trajs:
        L = len(traj)
        if L < trunc_length:
            continue
        for i in range(L - trunc_length + 1):
            frag = traj[i:i + trunc_length]
            end_seg = frag[-1]
            seg2frags[end_seg].append(frag)

    return seg2frags


# =============================== flow 裁剪，不加 UNK ==================================

def apply_min_flow_filter(
    traffic_ts: np.ndarray,  # [T, N, C_state]
    min_flow_count: float,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], List[int]]:
    """
    根据列上的总流量做 min_flow_count 过滤，不引入 UNK。
    """
    T, N, C = traffic_ts.shape
    col_sum = traffic_ts.sum(axis=(0, 2))  # [N,]

    if min_flow_count > 0:
        keep_idx = [i for i in range(N) if col_sum[i] >= min_flow_count]
    else:
        keep_idx = list(range(N))

    if len(keep_idx) == 0:
        raise ValueError(f"[apply_min_flow_filter] 所有 segment 的总流量都 < {min_flow_count}")

    traffic_ts_new = traffic_ts[:, keep_idx, :]  # [T, N_keep, C]
    old2new = {old: new for new, old in enumerate(keep_idx)}
    new2old = {new: old for old, new in old2new.items()}
    return traffic_ts_new, old2new, new2old, keep_idx


# =============================== 构造 t_list（使用 stride） ===============================

def build_time_index_list(
    T_total: int,
    tau_p: int,
    tau_d: int,
    tau_w: int,
    time_interval: int,
    t_stride: int,
) -> List[int]:
    """
    构造合法的中心时间索引 t_list，并用 stride 下采样。
    """
    steps_hour = 60 // time_interval
    steps_day = 24 * steps_hour
    steps_week = 7 * steps_day

    lookback = tau_p + tau_d * steps_day + tau_w * steps_week
    min_t = lookback
    max_t = T_total - tau_p

    if max_t <= min_t:
        raise ValueError(f"[build_time_index_list] T_total={T_total} 太短")

    t_list = list(range(min_t, max_t, t_stride))
    return t_list


# =============================== 主 Dataset (方案 B + 邻接矩阵) ===============================

class Trajnet_Dataset(Dataset):
    """
    方案 B：对每个时间 t，一次性为所有 segment 采样子轨迹，并构建前缀森林。

    - traffic_ts: [T, N_seg, C_state] 由 convert_traj2flow 生成（和 TRACK 一致）
    - flow_scalar: [T, N_seg] 用于 temporal encoder & target
    - seg2frags: seg -> list of length-ℓ fragments（滑动窗口）
    - adj_mask: [N_seg, N_seg] bool，相邻为 True，下游 SpatialAttention 直接用

    __getitem__ 返回：
        {
          "t": t,
          "raw_paths": 所有 fragment 的列表 (List[List[int]]),
          "tensor_paths": 对应的 tensor 列表,
          "recent":  [N_seg, 1, T1],
          "daily":   [N_seg, 1, T2*T1],
          "weekly":  [N_seg, 1, T3*T1],
          "future_flow": [N_seg, T1]  作为全路网的 ground truth
        }

    collate_fn (建议 batch_size=1):
        inputs = {
           "forest": forest,
           "paths":  paths_padded,
           "recent": recents,      # [B, N_seg, 1, T1]
           "daily":  dailys,
           "weekly": weeklys,
        }
        targets = future_flow      # [B, N_seg, T1]
    """

    def __init__(
        self,
        args,
        flag: str,
        traffic_ts_file: str = 'flow_10min.npy',
        roadid_col: str = "fid",
        trunc_length: int = 7,
        samples_per_segment: int = 5,
        force_recompute: bool = False,
    ):
        super().__init__()
        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.root_path = args.root_path
        self.road_shp_file = os.path.join(self.root_path, args.shp_file)
        self.traj_file = os.path.join(self.root_path, args.traj_file)        
        self.time_interval = int(getattr(args, "time_interval", 10))
        self.traffic_ts_file = os.path.join(self.root_path, f"flow_{self.time_interval}min.npy")
        self.roadid_col = roadid_col
        self.min_flow_count = args.min_flow_count

        os.makedirs(args.cache_dir, exist_ok=True)
        self.cache_dir = args.cache_dir
        self.force_recompute = force_recompute

        self.trunc_length = trunc_length
        self.samples_per_segment = samples_per_segment
        self.t_stride = args.tstride

        # TrajNet 超参 (T1/T2/T3)
        self.tau_p = int(args.T1)  # 预测步数 & recent 长度
        self.tau_d = int(args.T2)
        self.tau_w = int(args.T3)


        # ========== 1) 从 shp 构建初始 vocab（完全复用 TRACK 的逻辑） ==========
        vocab_cache = os.path.join(
            self.cache_dir,
            f"{os.path.basename(self.traj_file)}_{args.model}_segment_vocab.pkl"
        )

        if os.path.exists(vocab_cache) and not self.force_recompute:
            import pickle as pkl
            with open(vocab_cache, "rb") as fh:
                tmp = pkl.load(fh)
                self.seg2idx = tmp["seg2idx"]   # road_id -> idx
                self.idx2seg = tmp["idx2seg"]   # idx -> road_id
        else:
            self.seg2idx, self.idx2seg = build_full_segment_vocab_from_trajs_and_edge(
                self.road_shp_file,
                roadid_col=self.roadid_col,
                out_vocab_file=vocab_cache,
            )

        # ========== 2) traffic_ts：与 TRACK 一样用 convert_traj2flow ==========
        # traffic_ts: [T, N_full, C_state]
        if self.traffic_ts_file is not None and os.path.exists(self.traffic_ts_file):
            self.traffic_ts = np.load(self.traffic_ts_file)
        else:
            print(f"[Data_forTrajnet] traffic_ts_file {self.traffic_ts_file} 不存在，使用 convert_traj2flow 从 {self.traj_file} 生成")
            self.traffic_ts = convert_traj2flow(
                self.traj_file,
                len(self.idx2seg),
                idx2seg=self.idx2seg,
                bin_minutes=self.time_interval,
            )
            if traffic_ts_file is not None:
                np.save(traffic_ts_file, self.traffic_ts)

        self.T_total, N_full, C_state = self.traffic_ts.shape

        # ========== 3) 按 min_flow_count 裁剪（不加 UNK） ==========
        self.traffic_ts, self.old2new, self.new2old, keep_idx = apply_min_flow_filter(
            self.traffic_ts,
            self.min_flow_count,
        )
        self.T_total, self.N_seg, self.C_state = self.traffic_ts.shape
        # idx2seg 只保留被选中的 segment id（road_id）
        self.idx2seg = [self.idx2seg[i] for i in keep_idx]

        print(f"[Data_forTrajnet] traffic_ts 裁剪后: T={self.T_total}, N_seg={self.N_seg}, C_state={self.C_state}")

        # 单通道 flow（可以选用你想要的 channel，这里默认第0维）
        self.flow_scalar = self.traffic_ts[:, :, 0]  # [T, N_seg]

        # ========== 4) 构造裁剪后的 seg2idx，用于路网构建 ==========
        self.seg2idx_filtered = {rid: i for i, rid in enumerate(self.idx2seg)}

        # ========== 5) 基于 shp + filtered vocab 构建邻接矩阵 ==========

        edge_index, info = build_full_edges_from_shp_using_vocab(
            self.road_shp_file,
            self.seg2idx_filtered,
            roadid_col=self.roadid_col,
            from_col="u",    # 按你的 shp 字段名改
            to_col="v",
            assume_uv_present=True,
            coord_round=60,
            verbose=True,
            skip_missing=True,
        )
        edge_index = edge_index.long()
        src, dst = edge_index[0], edge_index[1]

        adj = torch.zeros(self.N_seg, self.N_seg, dtype=torch.bool)
        adj[src, dst] = True
        adj[dst, src] = True
        self.adj_mask = adj    # 下游直接用 dataset.adj_mask
        with open(os.path.join(self.cache_dir, f"adjacency_{self.min_flow_count}_{args.data}.pkl"), "wb") as f:
            import pickle as pkl
            pkl.dump(self.adj_mask, f)
            
        # ========== 6) 读取轨迹并 remap 到「裁剪后」seg idx，带缓存 ==========
        traj_df = None

        cache_remap_path = os.path.join(
            self.cache_dir,
            f"remapped_trajs_minflow{self.min_flow_count}_Nseg{self.N_seg}_{args.data}.pkl"
        )

        if os.path.exists(cache_remap_path) and (not self.force_recompute):
            print(f"[Data_forTrajnet] 加载 remapped_trajs 缓存: {cache_remap_path}")
            import pickle as pkl
            with open(cache_remap_path, "rb") as f:
                remapped_trajs = pkl.load(f)

        else:
            print(f"[Data_forTrajnet] 未发现缓存 → 读取原始轨迹 CSV 并重新构建 remapped_trajs")
            print(f"            CSV 路径 = {self.traj_file}")
            traj_df = pd.read_csv(self.traj_file, header=None)

            remapped_trajs: List[List[int]] = []

            for _, row in traj_df.iterrows():
                # row: driver_id, traj_id, offsets, segment sequence
                seq_raw = row[3]
                try:
                    items = ast.literal_eval(seq_raw)
                except Exception:
                    continue

                mapped = []
                for item in items:
                    # item: [seg_id, ts, speed, duration]
                    try:
                        road_id = int(item[0])
                    except Exception:
                        continue

                    if road_id not in self.seg2idx:
                        continue
                    old_idx = self.seg2idx[road_id]
                    if old_idx not in self.old2new:
                        continue
                    new_idx = self.old2new[old_idx]
                    mapped.append(new_idx)

                if mapped:
                    remapped_trajs.append(mapped)

            if len(remapped_trajs) == 0:
                raise ValueError(
                    "[Data_forTrajnet] remapped_trajs 为空，"
                    "请检查 min_flow_count 是否过高 或轨迹文件格式是否正确。"
                )

            # ---- 保存缓存 ----
            import pickle as pkl
            with open(cache_remap_path, "wb") as f:
                pkl.dump(remapped_trajs, f)
            print(f"[Data_forTrajnet] remapped_trajs 已缓存到: {cache_remap_path}")

        # ========== 7) 滑动窗口构造 seg2frags ==========
        self.seg2frags = build_seg2frags_with_sliding_window(
            remapped_trajs,
            self.trunc_length,
        )
        self.valid_segments = [seg for seg, frags in self.seg2frags.items() if frags]
        if len(self.valid_segments) == 0:
            raise ValueError("[Data_forTrajnet] seg2frags 为空，检查 trunc_length 或 min_flow_count")

        print(f"[Data_forTrajnet] 有有效子轨迹的 segment 数: {len(self.valid_segments)}")

        # ========== 8) 使用 stride 构建 t_list，并划分 train/val/test ==========
        all_t = build_time_index_list(
            T_total=self.T_total,
            tau_p=self.tau_p,
            tau_d=self.tau_d,
            tau_w=self.tau_w,
            time_interval=self.time_interval,
            t_stride=self.t_stride,
        )
        total_len = len(all_t)
        train_n = int(0.7 * total_len)
        val_n = int(0.1 * total_len)

        if flag == "train":
            self.t_list = all_t[:train_n]
        elif flag == "val":
            self.t_list = all_t[train_n: train_n + val_n]
        else:
            self.t_list = all_t[train_n + val_n:]

        print(f"[Data_forTrajnet] flag={flag}, time samples={len(self.t_list)}")

    # ===================== flow → recent/daily/weekly（对所有 N_seg） =====================

    def _slice_flow_for_t(self, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.flow_scalar  # [T, N_seg]
        steps_hour = 60 // self.time_interval
        steps_day = 24 * steps_hour
        steps_week = 7 * steps_day

        # recent: [t-tau_p, t)
        recent = torch.from_numpy(f[t - self.tau_p: t].T).unsqueeze(1).float()  # [N_seg, 1, tau_p]

        # daily
        daily_chunks = []
        for d in range(1, self.tau_d + 1):
            base = t - d * steps_day
            daily_chunks.append(f[base - self.tau_p: base])
        daily = torch.from_numpy(np.concatenate(daily_chunks, axis=0).T).unsqueeze(1).float()  # [N_seg, 1, tau_d*tau_p]

        # weekly
        weekly_chunks = []
        for w in range(1, self.tau_w + 1):
            base = t - w * steps_week
            weekly_chunks.append(f[base - self.tau_p: base])
        weekly = torch.from_numpy(np.concatenate(weekly_chunks, axis=0).T).unsqueeze(1).float()  # [N_seg, 1, tau_w*tau_p]

        return recent, daily, weekly

    # ===================== Dataset 接口 =====================

    def __len__(self) -> int:
        return len(self.t_list)

    def __getitem__(self, idx: int):
        """
        方案 B：对一个时间 t，为所有 segment 采样若干条子轨迹，
        返回全路网的 recent/daily/weekly 和 future_flow。
        """
        t = self.t_list[idx]

        # 1) 为每个 segment 采样若干 fragment
        all_raw_paths: List[List[int]] = []
        all_tensor_paths: List[torch.Tensor] = []

        for seg, frags in self.seg2frags.items():
            if not frags:
                continue
            if len(frags) >= self.samples_per_segment:
                chosen = random.sample(frags, self.samples_per_segment)
            else:
                chosen = frags
            for frag in chosen:
                all_raw_paths.append(frag)
                all_tensor_paths.append(torch.tensor(frag, dtype=torch.long))

        # 2) flow → recent/daily/weekly
        recent, daily, weekly = self._slice_flow_for_t(t)

        # 3) 整个路网的 future_flow，作为 ground truth 矩阵 [N_seg, tau_p]
        future_flow = torch.from_numpy(self.flow_scalar[t: t + self.tau_p].T).float()  # [N_seg, tau_p]

        return {
            "t": t,
            "raw_paths": all_raw_paths,
            "tensor_paths": all_tensor_paths,
            "recent": recent,          # [N_seg,1,T1]
            "daily": daily,
            "weekly": weekly,
            "future_flow": future_flow # [N_seg,T1]
        }

    # ===================== collate_fn =====================

    def collate_fn(self, batch):
        """
        batch: List[dict]，每个 dict 是 __getitem__ 的返回：
        {
            "t": t,
            "raw_paths": List[List[int]],
            "tensor_paths": List[Tensor],
            "recent": [N_seg,1,T1],
            "daily":  [N_seg,1,T2*T1],
            "weekly": [N_seg,1,T3*T1],
            "future_flow": [N_seg,T1]
        }

        返回:
        inputs = {
            "forest": List[List[TrieNode]]  长度 B
            "recent": [B,N_seg,1,T1]
            "daily":  [B,N_seg,1,T2*T1]
            "weekly": [B,N_seg,1,T3*T1]
        }
        targets: [B,N_seg,T1]
        """
        B = len(batch)

        # 1) 每个样本一棵前缀森林
        forests = []
        for s in batch:
            forest_i = build_prefix_forest(s["raw_paths"])
            forests.append(forest_i)

        # 2) temporal 输入
        recents = torch.stack([s["recent"] for s in batch], dim=0)   # [B,N_seg,1,T1]
        dailys  = torch.stack([s["daily"]  for s in batch], dim=0)   # [B,N_seg,1,T2*T1]
        weeklys = torch.stack([s["weekly"] for s in batch], dim=0)   # [B,N_seg,1,T3*T1]

        # 3) 全路网未来 flow 作为 target
        targets = torch.stack([s["future_flow"] for s in batch], dim=0)  # [B,N_seg,T1]

        inputs = {
            "forest": forests,
            "recent": recents,
            "daily":  dailys,
            "weekly": weeklys,
        }
        return inputs, targets


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # args.root_path = "data/GaiyaData/TRACK"
    # args.shp_file = "roads_chengdu.shp"
    # args.traj_file = "traj_converted.csv"
    # args.cache_dir = "./cache"
    # args.model = "Trajnet"
    # args.tstride = 20
    # args.data = "chengdu"
    args.root_path = "data/Porto/match_jll"
    args.shp_file = "map/edges.shp"
    args.traj_file = "traj_porto.csv"
    args.cache_dir = "./cache"
    args.model = "Trajnet"
    args.tstride = 300
    args.min_flow_count = 25000
    args.data = "porto"
    args.T1 = 6
    args.T2 = 2
    args.T3 = 2
    dataset = Trajnet_Dataset(
        args,
        flag="train")
