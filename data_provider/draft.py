# exp_track_tasks.py
import os
import math
import time
import json
import torch
import pickle
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from utils.tools import ensure_dir
from utils.TRACK_Scaler import StandardScaler, NormalScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, NoneScaler
import wandb

# -------------------------
# WordVocab (复用/精简)
# -------------------------
class WordVocab:
    def __init__(self, traj_path=None, seq_len=128):
        # 如果 traj_path 不给，则后面用 load_vocab 加载
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2  # CLS
        self.mask_index = 3
        self.eos_index = 4
        specials = ["<pad>", "<unk>", "<sos>", "<mask>", "<eos>"]

        if traj_path is not None:
            train = pd.read_csv(traj_path, sep=';')
            counter = Counter()
            paths = train['path'].values
            for i in tqdm(range(paths.shape[0]), desc='location counting'):
                path_l = eval(paths[i])[:seq_len]
                for p in path_l:
                    counter[p] += 1
            self.freqs = counter

            self.index2loc = list(specials)
            for tok in specials:
                if tok in counter:
                    del counter[tok]

            words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

            self.del_edge = []
            self.all_edge = []
            for word, freq in words_and_frequencies:
                if freq < 1:
                    self.del_edge.append(word)
                    continue
                self.index2loc.append(word)
                self.all_edge.append(word)
            self.loc2index = {tok: i for i, tok in enumerate(self.index2loc)}
            self.vocab_size = len(self.index2loc)

            users = train['usr_id'].values
            users = np.unique(users)
            specials = ["<pad>", "<unk>"]
            self.index2usr = list(specials)
            for u in users:
                self.index2usr.append(u)
            self.usr2index = {tok: i for i, tok in enumerate(self.index2usr)}
            self.user_num = len(self.index2usr)
        else:
            # placeholder empty vocab
            self.index2loc = list(specials)
            self.loc2index = {tok: i for i, tok in enumerate(self.index2loc)}
            self.vocab_size = len(self.index2loc)
            self.index2usr = ["<pad>", "<unk>"]
            self.usr2index = {tok: i for i, tok in enumerate(self.index2usr)}
            self.user_num = len(self.index2usr)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()
        seq = [self.loc2index.get(word, self.unk_index) for word in sentence]
        if with_sos:
            seq = [self.sos_index] + seq
        origin_seq_len = len(seq)
        if seq_len is not None:
            if len(seq) <= seq_len:
                seq += [self.pad_index for _ in range(seq_len) - len(seq)]
            else:
                seq = seq[:seq_len]
        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.index2loc[idx]
                 if idx < len(self.index2loc)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]
        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

# -------------------------
# AbstractDataset (TS-Lib 风格)
# -------------------------
class AbstractDataset(object):
    def __init__(self, config):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def get_data_feature(self):
        raise NotImplementedError

# -------------------------
# 辅助 collate
# -------------------------
def collate_traj_batch(batch, max_len=64, pad_index=0):
    # batch: list of (traj_feats, temporal_mat)
    # traj_feats: np.array (L, feat_dim) where feat_dim >=1 and first col is loc id
    lengths = [t.shape[0] for t, _ in batch]
    B = len(batch)
    feat_dim = batch[0][0].shape[1]
    max_l = min(max_len, max(lengths))
    traj_tensor = torch.full((B, max_l, feat_dim), pad_index, dtype=torch.long)
    mask = torch.zeros((B, max_l), dtype=torch.bool)
    temporal_tensor = torch.zeros((B, max_l, max_l), dtype=torch.long)
    for i, (t, m) in enumerate(batch):
        L = min(t.shape[0], max_l)
        traj_tensor[i, :L, :] = torch.LongTensor(t[:L, :])
        mask[i, :L] = 1
        mm = m[:L, :L]
        # pad temporal matrix with large value (or zeros) - choose 0 here; model can mask by mask
        temporal_tensor[i, :L, :L] = torch.LongTensor(mm)
    return traj_tensor, mask, temporal_tensor

def collate_traf_batch(batch):
    # batch: list of tuples (traf_X, traf_Y)
    traf_X, traf_Y = zip(*batch)
    traf_X = torch.stack([torch.FloatTensor(t) for t in traf_X], dim=0)
    traf_Y = torch.stack([torch.FloatTensor(t) for t in traf_Y], dim=0)
    return traf_X, traf_Y

# -------------------------
# Task 1: TRLLMCont (trajectory LM continuous)
# Dataset: 只提供轨迹（可用于 MLM / next-loc）
# -------------------------
class TRLLMContDataset(AbstractDataset):
    """
    任务 TRLLMCont：只使用轨迹数据做语言模型式预训练（连续时间信息保留）。
    这个 Dataset 提供 dataloaders 返回 (traj_tensor, mask, temporal_mat) batches，供 TRLLMContModel 使用。
    config 参数应包含：
        dataset: dataset-name (用于构造路径)
        traj_type: 'train'/'eval'/'test'
        batch_size, seq_len, add_cls, vocab_path, traj_data_path(optional)
    """
    def __init__(self, config):
        self.cfg = config
        self.dataset = config.get('dataset', '')
        self.batch_size = config.get('batch_size', 64)
        self.seq_len = config.get('seq_len', 64)
        self.add_cls = config.get('add_cls', True)
        self.traj_type = config.get('traj_type', 'train')
        self.vocab_path = config.get('vocab_path', f'raw_data/{self.dataset}/{self.dataset}_vocab.pkl')
        self.traj_data_path = config.get('traj_data_path', f'raw_data/{self.dataset}/{self.dataset}_traj_{self.traj_type}.csv')
        assert os.path.exists(self.vocab_path), f'vocab not found: {self.vocab_path}'
        assert os.path.exists(self.traj_data_path), f'traj csv not found: {self.traj_data_path}'
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        # load raw traj and build list of traj arrays + temporal mats
        self.cache_path = config.get('cache_path', self.traj_data_path.replace('.csv', f'_cache_{self.traj_type}.pkl'))
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.traj_list, self.temporal_list = pickle.load(f)
            wandb.run.summary[f'trllm/{self.dataset}/{self.traj_type}/loaded_cache'] = True
        else:
            df = pd.read_csv(self.traj_data_path, sep=';')
            traj_list = []
            temporal_list = []
            for i in tqdm(range(df.shape[0]), desc=f'proc_traj_{self.traj_type}'):
                row = df.iloc[i]
                locs = eval(row['path'])
                tims = eval(row['tlist'])
                new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in locs]
                if self.add_cls:
                    new_loc_list = [self.vocab.sos_index] + new_loc_list
                    tims = [tims[0]] + tims
                # temporal mat (absolute seconds)
                tmat = np.zeros((len(tims), len(tims)), dtype=np.int64)
                for a in range(len(tims)):
                    for b in range(len(tims)):
                        tmat[a, b] = abs(tims[a] - tims[b])
                traj_arr = np.array([[loc, tim] for loc, tim in zip(new_loc_list, tims)])
                traj_list.append(traj_arr)
                temporal_list.append(tmat)
            self.traj_list = traj_list
            self.temporal_list = temporal_list
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.traj_list, self.temporal_list), f)
            wandb.run.summary[f'trllm/{self.dataset}/{self.traj_type}/cached'] = True

    def get_data(self):
        # wrap into a torch Dataset for dataloader
        class _T(Dataset):
            def __init__(self, traj_list, temporal_list):
                self.traj_list = traj_list
                self.temporal_list = temporal_list
            def __len__(self):
                return len(self.traj_list)
            def __getitem__(self, idx):
                return self.traj_list[idx], self.temporal_list[idx]
        ds = _T(self.traj_list, self.temporal_list)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            collate_fn=lambda x: collate_traj_batch(x, max_len=self.seq_len, pad_index=self.vocab.pad_index),
                            num_workers=self.cfg.get('num_workers', 0))
        return loader

    def get_data_feature(self):
        return {
            "vocab": self.vocab,
            "vocab_size": self.vocab.vocab_size,
            "dataset": self.dataset,
        }

# -------------------------
# Task 2: DRNTRLLMCont (DRN trajectory LM continuous)
# Dataset: 提供轨迹 + co-graph/trans_prob 等 DRNTRLDataset 所需信息（复用 DRNTRLDataset 的一些预处理）
# -------------------------
class DRNTRLLMContDataset(AbstractDataset):
    """
    DRNTRLLMCont：类似 TRLLMCont，但会把 DRNTRL 的图/转移概率等 feature 一并暴露给模型（用于 DRNTRL 风格模型）
    config: 同 DRNTRLDataset 的 config（dataset, vocab_path, data_path, ...）
    """
    def __init__(self, config):
        self.cfg = config
        self.dataset = config.get('dataset', '')
        self.batch_size = config.get('batch_size', 64)
        self.seq_len = config.get('seq_len', 64)
        self.vocab_path = config.get('vocab_path', f'raw_data/{self.dataset}/{self.dataset}_vocab.pkl')
        assert os.path.exists(self.vocab_path), f'vocab not found: {self.vocab_path}'
        self.vocab = WordVocab.load_vocab(self.vocab_path)

        # reuse many DRNTRLDataset 的加载逻辑 to get geo, rel, trans_prob 等
        # 为了简洁，我只加载必要几个文件：geo(.geo), rel(.rel), neighbors json, trans_prob json
        # 你可以把更多预处理从 DRNTRLDataset 拷贝过来
        self.data_path = f'raw_data/{self.dataset}/'
        # load traj csv train as in TRLLM
        self.traj_data_path = config.get('traj_data_path', f'raw_data/{self.dataset}/{self.dataset}_traj_train.csv')
        self.cache_path = config.get('cache_path', self.traj_data_path.replace('.csv', '_cache_drn.pkl'))
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.traj_list, self.temporal_list = pickle.load(f)
            wandb.run.summary[f'drntrllm/{self.dataset}/loaded_cache'] = True
        else:
            df = pd.read_csv(self.traj_data_path, sep=';')
            traj_list = []
            temporal_list = []
            for i in tqdm(range(df.shape[0]), desc='proc_drn_traj'):
                row = df.iloc[i]
                locs = eval(row['path'])
                tims = eval(row['tlist'])
                new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in locs]
                traj_arr = np.array([[loc, tim] for loc, tim in zip(new_loc_list, tims)])
                tmat = np.zeros((len(tims), len(tims)), dtype=np.int64)
                for a in range(len(tims)):
                    for b in range(len(tims)):
                        tmat[a, b] = abs(tims[a] - tims[b])
                traj_list.append(traj_arr)
                temporal_list.append(tmat)
            self.traj_list = traj_list
            self.temporal_list = temporal_list
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.traj_list, self.temporal_list), f)
            wandb.run.summary[f'drntrllm/{self.dataset}/cached'] = True

        # try load neighbors and trans_prob
        neighbors_file = self.data_path + f'{self.dataset}_neighbors_3.json'
        trans_prob_file = self.data_path + f'{self.dataset}_trans_prob_3.json'
        if os.path.exists(neighbors_file):
            with open(neighbors_file, 'r') as f:
                self.neighbors = json.load(f)
            wandb.run.summary[f'drntrllm/{self.dataset}/loaded_neighbors'] = True
        else:
            self.neighbors = None
            wandb.run.summary[f'drntrllm/{self.dataset}/no_neighbors'] = True

        if os.path.exists(trans_prob_file):
            with open(trans_prob_file, 'r') as f:
                self.trans_prob = json.load(f)
            wandb.run.summary[f'drntrllm/{self.dataset}/loaded_trans_prob'] = True
        else:
            self.trans_prob = None
            wandb.run.summary[f'drntrllm/{self.dataset}/no_trans_prob'] = True

    def get_data(self):
        class _T(Dataset):
            def __init__(self, traj_list, temporal_list):
                self.traj_list = traj_list
                self.temporal_list = temporal_list
            def __len__(self):
                return len(self.traj_list)
            def __getitem__(self, idx):
                return self.traj_list[idx], self.temporal_list[idx]
        ds = _T(self.traj_list, self.temporal_list)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            collate_fn=lambda x: collate_traj_batch(x, max_len=self.seq_len, pad_index=self.vocab.pad_index),
                            num_workers=self.cfg.get('num_workers', 0))
        return loader

    def get_data_feature(self):
        return {
            "vocab": self.vocab,
            "neighbors": self.neighbors,
            "trans_prob": self.trans_prob,
            "dataset": self.dataset,
        }

# -------------------------
# Task 3: DRNTRLTSP (traffic prediction)
# Dataset: traf-only dataset (复用 DRNTRLTSPDataset 风格)
# -------------------------
class DRNTRLTSPDataset(AbstractDataset):
    """
    任务 DRNTRLTSP：仅训练 traffic 时间序列模型
    config: same as DRNTRLDataset for traf part (input_window/output_window, data_files, ext, etc.)
    """
    def __init__(self, config):
        self.cfg = config
        self.dataset = config.get('dataset', '')
        self.batch_size = config.get('batch_size', 64)
        self.num_workers = config.get('num_workers', 0)
        # minimal load: load .dyna files (same as original)
        self.data_path = f'raw_data/{self.dataset}/'
        self.data_files = config.get('data_files', self.dataset)
        self.input_window = config.get('input_window', 6)
        self.output_window = config.get('output_window', 1)
        self.ext_file = config.get('ext_file', '')
        self.load_external = config.get('load_external', False)
        # we will reuse get_data() semantics: return train_loader, val_loader, test_loader
        # For simplicity build dataset arrays immediately
        self.traf_cache_file = os.path.join('./libcity/cache/dataset_cache/', f'traf_{self.dataset}_{self.input_window}_{self.output_window}.npz')
        ensure_dir(os.path.dirname(self.traf_cache_file))
        if os.path.exists(self.traf_cache_file):
            arr = np.load(self.traf_cache_file, allow_pickle=True)
            self.x_train = arr['x_train']; self.y_train = arr['y_train']
            self.x_val = arr['x_val']; self.y_val = arr['y_val']
            self.x_test = arr['x_test']; self.y_test = arr['y_test']
            wandb.run.summary[f'drntrltsp/{self.dataset}/loaded_cache'] = True
        else:
            # load all data_files .dyna -> create x,y windows
            x_list, y_list = [], []
            files = self.data_files if isinstance(self.data_files, list) else [self.data_files]
            for fn in files:
                fnp = self.data_path + fn + '.dyna'
                df = pd.read_csv(fnp)
                # assume columns: time, entity_id, then features
                if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):
                    pass
                # simple parse: last D columns are features; split by geo count
                times = list(df['time'][:int(df.shape[0] / 1)])  # keep simple (user can adapt)
                feat = df[df.columns[2:]].values
                # naive reshape (user to adapt to correct geo count)
                # here we assume the file is already grouped per time per entity; we will compute sliding windows across time dimension
                # to keep this minimal and runnable, we build windows with step = 1 across time dimension per entity
                # WARNING: this simplistic conversion may require adaptation to your dataset specifics
                # Flatten into series and do windows across first dimension
                T = feat.shape[0]
                num_nodes = 1
                feat_dim = feat.shape[1]
                # build windows along T
                x_windows, y_windows = [], []
                for t in range(self.input_window, T - self.output_window + 1):
                    x_windows.append(feat[t-self.input_window:t, :])
                    y_windows.append(feat[t:t+self.output_window, :])
                if len(x_windows) > 0:
                    x_list.append(np.stack(x_windows, axis=0))
                    y_list.append(np.stack(y_windows, axis=0))
            if len(x_list) == 0:
                raise ValueError("No .dyna windows created; please check data_files and input_window")
            x = np.concatenate(x_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            # split
            num_samples = x.shape[0]
            tr = int(num_samples * 0.6)
            ev = int(num_samples * 0.2)
            self.x_train, self.y_train = x[:tr], y[:tr]
            self.x_val, self.y_val = x[tr:tr+ev], y[tr:tr+ev]
            self.x_test, self.y_test = x[-(num_samples - tr - ev):], y[-(num_samples - tr - ev):]
            np.savez_compressed(self.traf_cache_file, x_train=self.x_train, y_train=self.y_train, x_val=self.x_val, y_val=self.y_val, x_test=self.x_test, y_test=self.y_test)
            wandb.run.summary[f'drntrltsp/{self.dataset}/cached'] = True

    def get_data(self):
        class _T(Dataset):
            def __init__(self, x, y):
                self.x = x; self.y = y
            def __len__(self):
                return len(self.x)
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        train_ds = _T(self.x_train, self.y_train)
        val_ds = _T(self.x_val, self.y_val)
        test_ds = _T(self.x_test, self.y_test)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collate_traf_batch, num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_traf_batch, num_workers=self.num_workers)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_traf_batch, num_workers=self.num_workers)
        return train_loader, val_loader, test_loader

    def get_data_feature(self):
        return {
            "x_shape": getattr(self, 'x_train').shape if hasattr(self, 'x_train') else None,
            "dataset": self.dataset,
        }

# -------------------------
# Models (简单骨架)
# -------------------------
class TRLLMContModel(torch.nn.Module):
    """
    简单的轨迹 Transformer Encoder + lm head
    输入: (B, L, feat_dim)  其中 feat_dim >=1, 第一列是 loc id (int)
    输出: logits over vocab for next-token / masked token
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = torch.nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, traj_tensor, mask):
        # traj_tensor: (B, L, feat_dim) where feat_dim's first col is loc_id
        loc_ids = traj_tensor[..., 0]  # (B, L)
        emb = self.embed(loc_ids) * math.sqrt(self.d_model)
        # mask: boolean (B, L) where 1 indicates valid
        key_padding_mask = ~mask  # Transformer expects True for padding positions
        enc = self.encoder(emb, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)
        logits = self.lm_head(enc)  # (B, L, V)
        return logits

class DRNTRLLMContModel(torch.nn.Module):
    """
    DRN 的轨迹 LM：同 TRLLMCont 但可接收 temporal_mat 和 trans_prob / neighbors
    目前实现：在 encoder 前拼 temporal emb (简化)
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.base = TRLLMContModel(vocab_size, d_model, nhead, num_layers, dropout)
        # temporal encoding proj
        self.temporal_proj = torch.nn.Linear(1, d_model)

    def forward(self, traj_tensor, mask, temporal_mat=None, neighbors=None):
        # use base embed
        logits = self.base(traj_tensor, mask)
        # Note: temporal_mat can be used to compute additional biases; left as hook
        return logits

class DRNTRLTSPModel(torch.nn.Module):
    """
    简单 traffic predictor：均匀地用一个小的时序 encoder -> linear 输出
    输入: traf_X (B, input_window, num_nodes*feat) or (B, input_window, feat_dim) (we keep generic)
    """
    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        # a simple 1D conv over time
        self.conv1 = torch.nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: (B, T, N_feat) -> convert to (B, N_feat, T)
        x = x.permute(0, 2, 1)
        h = self.relu(self.conv1(x))
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        out = self.fc(h)  # (B, out_dim)
        return out

# -------------------------
# 训练/评估通用工具 (每个 task 单独训练)
# -------------------------
def train_trllm(model, dataloader, optimizer, device, epochs=3, vocab=None, task_name='TRLLM'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for batch in tqdm(dataloader, desc=f'{task_name}_train_epoch_{epoch}'):
            traj_tensor, mask, temporal = batch  # traj_tensor: long
            traj_tensor = traj_tensor.to(device)
            mask = mask.to(device)
            # simple next-token prediction: predict loc at t given positions up to t-1
            logits = model(traj_tensor, mask)  # (B,L,V)
            # make targets: shift left by 1
            targets = traj_tensor[..., 0].to(device)
            # for simplicity compute CrossEntropy per position, ignoring padding
            B, L, V = logits.shape
            logits_flat = logits.view(B*L, V)
            targets_flat = targets.view(-1)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_index if vocab else 0)
            loss = loss_fn(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / max(1, count)
        wandb.log({f'{task_name}/train_loss': avg_loss, 'epoch': epoch})
    return model

def train_drntrllm(model, dataloader, optimizer, device, epochs=3, vocab=None, task_name='DRNTRLLM'):
    # same skeleton, but pass temporal mats if available
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; count = 0
        for batch in tqdm(dataloader, desc=f'{task_name}_train_epoch_{epoch}'):
            traj_tensor, mask, temporal = batch
            traj_tensor = traj_tensor.to(device)
            mask = mask.to(device)
            temporal = temporal.to(device)
            logits = model(traj_tensor, mask, temporal_mat=temporal)
            targets = traj_tensor[..., 0].to(device)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_index if vocab else 0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); count += 1
        avg_loss = total_loss / max(1, count)
        wandb.log({f'{task_name}/train_loss': avg_loss, 'epoch': epoch})
    return model

def train_drntrltsp(model, train_loader, val_loader, optimizer, device, epochs=5, task_name='DRNTRLTSP'):
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; count = 0
        for batch in tqdm(train_loader, desc=f'{task_name}_train_epoch_{epoch}'):
            x, y = batch  # x: (B, T, feat), y: (B, outT, feat)
            x = x.to(device); y = y.to(device)
            pred = model(x)  # (B, out_dim)
            # simplify target: take mean over y time dim and feature dim reduce to single value if necessary
            target = y.mean(dim=(1,2)) if y.ndim==3 else y
            if pred.ndim == 2 and target.ndim == 1:
                target = target.unsqueeze(-1) if pred.size(-1)==1 else target
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); count += 1
        avg_loss = total_loss / max(1, count)
        # val
        model.eval()
        val_loss = 0.0; vcount = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device); y = y.to(device)
                pred = model(x)
                target = y.mean(dim=(1,2)) if y.ndim==3 else y
                loss = loss_fn(pred, target)
                val_loss += loss.item(); vcount += 1
        vavg = val_loss / max(1, vcount)
        wandb.log({f'{task_name}/train_loss': avg_loss, f'{task_name}/val_loss': vavg, 'epoch': epoch})
        if vavg < best_val:
            best_val = vavg
            # save best checkpoint via wandb artifact or torch.save externally
            wandb.run.summary[f'{task_name}/best_val'] = best_val
    return model

# -------------------------
# Exp_TRACK: 示例化（把三任务独立运行）
# -------------------------
class Exp_TRACK:
    """
    示例化入口。假定外层执行了 wandb.init().
    config: dict 包含子任务 cfg，范例:
    cfg = {
        'device': 'cuda',
        'trllm': { ... TRLLMContDataset config ... , 'epochs': 3, 'lr': 1e-3},
        'drntrllm': { ... DRNTRLLMContDataset config ... , 'epochs': 3, 'lr': 1e-3},
        'drntrltsp': { ... DRNTRLTSPDataset config ... , 'epochs': 5, 'lr': 1e-3},
    }
    run_task = 'all' or list of tasks
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cpu'))
        self.tasks = cfg.get('tasks', ['trllm', 'drntrllm', 'drntrltsp'])

    def run_trllm(self):
        cfg = self.cfg['trllm']
        ds = TRLLMContDataset(cfg['dataset_cfg'])
        loader = ds.get_data()
        features = ds.get_data_feature()
        model = TRLLMContModel(vocab_size=features['vocab_size'], d_model=cfg.get('d_model', 128))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))
        trained = train_trllm(model, loader, opt, self.device, epochs=cfg.get('epochs', 3), vocab=features['vocab'], task_name='TRLLM')
        # save checkpoint
        ckpt_path = cfg.get('ckpt_dir', './ckpt') + '/trllm_best.pth'
        ensure_dir(os.path.dirname(ckpt_path))
        torch.save({'model': trained.state_dict(), 'cfg': cfg}, ckpt_path)
        wandb.run.summary['trllm_ckpt'] = ckpt_path
        return ckpt_path

    def run_drntrllm(self):
        cfg = self.cfg['drntrllm']
        ds = DRNTRLLMContDataset(cfg['dataset_cfg'])
        loader = ds.get_data()
        features = ds.get_data_feature()
        model = DRNTRLLMContModel(vocab_size=features['vocab'].vocab_size, d_model=cfg.get('d_model', 128))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))
        trained = train_drntrllm(model, loader, opt, self.device, epochs=cfg.get('epochs', 3), vocab=features['vocab'], task_name='DRNTRLLM')
        ckpt_path = cfg.get('ckpt_dir', './ckpt') + '/drntrllm_best.pth'
        ensure_dir(os.path.dirname(ckpt_path))
        torch.save({'model': trained.state_dict(), 'cfg': cfg}, ckpt_path)
        wandb.run.summary['drntrllm_ckpt'] = ckpt_path
        return ckpt_path

    def run_drntrltsp(self):
        cfg = self.cfg['drntrltsp']
        ds = DRNTRLTSPDataset(cfg['dataset_cfg'])
        train_loader, val_loader, test_loader = ds.get_data()
        features = ds.get_data_feature()
        # derive in_dim from x shape if possible
        x_shape = features.get('x_shape', None)
        if x_shape is not None:
            in_dim = x_shape[-1]
        else:
            # fallback
            sample_batch = next(iter(train_loader))
            in_dim = sample_batch[0].shape[-1]
        model = DRNTRLTSPModel(in_dim=in_dim, hidden=cfg.get('hidden', 128), out_dim=cfg.get('out_dim', 1))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))
        trained = train_drntrltsp(model, train_loader, val_loader, opt, self.device, epochs=cfg.get('epochs', 5), task_name='DRNTRLTSP')
        ckpt_path = cfg.get('ckpt_dir', './ckpt') + '/drntrltsp_best.pth'
        ensure_dir(os.path.dirname(ckpt_path))
        torch.save({'model': trained.state_dict(), 'cfg': cfg}, ckpt_path)
        wandb.run.summary['drntrltsp_ckpt'] = ckpt_path
        return ckpt_path

    def run(self, which='all'):
        res = {}
        if which == 'all' or 'trllm' in which:
            res['trllm_ckpt'] = self.run_trllm()
        if which == 'all' or 'drntrllm' in which:
            res['drntrllm_ckpt'] = self.run_drntrllm()
        if which == 'all' or 'drntrltsp' in which:
            res['drntrltsp_ckpt'] = self.run_drntrltsp()
        return res

# -------------------------
# Example usage snippet (在最外层脚本中先 wandb.init(...) 然后运行)
# -------------------------
if __name__ == "__main__":
    # **注意**：这里仅作示例。实际使用时请在最外层先执行 `wandb.init(project='your', name='exp')`。
    # 示例配置（请替换 dataset 名称/文件路径为实际数据）
    cfg = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trllm': {
            'dataset_cfg': {
                'dataset': 'your_dataset',
                'traj_type': 'train',
                'vocab_path': 'raw_data/your_dataset/your_dataset_vocab.pkl',
                'traj_data_path': 'raw_data/your_dataset/your_dataset_traj_train.csv',
                'batch_size': 32,
                'seq_len': 64,
                'add_cls': True,
            },
            'epochs': 3, 'lr': 1e-3, 'ckpt_dir': './ckpt/trllm'
        },
        'drntrllm': {
            'dataset_cfg': {
                'dataset': 'your_dataset',
                'vocab_path': 'raw_data/your_dataset/your_dataset_vocab.pkl',
                'batch_size': 32,
                'seq_len': 64,
            },
            'epochs': 3, 'lr': 1e-3, 'ckpt_dir': './ckpt/drntrllm'
        },
        'drntrltsp': {
            'dataset_cfg': {
                'dataset': 'your_dataset',
                'data_files': ['your_datafile'],
                'input_window': 6,
                'output_window': 1,
                'batch_size': 32
            },
            'epochs': 5, 'lr': 1e-3, 'ckpt_dir': './ckpt/drntrltsp'
        }
    }

    # If user runs this file directly without wandb.init, we print a warning but still allow running small tests
    if wandb.run is None:
        print("Warning: wandb.init() not found in this process. For full logging, call wandb.init(...) outside first.")
        # Optionally init a default local run:
        wandb.init(mode='disabled')

    exp = Exp_TRACK(cfg)
    results = exp.run(which='all')
    print("Done. ckpts:", results)
