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
from logging import getLogger
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from utils.TRACK_data_util import WordVocab, AbstractDataset
from utils.tools import ensure_dir
from utils.TRACK_Scaler import StandardScaler, NormalScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, NoneScaler

# -------------------------
# Wandb-safe logging helpers (you already provided these)
# -------------------------
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

def _safe_wandb_log(key: str, value):
    """
    Unified wandb logger: logs a key/value pair under the given key.
    Falls back to print when wandb isn't available or not initialized.
    """
    try:
        if _HAS_WANDB and wandb.run is not None:
            # If value is string, wrap in dict for consistency
            wandb.log({key: value})
            # also keep a last-seen summary entry (convenience)
            try:
                # store last scalar/string under summary for quick view
                if isinstance(value, (int, float)):
                    wandb.run.summary[key] = float(value)
                else:
                    wandb.run.summary[f"{key}_last"] = str(value)[:4096]
            except Exception:
                pass
        else:
            print(f"[WANDB LOG] {key}: {value}")
    except Exception:
        print(f"[WANDB LOG ERROR] {key}: {value}")

def _winfo(msg: str):
    _safe_wandb_log("dataset/info", msg)

def _wwarn(msg: str):
    _safe_wandb_log("dataset/warn", msg)

def safe_wandb_metric(name: str, value):
    """Log numeric metric under dataset/metric/<name>"""
    _safe_wandb_log(f"dataset/metric/{name}", float(value))

class TrajDataset(Dataset):

    def __init__(
        self, data_name, data_type, vocab,
        add_cls=False, max_train_size=None, temporal_index=False, seq_len=64,
    ):
        self.vocab = vocab
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self.temporal_index = temporal_index
        self.seq_len = seq_len
        self._logger = getLogger()

        self.data_path = 'raw_data/{0}/{0}_traj_{1}.csv'.format(data_name, data_type)
        self.cache_path = 'raw_data/{0}/cache_{0}_traj_{1}_{2}.pkl'.format(data_name, data_type, add_cls)
        if self.temporal_index:
            self.temporal_mat_path = self.cache_path[:-4] + '_temporal_mat_index.pkl'
        else:
            self.temporal_mat_path = self.cache_path[:-4] + '_temporal_mat.pkl'
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.cache_path) and os.path.exists(self.temporal_mat_path):
            self.traj_list = pickle.load(open(self.cache_path, 'rb'))
            _winfo('Loaded trajectory dataset from {}'.format(self.cache_path))
            self.temporal_mat_list = pickle.load(open(self.temporal_mat_path, 'rb'))
            _winfo('Loaded temporal matrix from {}'.format(self.temporal_mat_path))
        else:
            data = pd.read_csv(self.data_path, sep=';')
            self.traj_list, self.temporal_mat_list = self.data_processing(data)
        if self.max_train_size is not None:
            self.traj_list = self.traj_list[:self.max_train_size]
            self.temporal_mat_list = self.temporal_mat_list[:self.max_train_size]

    def _cal_mat(self, tim_list):
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if self.temporal_index:
                    off = abs(i - j)
                else:
                    off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)

    def data_processing(self, origin_data, desc=None, cache_path=None, temporal_mat_path=None):
        _winfo('Processing dataset in TrajDataset!')
        sub_data = origin_data[['path', 'tlist', 'usr_id', 'traj_id']]
        traj_list = []
        temporal_mat_list = []
        for i in tqdm(range(sub_data.shape[0]), desc=desc):
            traj = sub_data.iloc[i]
            loc_list = eval(traj['path'])
            tim_list = eval(traj['tlist'])
            usr_id = traj['usr_id']
            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.fromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.usr2index.get(usr_id, self.vocab.unk_index)] * len(new_loc_list)
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
            temporal_mat = self._cal_mat(tim_list)  # (seq_len, seq_len)
            temporal_mat_list.append(temporal_mat)
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks, usr_list]).transpose((1, 0))  # (seq_len, feat_dim)
            traj_list.append(traj_fea)
        if cache_path is None:
            cache_path = self.cache_path
        if temporal_mat_path is None:
            temporal_mat_path = self.temporal_mat_path
        pickle.dump(traj_list, open(cache_path, 'wb'))
        pickle.dump(temporal_mat_list, open(temporal_mat_path, 'wb'))
        safe_wandb_metric('traj_count', len(traj_list))
        return traj_list, temporal_mat_list  # [loc, tim, mins, weeks, usr]

    def __len__(self):
        return len(self.traj_list)

    def __getitem__(self, ind):
        traj_ind = self.traj_list[ind]
        temporal_mat = self.temporal_mat_list[ind]
        return torch.LongTensor(traj_ind), torch.LongTensor(temporal_mat)

class TRLDataset(AbstractDataset):

    def __init__(self, config):
        self._logger = getLogger()
        self.config = config
        self.dataset = config.get('dataset', '')
        self.geo_file = config.get('geo_file', self.dataset)
        self.rel_file = config.get('rel_file', self.dataset)
        self.cache_dataset = config.get('cache_dataset', True)
        self.batch_size = config.get('batch_size', 256)
        self.num_workers = config.get('num_workers', 0)
        self.train_rate = config.get('train_rate', 0.6)
        self.part_train_rate = config.get('part_train_rate', 1)
        self.eval_rate = config.get('eval_rate', 0.2)
        self.bidir = config.get('bidir', False)
        self.device = config.get('device', torch.device('cpu'))
        self.time_intervals = config.get('time_intervals', 1800)
        begin_time = config.get('begin_time', '2018-10-01 00:00:00')
        self.begin_timestamp = time.mktime(time.strptime(begin_time, "%Y-%m-%d %H:%M:%S"))

        self.data_path = f'./raw_data/{self.dataset}/'
        if not os.path.exists(self.data_path):
            raise ValueError(f"Dataset {self.dataset} not exist! Please ensure the path {self.data_path} exist!")

        self.max_train_size = config.get('max_train_size', None)
        self.vocab_path = config.get('vocab_path', None)
        self.min_freq = config.get('min_freq', 1)
        self.merge = config.get('merge', True)
        if self.vocab_path is None:
            self.vocab_path = 'raw_data/{0}/{0}_vocab.pkl'.format(self.dataset)
        self.seq_len = config.get('seq_len', 64)
        self.add_cls = config.get('add_cls', True)
        self._load_vocab()
        self.collate_fn = None
        self.add_degree = config.get('add_degree', False)
        self.neighbors_K = config.get('neighbors_K', 1)
        self.load_trans_prob = config.get('load_trans_prob', False)
        self.load_t_trans_prob = config.get('load_t_trans_prob', True)
        self.normal_feature = config.get('normal_feature', False)
        self.temporal_index = config.get('temporal_index', False)
        self.bidir_adj_mx = config.get('bidir_adj_mx', False)

        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self.geo_file = self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.rel'):
            self.rel_file = self._load_rel()
        else:
            self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
            self.adj_mx_encoded = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)
        self.node_features, self.node_fea_dim = self._load_geo_feature(self.geo_file)
        self.edge_index, self.loc_trans_prob, self.t_loc_trans_prob = self._load_k_neighbors_and_trans_prob()

    def _load_vocab(self):
        assert os.path.exists(self.vocab_path), 'Vocab at %s not found' % self.vocab_path
        _winfo("Loading Vocab from {}".format(self.vocab_path))
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.usr_num = self.vocab.user_num
        self.vocab_size = self.vocab.vocab_size
        _winfo(f'vocab_path={self.vocab_path}, usr_num={self.usr_num}, vocab_size={self.vocab_size}')

    def _load_geo(self):
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        # 使用WordVocab中的loc2index需要确保geo文件和index2loc中geo_id的顺序一致
        self.geo_to_ind = {geo: ind - 5 for geo, ind in self.vocab.loc2index.items()}
        self.ind_to_geo = self.vocab.index2loc[5:]
        _winfo(f"Loaded file {self.geo_file}.geo, num_nodes={str(len(self.geo_ids))}")
        return geofile

    def _load_geo_feature(self, road_info):
        node_fea_path = self.data_path + f'{self.dataset}_node_features.npy'
        if self.add_degree:
            node_fea_path = node_fea_path[:-4] + '_degree.npy'
        if os.path.exists(node_fea_path):
            node_features = np.load(node_fea_path, allow_pickle=True)
        else:
            useful = ['highway', 'lanes', 'length', 'maxspeed']
            if self.add_degree:
                useful += ['outdegree', 'indegree']
            node_features = road_info[useful]
            norm_list = ['length']
            for col in norm_list:
                d = node_features[col]
                min_ = d.min()
                max_ = d.max()
                dnew = (d - min_) / (max_ - min_)
                node_features = node_features.drop(col, axis=1)
                node_features.insert(useful.index(col), col, dnew)
            onehot_list = ['lanes', 'maxspeed', 'highway']
            if self.add_degree:
                onehot_list += ['outdegree', 'indegree']
            for col in onehot_list:
                dum_col = pd.get_dummies(node_features[col], col)
                node_features = node_features.drop(col, axis=1)
                node_features = pd.concat([node_features, dum_col], axis=1)
            node_features = node_features.values
            np.save(node_fea_path, node_features)

        _winfo(f"node_features: {str(node_features.shape)}")  # (N, fea_dim)
        node_fea_vec = np.zeros((self.vocab.vocab_size, node_features.shape[1]))
        for ind in range(len(node_features)):
            if ind < len(self.ind_to_geo):
                geo_id = self.ind_to_geo[ind]
                encoded_geo_id = self.vocab.loc2index[geo_id]
                node_fea_vec[encoded_geo_id] = node_features[ind]
            else:
                _wwarn(f"Index {ind} >= len(ind_to_geo), skipping node features write")
        if self.normal_feature:
            _winfo('node_features by a/row_sum(a)')  # (vocab_size, fea_dim)
            row_sum = np.clip(node_fea_vec.sum(1), a_min=1, a_max=None)
            for i in range(len(node_fea_vec)):
                node_fea_vec[i, :] = node_fea_vec[i, :] / row_sum[i]
        node_fea_pe = torch.from_numpy(node_fea_vec).float().to(self.device)  # (vocab_size, fea_dim)
        _winfo(f'node_features_encoded: {str(node_fea_pe.shape)}')  # (vocab_size, fea_dim)
        return node_fea_pe, node_fea_pe.shape[1]

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')[['origin_id', 'destination_id']]
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        self.adj_mx_encoded = np.zeros((self.vocab.vocab_size, self.vocab.vocab_size), dtype=np.float32)
        for row in relfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                _wwarn(f"{row[0]} or {row[1]} not in geo_to_ind, skipping")
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            self.adj_mx_encoded[self.vocab.loc2index[row[0]], self.vocab.loc2index[row[1]]] = 1
            if self.bidir_adj_mx:
                self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
                self.adj_mx_encoded[self.vocab.loc2index[row[1]], self.vocab.loc2index[row[0]]] = 1

        _winfo("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape) +
                          ', edges=' + str(self.adj_mx.sum()))
        _winfo('adj_mx_encoded, shape=' + str(self.adj_mx_encoded.shape) +
                          ', edges=' + str(self.adj_mx_encoded.sum()))
        return relfile

    def _load_k_neighbors_and_trans_prob(self):
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()
        loc_trans_prob = None
        t_loc_trans_prob = None
        geoid2neighbors = json.load(open(self.data_path + self.dataset + f'_neighbors_{self.neighbors_K}.json'))
        if self.load_trans_prob:
            loc_trans_prob = []
            link2prob = json.load(open(self.data_path + self.dataset + f'_trans_prob_{self.neighbors_K}.json'))
        if self.load_t_trans_prob:
            t_link2prob = json.load(open(self.data_path + self.dataset + f'_t_trans_prob_{self.neighbors_K}.json'))
            t_loc_trans_prob = [[] for _ in range(len(t_link2prob))]
        for k, v in geoid2neighbors.items():
            src_node = self.vocab.loc2index[int(k)]
            for tgt in v:
                trg_node = self.vocab.loc2index[int(tgt)]
                if (src_node, trg_node) not in seen_edges:
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)
                    seen_edges.add((src_node, trg_node))
                    if self.load_trans_prob:
                        loc_trans_prob.append(link2prob[str(k) + '_' + str(tgt)])
                    if self.load_t_trans_prob:
                        for j in range(len(t_link2prob)):
                            t_loc_trans_prob[j].append(t_link2prob[str(j)][str(k) + '_' + str(tgt)])
        # add_self_edge
        for i in range(self.vocab.vocab_size):
            if (i, i) not in seen_edges:
                source_nodes_ids.append(i)
                target_nodes_ids.append(i)
                seen_edges.add((i, i))
                if self.load_trans_prob:
                    loc_trans_prob.append(link2prob.get(str(i) + '_' + str(i), 0.0))
                if self.load_t_trans_prob:
                    for j in range(len(t_link2prob)):
                        t_loc_trans_prob[j].append(t_link2prob[str(j)].get(str(i) + '_' + str(i), 0.0))
        # shape = (2, E), where E is the number of edges in the graph
        edge_index = torch.from_numpy(np.row_stack((source_nodes_ids, target_nodes_ids))).long().to(self.device)
        _winfo(f'trajectory edge_index: {str(edge_index.shape)}')
        if self.load_trans_prob:
            loc_trans_prob = torch.from_numpy(np.array(loc_trans_prob)).unsqueeze(1).float().to(self.device)  # (E, 1)
            _winfo(f'Trajectory loc-transfer prob shape={loc_trans_prob.shape}')
        if self.load_t_trans_prob:
            t_loc_trans_prob = torch.from_numpy(np.array(t_loc_trans_prob)).unsqueeze(2).float().to(self.device)  # (336, E, 1)
            _winfo(f'Trajectory dynamic loc-transfer prob shape={t_loc_trans_prob.shape}')
        return edge_index, loc_trans_prob, t_loc_trans_prob

    def _gen_dataset(self):
        train_dataset = TrajDataset(
            data_name=self.dataset, data_type='train', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=self.max_train_size, temporal_index=self.temporal_index,
        )
        eval_dataset = TrajDataset(
            data_name=self.dataset, data_type='eval', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
        )
        test_dataset = TrajDataset(
            data_name=self.dataset, data_type='test', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
        )
        return train_dataset, eval_dataset, test_dataset

    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        assert self.collate_fn is not None
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls),
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls),
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls),
        )
        self.num_batches = len(train_dataloader)

        return train_dataloader, eval_dataloader, test_dataloader

    def get_data(self):
        train_dataset, eval_dataset, test_dataset = self._gen_dataset()
        _winfo("Creating Dataloader!")
        return self._gen_dataloader(train_dataset, eval_dataset, test_dataset)

    def get_data_feature(self):
        data_feature = {
            "usr_num": self.usr_num, "num_nodes": self.num_nodes,
            "vocab": self.vocab, "vocab_size": self.vocab_size,
            "geo_file": self.geo_file,
            "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
            "node_features": self.node_features, "node_fea_dim": self.node_fea_dim,
            "edge_index": self.edge_index, "loc_trans_prob": self.loc_trans_prob, "t_loc_trans_prob": self.t_loc_trans_prob,
            "adj_mx_encoded": self.adj_mx_encoded,
        }
        return data_feature

class TrajTrafDataset(Dataset):

    def __init__(
        self, data_name, traf_data, traf_name, data_type, traj_batch_size, vocab,
        add_cls=False, max_train_size=None, temporal_index=False, seq_len=64,
        begin_timestamp=None, begin_ti=0, time_intervals=1800,
    ):
        self.traf_data = traf_data
        self.vocab = vocab
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self.temporal_index = temporal_index
        self.traj_batch_size = traj_batch_size
        self.begin_timestamp = begin_timestamp
        self.begin_ti = begin_ti
        self.time_intervals = time_intervals
        self.seq_len = seq_len

        self.traj_data_path = 'raw_data/{0}/{0}_traj_{1}.csv'.format(data_name, data_type)
        self.traj_cache_path = 'raw_data/{0}/cache_{0}_traj_{1}_{2}.pkl'.format(data_name, data_type, add_cls)
        self.traj_dict_cache_path = 'raw_data/{0}/cache_{0}_traj_dict_{1}_{2}_{3}.pkl'.format(data_name, data_type, add_cls, time_intervals)
        if self.temporal_index:
            self.temporal_mat_path = self.traj_cache_path[:-4] + '_temporal_mat_index.pkl'
            self.temporal_mat_dict_path = self.traj_dict_cache_path[:-4] + '_temporal_mat_index.pkl'
        else:
            self.temporal_mat_path = self.traj_cache_path[:-4] + '_temporal_mat.pkl'
            self.temporal_mat_dict_path = self.traj_dict_cache_path[:-4] + '_temporal_mat.pkl'
        self.traj_traf_cache_path = 'raw_data/{0}/cache_{0}_traj_traf_{1}_{2}_{3}_{4}_{5}.pkl'.format(
            data_name, data_type, add_cls, time_intervals, traj_batch_size, traf_name,
        )
        self._load_data()

    def _load_traj(self):
        if os.path.exists(self.traj_cache_path) and os.path.exists(self.temporal_mat_path):
            self.traj_list = pickle.load(open(self.traj_cache_path, 'rb'))
            _winfo('Loaded trajectory dataset from {}'.format(self.traj_cache_path))
            self.temporal_mat_list = pickle.load(open(self.temporal_mat_path, 'rb'))
            _winfo('Loaded temporal matrix from {}'.format(self.temporal_mat_path))
        else:
            data = pd.read_csv(self.traj_data_path, sep=';')
            self.traj_list, self.temporal_mat_list = self.data_processing(data)
        '''
        if self.max_train_size is not None:
            self.traj_list = self.traj_list[:self.max_train_size]
            self.temporal_mat_list = self.temporal_mat_list[:self.max_train_size]
        '''

    def _load_traj_dict(self):
        if os.path.exists(self.traj_dict_cache_path) and os.path.exists(self.temporal_mat_dict_path):
            self.traj_dict = pickle.load(open(self.traj_dict_cache_path, 'rb'))
            _winfo(f'Loaded dictionary of trajectory dataset from {self.traj_dict_cache_path}')
            self.temporal_mat_dict = pickle.load(open(self.temporal_mat_dict_path, 'rb'))
            _winfo(f'Loaded dictionary of temporal matrix from {self.temporal_mat_dict_path}')
        else:
            self._load_traj()
            self.traj_dict = dict()
            self.temporal_mat_dict = dict()
            for traj, temporal_mat in zip(self.traj_list, self.temporal_mat_list):
                start_time_ind = (traj[0][1] - self.begin_timestamp) // self.time_intervals
                if start_time_ind not in self.traj_dict:
                    self.traj_dict[start_time_ind] = []
                    self.temporal_mat_dict[start_time_ind] = []
                self.traj_dict[start_time_ind].append(traj)
                self.temporal_mat_dict[start_time_ind].append(temporal_mat)
            _winfo(f'Saving trajectory dictionary at {self.traj_dict_cache_path}')
            pickle.dump(self.traj_dict, open(self.traj_dict_cache_path, 'wb'))
            _winfo(f'Saving temporal matrix dictionary at {self.temporal_mat_dict_path}')
            pickle.dump(self.temporal_mat_dict, open(self.temporal_mat_dict_path, 'wb'))

    def _load_data(self):
        if os.path.exists(self.traj_traf_cache_path):
            self.traj_traf_list = pickle.load(open(self.traj_traf_cache_path, 'rb'))
            _winfo(f'Loaded trajectory-traffic dataset from {self.traj_traf_cache_path}')
        else:
            self._load_traj_dict()
            self.traj_traf_list = []
            for ti in self.traj_dict:
                i = int(ti)
                if i < self.begin_ti or i - self.begin_ti >= len(self.traf_data):
                    _winfo(f"Existing {i} key in trajectory dictionary")
                    continue
                traf_X, traf_Y = self.traf_data[i - self.begin_ti]

                traj_num = len(self.traj_dict[i])
                num_traj_batches = traj_num // self.traj_batch_size
                for j in range(num_traj_batches):
                    traj = self.traj_dict[i][j * self.traj_batch_size: (j + 1) * self.traj_batch_size]
                    temporal_mat = self.temporal_mat_dict[i][j * self.traj_batch_size: (j + 1) * self.traj_batch_size]
                    self.traj_traf_list.append((traj, temporal_mat, traf_X, traf_Y))

                if traj_num <= num_traj_batches * self.traj_batch_size:
                    continue
                traj = self.traj_dict[i][num_traj_batches * self.traj_batch_size:]
                temporal_mat = self.temporal_mat_dict[i][num_traj_batches * self.traj_batch_size:]
                sample_traj_num = (num_traj_batches + 1) * self.traj_batch_size - traj_num
                for _ in range(sample_traj_num):
                    ind = random.randint(0, traj_num - 1)
                    traj.append(self.traj_dict[i][ind])
                    temporal_mat.append(self.temporal_mat_dict[i][ind])
                assert len(traj) == self.traj_batch_size
                self.traj_traf_list.append((traj, temporal_mat, traf_X, traf_Y))

            _winfo(f'Saving trajectory-traffic dataset at {self.traj_traf_cache_path}')
            pickle.dump(self.traj_traf_list, open(self.traj_traf_cache_path, 'wb'))

    def _cal_mat(self, tim_list):
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if self.temporal_index:
                    off = abs(i - j)
                else:
                    off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)

    def data_processing(self, origin_data, desc=None, cache_path=None, temporal_mat_path=None):
        _winfo('Processing dataset in TrajTrafDataset!')
        sub_data = origin_data[['path', 'tlist', 'usr_id', 'traj_id']]
        traj_list = []
        temporal_mat_list = []
        for i in tqdm(range(sub_data.shape[0]), desc=desc):
            traj = sub_data.iloc[i]
            loc_list = eval(traj['path'])
            tim_list = eval(traj['tlist'])
            usr_id = traj['usr_id']
            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.fromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.usr2index.get(usr_id, self.vocab.unk_index)] * len(new_loc_list)
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
            temporal_mat = self._cal_mat(tim_list)  # (seq_len, seq_len)
            temporal_mat_list.append(temporal_mat)
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks, usr_list]).transpose((1, 0))  # (seq_length, feat_dim)
            traj_list.append(traj_fea)
        if cache_path is None:
            cache_path = self.traj_cache_path
        if temporal_mat_path is None:
            temporal_mat_path = self.temporal_mat_path
        pickle.dump(traj_list, open(cache_path, 'wb'))
        pickle.dump(temporal_mat_list, open(temporal_mat_path, 'wb'))
        return traj_list, temporal_mat_list  # [loc, tim, mins, weeks, usr]

    def __len__(self):
        return len(self.traj_traf_list)

    def __getitem__(self, ind):
        """
        Args:
            ind: integer index of sample in dataset
        Returns:
            trajs_t: list of torch traj
            temporal_mats_t: list of torch temporal mat
            traf_X: (T, N, D)
            traf_Y: (1, N, D)
        """
        trajs, temporal_mats, traf_X, traf_Y = self.traj_traf_list[ind]
        trajs_t = [torch.LongTensor(traj) for traj in trajs]
        temporal_mats_t = [torch.LongTensor(temporal_mat) for temporal_mat in temporal_mats]
        return trajs_t, temporal_mats_t, torch.FloatTensor(traf_X), torch.FloatTensor(traf_Y)

class DRNTRLDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.dataset = config.get('dataset', '')
        self.cache_dataset = config.get('cache_dataset', True)
        self.batch_size = config.get('batch_size', 4)
        self.test_batch_size = config.get('test_batch_size', self.batch_size)
        self.num_workers = config.get('num_workers', 0)
        self.train_rate = config.get('train_rate', 0.6)
        self.part_train_rate = config.get("part_train_rate", 1)
        self.eval_rate = config.get('eval_rate', 0.2)
        self.bidir = config.get('bidir', False)
        self.device = config.get('device', torch.device('cpu'))
        self.time_intervals = config.get('time_intervals', 1800)
        begin_time = config.get('begin_time', '2018-10-01 00:00:00')
        self.begin_timestamp = time.mktime(time.strptime(begin_time, "%Y-%m-%d %H:%M:%S"))
        
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = f'./raw_data/{self.dataset}/'
        if not os.path.exists(self.data_path):
            raise ValueError(f"Dataset {self.dataset} not exist! Please ensure the path {self.data_path} exist!")
        self.shuffle = config.get("shuffle", True)

        ## Traffic
        self.pad_with_last_sample = config.get('pad_with_last_sample', True)
        self.scaler_type = config.get('scaler', 'standard')
        self.ext_scaler_type = config.get('ext_scaler', 'none')
        self.load_external = config.get('load_external', True)
        self.normal_external = config.get('normal_external', False)
        self.add_time_in_day = config.get('add_time_in_day', True)
        self.add_day_in_week = config.get('add_day_in_week', True)
        self.input_window = config.get('input_window', 6)
        self.output_window = config.get('output_window', 1)
        self.data_col = config.get('data_col', '')
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.part_train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample) + '_' + str("".join(self.data_col))
        self.traf_cache_file_name = os.path.join('./libcity/cache/dataset_cache/', f'traf_{self.parameters_str}.npz')
        self.weight_col = config.get('weight_col', '')
        self.ext_col = config.get('ext_col', '')
        self.geo_file = config.get('geo_file', self.dataset)
        self.rel_file = config.get('rel_file', self.dataset)
        self.data_files = config.get('data_files', self.dataset)
        self.ext_file = config.get('ext_file', self.dataset)
        self.output_dim = config.get('output_dim', 1)
        self.init_weight_inf_or_zero = config.get('init_weight_inf_or_zero', 'zero')
        self.set_weight_link_or_dist = config.get('set_weight_link_or_dist', 'link')
        self.calculate_weight_adj = config.get('calculate_weight_adj', False)
        self.weight_adj_epsilon = config.get('weight_adj_epsilon', 0.1)
        self.points_per_hour = 3600 // self.time_intervals
        self.sem_neighbor_num = config.get("sem_neighbor_num", 20)

        ## Trajectory
        self.max_train_size = config.get('max_train_size', None)
        self.vocab_path = config.get('vocab_path', None)
        self.min_freq = config.get('min_freq', 1)
        self.merge = config.get('merge', True)
        if self.vocab_path is None:
            self.vocab_path = 'raw_data/{0}/{0}_vocab.pkl'.format(self.dataset)
        self.seq_len = config.get('seq_len', 64)
        self.add_cls = config.get('add_cls', True)
        self._load_vocab()
        self.collate_fn = None
        self.add_degree = config.get('add_degree', False)
        self.neighbors_K = config.get('neighbors_K', 3)
        self.load_trans_prob = config.get('load_trans_prob', True)
        self.load_t_trans_prob = config.get('load_t_trans_prob', True)
        self.normal_feature = config.get('normal_feature', False)
        self.temporal_index = config.get('temporal_index', False)
        self.traj_batch_size = config.get('traj_batch_size', 64)
        self.test_traj_batch_size = config.get('test_traj_batch_size', self.traj_batch_size)

        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self.geo_file = self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.rel'):
            self._load_rel()
        else:
            self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        self.node_features, self.node_fea_dim = self._load_geo_feature(self.geo_file)
        self.traj_edge_index, self.traj_loc_trans_prob, self.traj_t_loc_trans_prob, self.traf_edge_index, self.traf_loc_trans_prob = self._load_k_neighbors_and_trans_prob()
        self._load_g_edge_index()
        self.co_traj_edge_index, self.co_traj_loc_trans_prob, self.co_traf_edge_index, self.co_traf_loc_trans_prob = self._load_k_neighbors_and_trans_prob_co()

    def _load_vocab(self):
        assert os.path.exists(self.vocab_path), 'Vocab at %s not found' % self.vocab_path
        _winfo("Loading Vocab from {}".format(self.vocab_path))
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.usr_num = self.vocab.user_num
        self.vocab_size = self.vocab.vocab_size
        _winfo(f'vocab_path={self.vocab_path}, usr_num={self.usr_num}, vocab_size={self.vocab_size}')

    def _load_geo(self):
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        # 统一使用WordVocab中的loc2index
        self.geo_to_ind = {geo: ind - 5 for geo, ind in self.vocab.loc2index.items()}
        self.ind_to_geo = self.vocab.index2loc[5:]
        _winfo(f"Loaded file {self.geo_file}.geo, num_nodes={str(len(self.geo_ids))}")
        return geofile

    def _load_geo_feature(self, road_info):
        node_fea_path = self.data_path + f'{self.dataset}_node_features.npy'
        if self.add_degree:
            node_fea_path = node_fea_path[:-4] + '_degree.npy'
        if os.path.exists(node_fea_path):
            node_features = np.load(node_fea_path, allow_pickle=True)
        else:
            useful = ['highway', 'lanes', 'length', 'maxspeed']
            if self.add_degree:
                useful += ['outdegree', 'indegree']
            node_features = road_info[useful]
            norm_list = ['length']
            for col in norm_list:
                d = node_features[col]
                min_ = d.min()
                max_ = d.max()
                dnew = (d - min_) / (max_ - min_)
                node_features = node_features.drop(col, 1)
                node_features.insert(useful.index(col), col, dnew)
            onehot_list = ['lanes', 'maxspeed', 'highway']
            if self.add_degree:
                onehot_list += ['outdegree', 'indegree']
            for col in onehot_list:
                dum_col = pd.get_dummies(node_features[col], col)
                node_features = node_features.drop(col, axis=1)
                node_features = pd.concat([node_features, dum_col], axis=1)
            node_features = node_features.values
            np.save(node_fea_path, node_features)

        _winfo(f"node_features: {str(node_features.shape)}")  # (N, fea_dim)
        node_fea_vec = np.zeros((self.vocab.vocab_size, node_features.shape[1]))
        for ind in range(min(len(node_features), len(self.ind_to_geo))):
            geo_id = self.ind_to_geo[ind]
            if geo_id in self.vocab.loc2index:
                encoded_geo_id = self.vocab.loc2index[geo_id]
                node_fea_vec[encoded_geo_id] = node_features[ind]
            else:
                _wwarn(f"geo_id {geo_id} not found in vocab.loc2index")
        if self.normal_feature:
            _winfo('node_features by a/row_sum(a)')  # (vocab_size, fea_dim)
            row_sum = np.clip(node_fea_vec.sum(1), a_min=1, a_max=None)
            for i in range(len(node_fea_vec)):
                node_fea_vec[i, :] = node_fea_vec[i, :] / row_sum[i]
        node_fea_pe = torch.from_numpy(node_fea_vec).float().to(self.device)  # (vocab_size, fea_dim)
        _winfo(f'node_features_encoded: {str(node_fea_pe.shape)}')  # (vocab_size, fea_dim)
        return node_fea_pe, node_fea_pe.shape[1]

    def _load_k_neighbors_and_trans_prob_co(self):
        co_traj_source_nodes_ids, co_traj_target_nodes_ids = [], []
        co_traf_source_nodes_ids, co_traf_target_nodes_ids = [], []

        co_traj_seen_edges = set()
        co_traf_seen_edges = set()

        co_traj_dist = []
        co_traf_dist = []

        reach_sec = self.config.get('reach_sec', 30)
        reach_file = f"{self.dataset}_reachable_{reach_sec}"
        reachfile = pd.read_csv(f"{self.data_path}{reach_file}.rel")
        reach_df = reachfile[~reachfile[self.weight_col].isna()][['origin_id', 'destination_id', self.weight_col]]
        spd_mx = np.load(f"{self.data_path}{self.dataset}_shortest.npz")["result"]
        for row in reach_df.values:
            src_geo, src_ind = row[0], self.geo_to_ind[row[0]]
            trg_geo, trg_ind = row[1], self.geo_to_ind[row[1]]
            spd = spd_mx[src_ind][trg_ind]
            log_spd = 1 / math.log(math.e + spd)
            co_traf_src_node = self.vocab.loc2index[src_geo]
            co_traf_trg_node = self.geo_to_ind[trg_geo]
            co_traj_src_node = self.geo_to_ind[src_geo]
            co_traj_trg_node = self.vocab.loc2index[trg_geo]
            if (co_traf_src_node, co_traf_trg_node) not in co_traf_seen_edges:
                co_traf_source_nodes_ids.append(co_traf_src_node)
                co_traf_target_nodes_ids.append(co_traf_trg_node)
                co_traj_source_nodes_ids.append(co_traj_src_node)
                co_traj_target_nodes_ids.append(co_traj_trg_node)
                co_traf_seen_edges.add((co_traf_src_node, co_traf_trg_node))
                co_traj_seen_edges.add((co_traj_src_node, co_traj_trg_node))
                co_traf_dist.append(log_spd)
                co_traj_dist.append(log_spd)
        # add_self_edge
        for i in range(5, self.vocab_size):
            src = self.geo_to_ind[self.vocab.index2loc[i]]
            if (src, i) not in co_traj_seen_edges:
                co_traj_source_nodes_ids.append(src)
                co_traj_target_nodes_ids.append(i)
                co_traj_seen_edges.add((src, i))
                co_traj_dist.append(1.0)
        for i in range(self.num_nodes):
            src = self.vocab.loc2index[self.ind_to_geo[i]]
            if (src, i) not in co_traf_seen_edges:
                co_traf_source_nodes_ids.append(src)
                co_traf_target_nodes_ids.append(i)
                co_traf_seen_edges.add((src, i))
                co_traf_dist.append(1.0)
        # shape = (2, E), where E is the number of edges in the graph
        co_traj_edge_index = torch.from_numpy(np.row_stack((co_traj_source_nodes_ids, co_traj_target_nodes_ids))).long().to(self.device)
        _winfo(f'Co trajectory edge_index: {str(co_traj_edge_index.shape)}')
        co_traf_edge_index = torch.from_numpy(np.row_stack((co_traf_source_nodes_ids, co_traf_target_nodes_ids))).long().to(self.device)
        _winfo(f'Co traffic edge_index: {str(co_traf_edge_index.shape)}')
        co_traj_dist = torch.from_numpy(np.array(co_traj_dist)).unsqueeze(1).float().to(self.device)  # (E, 1)
        _winfo(f'Co Trajectory distance shape={co_traj_dist.shape}')
        co_traf_dist = torch.from_numpy(np.array(co_traf_dist)).unsqueeze(1).float().to(self.device)  # (E, 1)
        _winfo(f'Co Traffic distance shape={co_traf_dist.shape}')
        return co_traj_edge_index, co_traj_dist, co_traf_edge_index, co_traf_dist

    def _get_dtw(self):
        # cache_path = self.data_path + 'dtw_' + self.dataset + '.npy'
        cache_path = self.data_path + 'cosine_' + self.dataset + '.npy'
        if not os.path.exists(cache_path):
            for ind, filename in enumerate(self.data_files):
                if ind == 0:
                    df = self._load_dyna(filename)
                else:
                    df = np.concatenate((df, self._load_dyna(filename)), axis=0)
            if not os.path.exists(cache_path):
                data_mean = np.mean(
                    [df[24 * self.points_per_hour * i: 24 * self.points_per_hour * (i + 1)]
                    for i in range(int(df.shape[0] * self.train_rate) // (24 * self.points_per_hour))], axis=0)
                dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
                for i in tqdm(range(self.num_nodes)):
                    for j in range(i, self.num_nodes):
                        # dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
                        dtw_distance[i][j] = cosine_similarity(data_mean[:, i, :].reshape(1, -1), data_mean[:, j, :].reshape(1, -1))[0][0]
                for i in range(self.num_nodes):
                    for j in range(i):
                        dtw_distance[i][j] = dtw_distance[j][i]
                np.save(cache_path, dtw_distance)
        dtw_matrix = np.load(cache_path, allow_pickle=True)
        _winfo('Load DTW matrix from {}'.format(cache_path))
        return dtw_matrix

    def _load_g_edge_index(self):
        reachable_hop = self.config.get('reachable_hop', 3)
        relfile_path = f"{self.data_path}{self.rel_file}_reachable_{reachable_hop}_hop.rel"
        _winfo(f"Loading {self.data_path}{self.rel_file}_reachable_{reachable_hop}_hop.rel...")
        relfile = pd.read_csv(relfile_path)
        distance_df = relfile[~relfile[self.weight_col].isna()][['origin_id', 'destination_id', 'weight']]
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()
        for row in distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            source_node, target_node = self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]
            if (source_node, target_node) not in seen_edges:
                source_nodes_ids.append(source_node)
                target_nodes_ids.append(target_node)
                seen_edges.add((source_node, target_node))
                if self.bidir and (target_node, source_node) not in seen_edges:
                    source_nodes_ids.append(target_node)
                    target_nodes_ids.append(source_node)
                    seen_edges.add((target_node, source_node))
        
        dtw_matrix = self._get_dtw()
        dtw_top_indices = np.argsort(dtw_matrix, axis=1)[:, -self.sem_neighbor_num:]
        for i in range(dtw_top_indices.shape[0]):
            for j in range(dtw_top_indices.shape[1]):
                if (dtw_top_indices[i][j], i) not in seen_edges:
                    source_nodes_ids.append(dtw_top_indices[i][j])
                    target_nodes_ids.append(i)
                    seen_edges.add((dtw_top_indices[i][j], i))
                    if self.bidir and (i, dtw_top_indices[i][j]) not in seen_edges:
                        source_nodes_ids.append(i)
                        target_nodes_ids.append(dtw_top_indices[i][j])
                        seen_edges.add((i, dtw_top_indices[i][j]))

        # add self edge
        for i in range(len(self.geo_ids)):
            if (i, i) not in seen_edges:
                source_nodes_ids.append(i)
                target_nodes_ids.append(i)
                seen_edges.add((i, i))

        self.traf_g_edge_index = torch.from_numpy(np.row_stack((source_nodes_ids, target_nodes_ids))).long().to(self.device)
        _winfo('global edge_index: ' + str(self.traf_g_edge_index.shape))
        self.traf_g_loc_trans_prob = torch.ones(self.traf_g_edge_index.shape[1], 1).float().to(self.device)
        _winfo('global loc_trans_prob: ' + str(self.traf_g_loc_trans_prob.shape))

    def _load_k_neighbors_and_trans_prob(self):
        traj_source_nodes_ids, traj_target_nodes_ids = [], []
        traf_source_nodes_ids, traf_target_nodes_ids = [], []

        traj_seen_edges = set()
        traf_seen_edges = set()

        traj_loc_trans_prob = None
        traf_loc_trans_prob = None
        traj_t_loc_trans_prob = None

        geoid2neighbors = json.load(open(self.data_path + self.dataset + f'_neighbors_{self.neighbors_K}.json'))
        if self.load_trans_prob:
            traj_loc_trans_prob = []
            traf_loc_trans_prob = []
            link2prob = json.load(open(self.data_path + self.dataset + f'_trans_prob_{self.neighbors_K}.json'))
        if self.load_t_trans_prob:
            t_link2prob = json.load(open(self.data_path + self.dataset + f'_t_trans_prob_{self.neighbors_K}.json'))
            traj_t_loc_trans_prob = [[] for _ in range(len(t_link2prob))]
        for k, v in geoid2neighbors.items():
            traj_src_node = self.vocab.loc2index[int(k)]  # co_traf_src_node
            traf_src_node = self.geo_to_ind[int(k)]  # co_traj_src_node
            for tgt in v:
                traj_trg_node = self.vocab.loc2index[int(tgt)]  # co_traj_trg_node
                traf_trg_node = self.geo_to_ind[int(tgt)]  # co_traf_trg_node
                if (traj_src_node, traj_trg_node) not in traj_seen_edges:
                    traj_source_nodes_ids.append(traj_src_node)
                    traj_target_nodes_ids.append(traj_trg_node)
                    traf_source_nodes_ids.append(traf_src_node)
                    traf_target_nodes_ids.append(traf_trg_node)
                    traj_seen_edges.add((traj_src_node, traj_trg_node))
                    traf_seen_edges.add((traf_src_node, traf_trg_node))
                    if self.load_trans_prob:
                        traj_loc_trans_prob.append(link2prob[str(k) + '_' + str(tgt)])
                        traf_loc_trans_prob.append(link2prob[str(k) + '_' + str(tgt)])
                    if self.load_t_trans_prob:
                        for j in range(len(t_link2prob)):
                            traj_t_loc_trans_prob[j].append(t_link2prob[str(j)][str(k) + '_' + str(tgt)])
        # add_self_edge
        for i in range(self.vocab.vocab_size):
            if (i, i) not in traj_seen_edges:
                traj_source_nodes_ids.append(i)
                traj_target_nodes_ids.append(i)
                traj_seen_edges.add((i, i))
                if self.load_trans_prob:
                    traj_loc_trans_prob.append(link2prob.get(str(i) + '_' + str(i), 0.0))
                if self.load_t_trans_prob:
                    for j in range(len(t_link2prob)):
                        traj_t_loc_trans_prob[j].append(t_link2prob[str(j)].get(str(i) + '_' + str(i), 0.0))
        for i in range(self.num_nodes):
            if (i, i) not in traf_seen_edges:
                traf_source_nodes_ids.append(i)
                traf_target_nodes_ids.append(i)
                traf_seen_edges.add((i, i))
                if self.load_trans_prob:
                    traf_loc_trans_prob.append(link2prob.get(str(i) + '_' + str(i), 0.0))
        # shape = (2, E), where E is the number of edges in the graph
        traj_edge_index = torch.from_numpy(np.row_stack((traj_source_nodes_ids, traj_target_nodes_ids))).long().to(self.device)
        _winfo(f'trajectory edge_index: {str(traj_edge_index.shape)}')
        traf_edge_index = torch.from_numpy(np.row_stack((traf_source_nodes_ids, traf_target_nodes_ids))).long().to(self.device)
        _winfo(f'traffic edge_index: {str(traf_edge_index.shape)}')
        if self.load_trans_prob:
            traj_loc_trans_prob = torch.from_numpy(np.array(traj_loc_trans_prob)).unsqueeze(1).float().to(self.device)  # (E, 1)
            _winfo(f'Trajectory loc-transfer prob shape={traj_loc_trans_prob.shape}')
            traf_loc_trans_prob = torch.from_numpy(np.array(traf_loc_trans_prob)).unsqueeze(1).float().to(self.device)  # (E, 1)
            _winfo(f'Traffic loc-transfer prob shape={traf_loc_trans_prob.shape}')
        if self.load_t_trans_prob:
            traj_t_loc_trans_prob = torch.from_numpy(np.array(traj_t_loc_trans_prob)).unsqueeze(2).float().to(self.device)  # (336, E, 1)
            _winfo(f'Trajectory dynamic loc-transfer prob shape={traj_t_loc_trans_prob.shape}')
        return traj_edge_index, traj_loc_trans_prob, traj_t_loc_trans_prob, traf_edge_index, traf_loc_trans_prob

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        _winfo(f'set_weight_link_or_dist: {self.set_weight_link_or_dist}')
        _winfo(f'init_weight_inf_or_zero: {self.init_weight_inf_or_zero}')
        if self.weight_col != '':
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][['origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) != 5:
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            else:
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][['origin_id', 'destination_id', self.weight_col]]
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                _winfo(f"{row[0]} or {row[1]} not in WordVocab!")
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        _winfo("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        # if self.calculate_weight_adj:
        #     self._calculate_adjacency_matrix()

    def _load_dyna(self, filename):
        _winfo(f"Loading file {filename}.dyna")
        dynafile = pd.read_csv(self.data_path + filename + '.dyna')
        if self.data_col != '':
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'entity_id')
            dynafile = dynafile[data_col]
        else:
            dynafile = dynafile[dynafile.columns[2:]]
        self.timeslots = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timeslots = dict()
        if not dynafile['time'].isna().any():
            self.timeslots = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timeslots))
            self.timeslots = np.array(self.timeslots, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timeslots):
                self.idx_of_timeslots[_ts] = idx
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timeslots)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i: i + len_time].values)
        data = np.array(data, dtype=np.float)
        data = data.swapaxes(0, 1)
        _winfo(f"Loaded file {filename}.dyna, shape={str(data.shape)}")
        return data

    def _add_external_information(self, df, ext_data=None):
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = np.isnan(self.timeslots).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timeslots - self.timeslots.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timeslots.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.tile(np.array(dayofweek), [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(day_in_week)
        if ext_data is not None:
            pass
        data = np.concatenate(data_list, axis=-1)
        return data

    def _generate_input_data(self, df):
        num_samples = df.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def _generate_data(self):
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):
            ext_data = self._load_ext()
        else:
            ext_data = None
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        _winfo("Dataset created")
        _winfo("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y

    def _split_train_val_test(self, x, y):
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        x_train, y_train = x[int(num_train*(1 - self.part_train_rate)):num_train], y[int(num_train*(1 - self.part_train_rate)):num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]
        _winfo("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        _winfo("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        _winfo("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.traf_cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
            )
            _winfo('Saved at ' + self.traf_cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        x, y = self._generate_data()
        return self._split_train_val_test(x, y)

    def _load_cache_train_val_test(self):
        _winfo('Loading ' + self.traf_cache_file_name)
        cat_data = np.load(self.traf_cache_file_name, allow_pickle=True)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        _winfo("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        _winfo("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        _winfo("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _get_scalar(self, scaler_type, x_train, y_train):
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            _winfo('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            _winfo('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            _winfo('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            _winfo('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            _winfo('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            _winfo('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def _gen_dataset(self, traf_train_data, traf_eval_data, traf_test_data):
        traf_name = f"{self.input_window}_{self.scaler_type}_{self.load_external}_{self.add_time_in_day}_{self.add_day_in_week}"
        train_begin_ti = self.input_window
        train_dataset = TrajTrafDataset(
            data_name=self.dataset, traf_data=traf_train_data, traf_name=traf_name, data_type='train', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=self.max_train_size, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=train_begin_ti, time_intervals=self.time_intervals,
        )
        eval_begin_ti = train_begin_ti + len(traf_train_data)
        eval_dataset = TrajTrafDataset(
            data_name=self.dataset, traf_data=traf_eval_data, traf_name=traf_name, data_type='eval', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=eval_begin_ti, time_intervals=self.time_intervals,
        )
        test_begin_ti = eval_begin_ti + len(traf_eval_data)
        test_dataset = TrajTrafDataset(
            data_name=self.dataset, traf_data=traf_test_data, traf_name=traf_name, data_type='test', traj_batch_size=self.test_traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=test_begin_ti, time_intervals=self.time_intervals,
        )
        return train_dataset, eval_dataset, test_dataset

    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        assert self.collate_fn is not None
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
            collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls),
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
            collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls),
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls),
        )
        self.num_batches = len(train_dataloader)

        return train_dataloader, eval_dataloader, test_dataloader

    def get_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.cache_dataset and os.path.exists(self.traf_cache_file_name):
            x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type, x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type, x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        traf_train_data = list(zip(x_train, y_train))
        traf_eval_data = list(zip(x_val, y_val))
        traf_test_data = list(zip(x_test, y_test))
        _winfo(f"Size of traffic dataset:{str(len(traf_train_data))}/{str(len(traf_eval_data))}/{str(len(traf_test_data))}")
        train_dataset, eval_dataset, test_dataset = self._gen_dataset(traf_train_data, traf_eval_data, traf_test_data)

        _winfo("Creating Dataloader!")
        return self._gen_dataloader(train_dataset, eval_dataset, test_dataset)

    def get_data_feature(self):
        data_feature = {
            "usr_num": self.usr_num, "adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
            "vocab": self.vocab, "vocab_size": self.vocab_size,
            "geo_file": self.geo_file, "rel_file": self.rel_file,
            "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
            "node_features": self.node_features, "node_fea_dim": self.node_fea_dim,
            "traj_edge_index": self.traj_edge_index, "traj_loc_trans_prob": self.traj_loc_trans_prob, "traj_t_loc_trans_prob": self.traj_t_loc_trans_prob,
            "traf_edge_index": self.traf_edge_index, "traf_loc_trans_prob": self.traf_loc_trans_prob,
            "traf_g_edge_index": self.traf_g_edge_index, "traf_g_loc_trans_prob": self.traf_g_loc_trans_prob,
            "co_traj_edge_index": self.co_traj_edge_index, "co_traj_loc_trans_prob": self.co_traj_loc_trans_prob,
            "co_traf_edge_index": self.co_traf_edge_index, "co_traf_loc_trans_prob": self.co_traf_loc_trans_prob,
            "scaler": self.scaler, "ext_dim": self.ext_dim, "feature_dim": self.feature_dim,
            "output_dim": self.output_dim, "num_batches": self.num_batches,
        }
        return data_feature

class TrafDataset(Dataset):
    def __init__(self, traf_data):
        self.traf_data = traf_data

    def __getitem__(self, ind):
        traf_X, traf_Y = self.traf_data[ind]
        return torch.FloatTensor(traf_X), torch.FloatTensor(traf_Y)

    def __len__(self):
        return len(self.traf_data)


def collate_traf(data):
    traf_X, traf_Y = zip(*data)
    traf_X = torch.cat([t.unsqueeze(0) for t in traf_X], dim=0)
    traf_Y = torch.cat([t.unsqueeze(0) for t in traf_Y], dim=0)
    return traf_X, traf_Y


class DRNTRLTSPDataset(DRNTRLDataset):

    def __init__(self, config):
        super().__init__(config)

    def _gen_dataset(self, traf_train_data, traf_eval_data, traf_test_data):
        train_dataset = TrafDataset(traf_train_data)
        eval_dataset = TrafDataset(traf_eval_data)
        test_dataset = TrafDataset(traf_test_data)
        return train_dataset, eval_dataset, test_dataset

    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, collate_fn=collate_traf,
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, collate_fn=collate_traf,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers,
            shuffle=False, collate_fn=collate_traf,
        )
        self.num_batches = len(train_dataloader)

        return train_dataloader, eval_dataloader, test_dataloader
