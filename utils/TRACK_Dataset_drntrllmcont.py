import os
import torch
import random
import pickle
import pandas as pd
from TRACK_Dataset_drntrllm import TrajTrafDatasetLM, DRNTRLLMDataset, noise_mask, padding_mask


def _inner_slove_data(trajs, batch_size, max_len):
    traj_batch_size = len(trajs[0])
    feat_dim = trajs[0][0].shape[-1]
    traj = torch.zeros(batch_size, traj_batch_size, max_len, feat_dim, dtype=torch.long)
    padding_masks = []

    for j in range(batch_size):
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [traj.shape[0] for traj in trajs[j]]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)

        for i in range(traj_batch_size):
            end = min(lengths[i], max_len)
            trajs[j][i][:end, 1] = trajs[j][i][:end, 1] - trajs[j][i][0, 1]  # Absolute time to relative time
            traj[j, i, :end, :] = trajs[j][i][:end, :]

        padding_masks_j = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
        padding_masks.append(padding_masks_j)

    padding_masks = torch.cat([p.unsqueeze(0) for p in padding_masks], dim=0)
    return traj, padding_masks


def collate_unsuperv_mask_cont(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    trajs, masks, trajs1, trajs2, traf_X, traf_Y = zip(*data)

    traj_batch_size = len(trajs[0])
    feat_dim = trajs[0][0].shape[-1]
    traj = torch.zeros(batch_size, traj_batch_size, max_len, feat_dim, dtype=torch.long)
    target_masks = torch.zeros_like(traj, dtype=torch.bool)  # masks related to objective
    padding_masks = []
    targets = []

    for j in range(batch_size):
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [traj.shape[0] for traj in trajs[j]]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)

        for i in range(traj_batch_size):
            end = min(lengths[i], max_len)
            trajs[j][i][:end, 1] = trajs[j][i][:end, 1] - trajs[j][i][0, 1]  # Absolute time to relative time
            traj[j, i, :end, :] = trajs[j][i][:end, :]
            target_masks[j, i, :end, :] = masks[j][i][:end, :]

        padding_masks_j = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
        padding_masks.append(padding_masks_j)

        target_masks[j] = ~target_masks[j]
        target_masks[j] = target_masks[j] * padding_masks_j.unsqueeze(-1)

        targets_j = traj[j].clone()
        targets_j.masked_fill_(target_masks[j] == 0, vocab.pad_index)
        targets.append(targets_j)

        traj[j, :, :, 0:1].masked_fill_(target_masks[j, :, :, 0:1] == 1, vocab.mask_index)  # loc -> mask_index
        traj[j, :, :, 1:].masked_fill_(target_masks[j, :, :, 1:] == 1, vocab.pad_index)  # others -> pad_index
    targets = torch.cat([t.unsqueeze(0) for t in targets], dim=0).long()
    padding_masks = torch.cat([p.unsqueeze(0) for p in padding_masks], dim=0)

    traj1, padding_masks1 = _inner_slove_data(trajs1, batch_size, max_len)
    traj2, padding_masks2 = _inner_slove_data(trajs2, batch_size, max_len)

    traf_X = torch.cat([t.unsqueeze(0) for t in traf_X], dim=0)
    traf_Y = torch.cat([t.unsqueeze(0) for t in traf_Y], dim=0)
    return traj.long(), targets, target_masks, padding_masks,\
        traj1.long(), padding_masks1, traj2.long(), padding_masks2, traf_X, traf_Y


class TrajTrafDatasetLMCont(TrajTrafDatasetLM):

    def __init__(
        self, data_name, traf_data, traf_name, data_type, traj_batch_size, vocab,
        add_cls=True, max_train_size=None, temporal_index=False,
        begin_timestamp=None, begin_ti=0, time_intervals=1800,
        masking_ratio=0.15, masking_mode='together', distribution='random', avg_mask_len=2,
        data_augment1='trim', data_augment2='shift',
    ):
        temporal_mat_post = f"_temporal_mat{'_index' if temporal_index else ''}.pkl"
        self.data_augment1 = data_augment1
        aug_post1 = f"_enhancedby{data_augment1}" if data_augment1 else ""
        self.traj_data_path1 = 'raw_data/{0}/{0}_traj_{1}{2}.csv'.format(data_name, data_type, aug_post1)
        self.traj_cache_path1 = 'raw_data/{0}/cache_{0}_traj_{1}_{2}{3}.pkl'.format(data_name, data_type, add_cls, aug_post1)
        self.traj_dict_cache_path1 = 'raw_data/{0}/cache_{0}_traj_dict_{1}_{2}_{3}{4}.pkl'.format(data_name, data_type, add_cls, time_intervals, aug_post1)
        self.temporal_mat_path1 = self.traj_cache_path1[:-4] + temporal_mat_post

        self.data_augment2 = data_augment2
        aug_post2 = f"_enhancedby{data_augment2}" if data_augment2 else ""
        self.traj_data_path2 = 'raw_data/{0}/{0}_traj_{1}{2}.csv'.format(data_name, data_type, aug_post2)
        self.traj_cache_path2 = 'raw_data/{0}/cache_{0}_traj_{1}_{2}{3}.pkl'.format(data_name, data_type, add_cls, aug_post2)
        self.traj_dict_cache_path2 = 'raw_data/{0}/cache_{0}_traj_dict_{1}_{2}_{3}{4}.pkl'.format(data_name, data_type, add_cls, time_intervals, aug_post2)
        self.temporal_mat_path2 = self.traj_cache_path2[:-4] + temporal_mat_post

        self.aug_traj_traf_cache_path = 'raw_data/{0}/cache_{0}_traj_traf_{1}_{2}_{3}_{4}_{5}{6}{7}.pkl'.format(
            data_name, data_type, add_cls, time_intervals, traj_batch_size, traf_name, aug_post1, aug_post2,
        )

        super().__init__(
            data_name, traf_data, traf_name, data_type, traj_batch_size, vocab,
            add_cls, max_train_size, temporal_index,
            begin_timestamp, begin_ti, time_intervals,
            masking_ratio, masking_mode, distribution, avg_mask_len,
        )

    def _load_aug_traj(self, traj_cache_path, temporal_mat_path, traj_data_path):
        if os.path.exists(traj_cache_path):
            traj_list = pickle.load(open(traj_cache_path, 'rb'))
            self._logger.info('Loaded trajectory dataset from {}'.format(traj_cache_path))
            # temporal_mat_list = pickle.load(open(temporal_mat_path, 'rb'))
            # self._logger.info('Loaded temporal matrix from {}'.format(temporal_mat_path))
            return traj_list
        else:
            data = pd.read_csv(traj_data_path, sep=';')
            traj_list, temporal_mat_list = self.data_processing(data, cache_path=traj_cache_path, temporal_mat_path=temporal_mat_path)
            return traj_list

    def _load_aug_traj_dict(self, traj_dict_cache_path, traj_cache_path, temporal_mat_path, traj_data_path):
        if os.path.exists(traj_dict_cache_path):
            traj_dict = pickle.load(open(traj_dict_cache_path, 'rb'))
            self._logger.info(f'Loaded dictionary of trajectory dataset from {traj_dict_cache_path}')
            return traj_dict
        else:
            traj_list = self._load_aug_traj(traj_cache_path, temporal_mat_path, traj_data_path)
            traj_dict = dict()
            for traj in traj_list:
                start_time_ind = (traj[0][1] - self.begin_timestamp) // self.time_intervals
                if start_time_ind not in traj_dict:
                    traj_dict[start_time_ind] = []
                traj_dict[start_time_ind].append(traj)
            self._logger.info(f'Saving trajectory dictionary at {traj_dict_cache_path}')
            pickle.dump(traj_dict, open(traj_dict_cache_path, 'wb'))
            return traj_dict

    def _load_data(self):
        if os.path.exists(self.aug_traj_traf_cache_path):
            self.traj_traf_list = pickle.load(open(self.aug_traj_traf_cache_path, 'rb'))
            self._logger.info(f'Loaded trajectory-traffic dataset from {self.aug_traj_traf_cache_path}')
        else:
            self._load_traj_dict()
            traj_dict1 = self._load_aug_traj_dict(self.traj_dict_cache_path1, self.traj_cache_path1, self.temporal_mat_path1, self.traj_data_path1)
            traj_dict2 = self._load_aug_traj_dict(self.traj_dict_cache_path2, self.traj_cache_path2, self.temporal_mat_path2, self.traj_data_path2)
            self.traj_traf_list = []
            for ti in self.traj_dict:
                i = int(ti)
                if i < self.begin_ti or i - self.begin_ti >= len(self.traf_data):
                    self._logger.info(f"Existing {i} key in trajectory dictionary")
                    continue
                traf_X, traf_Y = self.traf_data[i - self.begin_ti]

                traj_num = len(self.traj_dict[i])
                assert traj_num == len(traj_dict1[i]) == len(traj_dict2[i]), f"{i}:{traj_num}, {len(traj_dict1[i])}, {len(traj_dict2[i])}"
                num_traj_batches = traj_num // self.traj_batch_size
                for j in range(num_traj_batches):
                    traj = self.traj_dict[i][j * self.traj_batch_size: (j + 1) * self.traj_batch_size]
                    traj1 = traj_dict1[i][j * self.traj_batch_size: (j + 1) * self.traj_batch_size]
                    traj2 = traj_dict2[i][j * self.traj_batch_size: (j + 1) * self.traj_batch_size]
                    self.traj_traf_list.append((traj, traj1, traj2, traf_X, traf_Y))

                if traj_num <= num_traj_batches * self.traj_batch_size:
                    continue
                traj = self.traj_dict[i][num_traj_batches * self.traj_batch_size:]
                traj1 = traj_dict1[i][num_traj_batches * self.traj_batch_size:]
                traj2 = traj_dict2[i][num_traj_batches * self.traj_batch_size:]
                sample_traj_num = (num_traj_batches + 1) * self.traj_batch_size - traj_num
                for _ in range(sample_traj_num):
                    ind = random.randint(0, traj_num - 1)
                    traj.append(self.traj_dict[i][ind])
                    traj1.append(traj_dict1[i][ind])
                    traj2.append(traj_dict2[i][ind])
                assert len(traj) == self.traj_batch_size
                self.traj_traf_list.append((traj, traj1, traj2, traf_X, traf_Y))

            self._logger.info(f'Saving trajectory-traffic dataset at {self.aug_traj_traf_cache_path}')
            pickle.dump(self.traj_traf_list, open(self.aug_traj_traf_cache_path, 'wb'))

    def __getitem__(self, ind):
        """
        Args:
            ind: integer index of sample in dataset
        Returns:
            traj: list of torch traj
            mask: list of torch mask
            traj1: list of torch augmented traj by data_augment1
            traj2: list of torch augmented traj by data_augment2
            traf_X: (T, N, D)
            traf_Y: (1, N, D)
        """
        trajs, trajs1, trajs2, traf_X, traf_Y = self.traj_traf_list[ind]
        masks = []
        for traj in trajs:
            mask = noise_mask(traj, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution, self.exclude_feats, self.add_cls)
            masks.append(torch.BoolTensor(mask))
        trajs_t = [torch.LongTensor(traj) for traj in trajs]
        trajs_t1 = [torch.LongTensor(traj) for traj in trajs1]
        trajs_t2 = [torch.LongTensor(traj) for traj in trajs2]
        return trajs_t, masks, trajs_t1, trajs_t2, torch.FloatTensor(traf_X), torch.FloatTensor(traf_Y)


class DRNTRLLMContDataset(DRNTRLLMDataset):

    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_unsuperv_mask_cont
        self.data_augment1 = config.get("data_augment1", 'trim')
        self.data_augment2 = config.get("data_augment2", 'shift')

    def _gen_dataset(self, traf_train_data, traf_eval_data, traf_test_data):
        traf_name = f"{self.scaler_type}_{self.load_external}_{self.add_time_in_day}_{self.add_day_in_week}"
        train_begin_ti = self.input_window
        train_dataset = TrajTrafDatasetLMCont(
            data_name=self.dataset, traf_data=traf_train_data, traf_name=traf_name, data_type='train', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=self.max_train_size, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=train_begin_ti, time_intervals=self.time_intervals,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            data_augment1=self.data_augment1, data_augment2=self.data_augment2,
        )
        eval_begin_ti = train_begin_ti + len(traf_train_data)
        eval_dataset = TrajTrafDatasetLMCont(
            data_name=self.dataset, traf_data=traf_eval_data, traf_name=traf_name, data_type='eval', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=eval_begin_ti, time_intervals=self.time_intervals,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            data_augment1=self.data_augment1, data_augment2=self.data_augment2,
        )
        test_begin_ti = eval_begin_ti + len(traf_eval_data)
        test_dataset = TrajTrafDatasetLMCont(
            data_name=self.dataset, traf_data=traf_test_data, traf_name=traf_name, data_type='test', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=test_begin_ti, time_intervals=self.time_intervals,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            data_augment1=self.data_augment1, data_augment2=self.data_augment2,
        )
        return train_dataset, eval_dataset, test_dataset
