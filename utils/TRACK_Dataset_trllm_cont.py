import os
import torch
import pickle
import pandas as pd
from utils.TRACK_Dataset_trllm import TRLLMDataset, TrajDatasetLM, collate_unsuperv_mask, padding_mask


def _inner_slove_data(features, batch_size, max_len, vocab=None):
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        features[i][:end, 1] = features[i][:end, 1] - features[i][0, 1]
        X[i, :end, :] = features[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    return X, padding_masks


def collate_unsuperv_cont(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features1, features2 = zip(*data)  # list of (seq_len, feat_dim)
    X1, padding_masks1 = _inner_slove_data(features1, batch_size, max_len, vocab)
    X2, padding_masks2 = _inner_slove_data(features2, batch_size, max_len, vocab)
    return X1.long(), X2.long(), padding_masks1, padding_masks2


def collate_unsuperv_mask_cont(data, max_len=None, vocab=None, add_cls=True):
    features, masks, temporal_mat, features1, features2 = zip(*data)
    data_for_mask = list(zip(features, masks, temporal_mat))
    dara_for_contra = list(zip(features1, features2))

    X1, X2, padding_masks1, padding_masks2 \
        = collate_unsuperv_cont(data=dara_for_contra, max_len=max_len, vocab=vocab, add_cls=add_cls)

    masked_x, targets, target_masks, padding_masks \
        = collate_unsuperv_mask(data=data_for_mask, max_len=max_len, vocab=vocab, add_cls=add_cls)
    return X1, X2, padding_masks1, padding_masks2, \
           masked_x, targets, target_masks, padding_masks


class TrajDatasetLMCont(TrajDatasetLM):

    def __init__(
        self, data_name, data_type, vocab,
        add_cls=True, max_train_size=None, temporal_index=False,
        masking_ratio=0.15, masking_mode='together', distribution='random', avg_mask_len=2,
        data_augment1='trim', data_augment2='shift',
    ):
        self.data_augment1 = data_augment1
        aug_post1 = f"_enhancedby{data_augment1}" if data_augment1 else ""
        self.data_path1 = 'raw_data/{0}/{0}_traj_{1}{2}.csv'.format(data_name, data_type, aug_post1)
        self.cache_path1 = 'raw_data/{0}/cache_{0}_traj_{1}_{2}{3}.pkl'.format(data_name, data_type, add_cls, aug_post1)

        self.data_augment2 = data_augment2
        aug_post2 = f"_enhancedby{data_augment2}" if data_augment2 else ""
        self.data_path2 = 'raw_data/{0}/{0}_traj_{1}{2}.csv'.format(data_name, data_type, aug_post2)
        self.cache_path2 = 'raw_data/{0}/cache_{0}_traj_{1}_{2}{3}.pkl'.format(data_name, data_type, add_cls, aug_post2)

        if temporal_index:
            self.temporal_mat_path1 = self.cache_path1[:-4] + '_temporal_mat_index.pkl'
            self.temporal_mat_path2 = self.cache_path2[:-4] + '_temporal_mat_index.pkl'
        else:
            self.temporal_mat_path1 = self.cache_path1[:-4] + '_temporal_mat.pkl'
            self.temporal_mat_path2 = self.cache_path2[:-4] + '_temporal_mat.pkl'

        super().__init__(
            data_name, data_type, vocab,
            add_cls, max_train_size, temporal_index,
            masking_ratio, masking_mode, distribution, avg_mask_len,
        )
        self._logger.info('Init TrajDatasetLMCont!')
        self._load_data_split()

    def _load_data_split(self):
        if os.path.exists(self.cache_path1) and os.path.exists(self.cache_path2):
            self.traj_list1 = pickle.load(open(self.cache_path1, 'rb'))
            # self.temporal_mat_list1 = pickle.load(open(self.temporal_mat_path1, 'rb'))
            self.traj_list2 = pickle.load(open(self.cache_path2, 'rb'))
            # self.temporal_mat_list2 = pickle.load(open(self.temporal_mat_path2, 'rb'))
            self._logger.info('Loaded dataset from {} and {}'.format(self.cache_path1, self.cache_path2))
        else:
            origin_data_df1 = pd.read_csv(self.data_path1, sep=';')
            origin_data_df2 = pd.read_csv(self.data_path2, sep=';')
            assert origin_data_df1.shape == origin_data_df2.shape
            self.traj_list1, self.temporal_mat_list1 = self.data_processing(
                origin_data_df1, cache_path=self.cache_path1, temporal_mat_path=self.temporal_mat_path1,
            )
            self.traj_list2, self.temporal_mat_list2 = self.data_processing(
                origin_data_df2, cache_path=self.cache_path2, temporal_mat_path=self.temporal_mat_path2,
            )

    def __len__(self):
        assert len(self.traj_list1) == len(self.traj_list2) == len(self.traj_list)
        return len(self.traj_list)

    def __getitem__(self, ind):
        traj_ind, mask, temporal_mat = super().__getitem__(ind)
        traj_ind1 = self.traj_list1[ind]  # (seq_len, feat_dim)
        traj_ind2 = self.traj_list2[ind]  # (seq_len, feat_dim)
        # temporal_mat1 = self.temporal_mat_list1[ind]  # (seq_len, seq_length)
        # temporal_mat2 = self.temporal_mat_list2[ind]  # (seq_len, seq_length)
        '''
        mask1 = None
        mask2 = None
        if 'mask' in self.data_augment1:
            mask1 = noise_mask(
                traj_ind1, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                self.exclude_feats, self.add_cls,
            )  # (seq_length, feat_dim) boolean array
        if 'mask' in self.data_augment2:
            mask2 = noise_mask(
                traj_ind2, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                self.exclude_feats, self.add_cls,
            )  # (seq_length, feat_dim) boolean array
        '''
        return traj_ind, mask, temporal_mat, \
               torch.LongTensor(traj_ind1), torch.LongTensor(traj_ind2),
            #    torch.LongTensor(temporal_mat1), torch.LongTensor(temporal_mat2), \
            #    torch.LongTensor(mask1) if mask1 else None, \
            #    torch.LongTensor(mask2) if mask2 else None


class TRLLMContDataset(TRLLMDataset):

    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_unsuperv_mask_cont
        self.data_augment1 = config.get("data_augment1", 'trim')
        self.data_augment2 = config.get("data_augment2", 'shift')

    def _gen_dataset(self):
        train_dataset = TrajDatasetLMCont(
            data_name=self.dataset, data_type='train', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=self.max_train_size, temporal_index=self.temporal_index,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            data_augment1=self.data_augment1, data_augment2=self.data_augment2,
        )
        eval_dataset = TrajDatasetLMCont(
            data_name=self.dataset, data_type='eval', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            data_augment1=self.data_augment1, data_augment2=self.data_augment2,
        )
        test_dataset = TrajDatasetLMCont(
            data_name=self.dataset, data_type='test', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            data_augment1=self.data_augment1, data_augment2=self.data_augment2,
        )
        return train_dataset, eval_dataset, test_dataset
