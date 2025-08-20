import torch
import numpy as np
from utils.TRACK_Dataset_trl import TRLDataset, TrajDataset


def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def noise_mask(X, masking_ratio, lm=2, mode='together', distribution='geometric', exclude_feats=None, add_cls=True):
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])
    else:
        mask = np.ones(X.shape, dtype=bool)
    if add_cls:
        mask[0] = True  # CLS at 0, set mask=1
    return mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)  1是正常,0表示Padding的位置
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1)))


def collate_unsuperv_mask(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, masks, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)

    # masks related to objective
    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        features[i][:end, 1] = features[i][:end, 1] - features[i][0, 1]
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    targets = X.clone()
    targets = targets.masked_fill_(target_masks == 0, vocab.pad_index)

    X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
    X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    return X.long(), targets.long(), target_masks, padding_masks


class TrajDatasetLM(TrajDataset):

    def __init__(
        self, data_name, data_type, vocab,
        add_cls=False, max_train_size=None, temporal_index=False,
        masking_ratio=0.15, masking_mode='together', distribution='random', avg_mask_len=2,
    ):
        super().__init__(
            data_name, data_type, vocab,
            add_cls=add_cls, max_train_size=max_train_size, temporal_index=temporal_index,
        )
        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.distribution = distribution
        self.avg_mask_len = avg_mask_len
        self.exclude_feats = None

    def __getitem__(self, ind):
        traj_ind = self.traj_list[ind]
        temporal_mat = self.temporal_mat_list[ind]
        mask = noise_mask(traj_ind, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution, self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        return torch.LongTensor(traj_ind), torch.LongTensor(mask), torch.LongTensor(temporal_mat)


class TRLLMDataset(TRLDataset):

    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_unsuperv_mask
        self.masking_ratio = config.get('masking_ratio', 0.15)
        self.masking_mode = config.get('masking_mode', 'together')
        self.distribution = config.get('distribution', 'geometric')
        self.avg_mask_len = config.get('avg_mask_len', 2)

    def _gen_dataset(self):
        train_dataset = TrajDatasetLM(
            data_name=self.dataset, data_type='train', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=self.max_train_size, temporal_index=self.temporal_index,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
        )
        eval_dataset = TrajDatasetLM(
            data_name=self.dataset, data_type='eval', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
        )
        test_dataset = TrajDatasetLM(
            data_name=self.dataset, data_type='test', vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
        )
        return train_dataset, eval_dataset, test_dataset
