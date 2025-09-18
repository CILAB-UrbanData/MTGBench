import torch
import numpy as np
from TRACK_Dataset_drntrl import DRNTRLDataset, TrajTrafDataset


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
    trajs, temporal_mats, masks, traf_X, traf_Y = zip(*data)

    traj_batch_size = len(trajs[0])
    feat_dim = trajs[0][0].shape[-1]
    traj = torch.zeros(batch_size, traj_batch_size, max_len, feat_dim, dtype=torch.long)
    temporal_mat = torch.zeros(batch_size, traj_batch_size, max_len, max_len, dtype=torch.long)
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
            temporal_mat[j, i, :end, :end] = temporal_mats[j][i][:end, :end]
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
    traf_X = torch.cat([t.unsqueeze(0) for t in traf_X], dim=0)
    traf_Y = torch.cat([t.unsqueeze(0) for t in traf_Y], dim=0)
    return traj.long(), targets, target_masks, padding_masks, temporal_mat.long(), traf_X, traf_Y


class TrajTrafDatasetLM(TrajTrafDataset):

    def __init__(
        self, data_name, traf_data, traf_name, data_type, traj_batch_size, vocab,
        add_cls=False, max_train_size=None, temporal_index=False,
        begin_timestamp=None, begin_ti=0, time_intervals=1800,
        masking_ratio=0.15, masking_mode='together', distribution='random', avg_mask_len=2,
    ):
        super().__init__(
            data_name, traf_data, traf_name, data_type, traj_batch_size, vocab,
            add_cls=add_cls, max_train_size=max_train_size, temporal_index=temporal_index,
            begin_timestamp=begin_timestamp, begin_ti=begin_ti, time_intervals=time_intervals,
        )
        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.distribution = distribution
        self.avg_mask_len = avg_mask_len
        self.exclude_feats = None

    def __getitem__(self, ind):
        """
        Args:
            ind: integer index of sample in dataset
        Returns:
            traj: list of torch traj
            temporal_mat: list of torch temporal mat
            mask: list of torch mask
            traf_X: (T, N, D)
            traf_Y: (1, N, D)
        """
        trajs, temporal_mats, traf_X, traf_Y = self.traj_traf_list[ind]
        masks = []
        for traj in trajs:
            mask = noise_mask(traj, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution, self.exclude_feats, self.add_cls)
            masks.append(torch.BoolTensor(mask))
        trajs_t = [torch.LongTensor(traj) for traj in trajs]
        temporal_mats_t = [torch.LongTensor(temporal_mat) for temporal_mat in temporal_mats]
        return trajs_t, temporal_mats_t, masks, torch.FloatTensor(traf_X), torch.FloatTensor(traf_Y)


class DRNTRLLMDataset(DRNTRLDataset):
    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_unsuperv_mask
        self.masking_ratio = config.get('masking_ratio', 0.15)
        self.masking_mode = config.get('masking_mode', 'together')
        self.distribution = config.get('distribution', 'geometric')
        self.avg_mask_len = config.get('avg_mask_len', 2)

    def _gen_dataset(self, traf_train_data, traf_eval_data, traf_test_data):
        traf_name = f"{self.scaler_type}_{self.load_external}_{self.add_time_in_day}_{self.add_day_in_week}"
        train_begin_ti = self.input_window
        train_dataset = TrajTrafDatasetLM(
            data_name=self.dataset, traf_data=traf_train_data, traf_name=traf_name, data_type='train', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=self.max_train_size, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=train_begin_ti, time_intervals=self.time_intervals,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
        )
        eval_begin_ti = train_begin_ti + len(traf_train_data)
        eval_dataset = TrajTrafDatasetLM(
            data_name=self.dataset, traf_data=traf_eval_data, traf_name=traf_name, data_type='eval', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=eval_begin_ti, time_intervals=self.time_intervals,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
        )
        test_begin_ti = eval_begin_ti + len(traf_eval_data)
        test_dataset = TrajTrafDatasetLM(
            data_name=self.dataset, traf_data=traf_test_data, traf_name=traf_name, data_type='test', traj_batch_size=self.traj_batch_size, vocab=self.vocab,
            add_cls=self.add_cls, max_train_size=None, temporal_index=self.temporal_index,
            begin_timestamp=self.begin_timestamp, begin_ti=test_begin_ti, time_intervals=self.time_intervals,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode, distribution=self.distribution, avg_mask_len=self.avg_mask_len,
        )
        return train_dataset, eval_dataset, test_dataset
