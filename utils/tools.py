import os
from datetime import datetime as dt
from datetime import date, timedelta
import numpy as np
import torch, scipy
import matplotlib.pyplot as plt
import pandas as pd
import math, wandb, time
from logging import getLogger

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_current_lr(optimizer):
    # 多 param_group 时可记录列表；这里取第一个最常用
    return optimizer.param_groups[0]["lr"]

#将adjust_learning_rate添加一个大分支，torch托管的lr走一边,手动调整的lr走老一边(判断方式为exp的self标志位)
def adjust_learning_rate(optimizer, epoch, args, lr_scheduler=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if not args.lr_istorch:
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == 'type3':
            lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif args.lradj == 'type4':
            lr_adjust = {epoch: args.learning_rate * (0.1 ** ((epoch - 1) // 50))}
        elif args.lradj == "cosine":
            lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
        elif args.lradj == "type5":
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 30))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
            wandb.log({"learning_rate": lr})
    else:
        if lr_scheduler is not None:
            lr_scheduler.step()
            print('Updating learning rate to {}'.format(get_current_lr(optimizer)))
            wandb.log({"learning_rate": get_current_lr(optimizer)}, step=epoch)


class Scheduler:

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True):
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch):
        return None

    def get_update_values(self, num_updates):
        return None

    def step(self, epoch, metric=None):
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates, metric=None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs


class CosineLRScheduler(Scheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        self._logger = getLogger()
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            self._logger.warning("Cosine annealing scheduler will have no effect on the learning "
                                 "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def date_range(date1, date2):
    # date1, date2 = '20160401', '20160428'
    datetime1 = dt.strptime(date1, '%Y%m%d')
    datetime2 = dt.strptime(date2, '%Y%m%d')
    days = (datetime2 - datetime1).days + 1
    date_list = [(datetime1 + timedelta(day)).strftime('%Y_%m%d') for day in range(days)]  #TODO:Gaiya的处理要在%y_%m%d格式下进行,适配
    return date_list

def time_difference(time1, time2):
    # format: '25/03/2016 00:00:04'
    # time_difference = time1 - time2
    return (dt.strptime(time1, '%d/%m/%Y %H:%M:%S') - dt.strptime(time2, '%d/%m/%Y %H:%M:%S')).total_seconds()


def df_to_csv(df, file_path, index=False):
    print('Saving to file at %s'%(file_path))
    if os.path.exists(file_path):
        temp_file_path = '%s_temp'%(file_path)
        df.to_csv(temp_file_path, index=index)
        os.system('rm %s'%(file_path))
        os.system('mv %s %s'%(temp_file_path, file_path))
    else:
        df.to_csv(file_path, index=index)
    print('Saved.')

def round_time(t, interval=5):
    # t = '25/03/2016 12:26:45'
    # output: '25/03/2016 12:25:00'
    # interval: in minutes
    interval = interval * 60 # convert minutes to seconds
    datetime = dt.strptime(t, '%d/%m/%Y %H:%M:%S')
    new_datetime = dt.fromtimestamp(int(time.mktime(datetime.timetuple())) // interval * interval)
    return new_datetime.strftime('%d/%m/%Y %H:%M:%S')

def to_sparse_tensor(dense_matrix):
    coo = scipy.sparse.coo_matrix(dense_matrix)

    indices = torch.LongTensor(np.vstack((coo.row, coo.col)).astype(np.int64))  # 强制转为int64
    values = torch.FloatTensor(coo.data)
    shape = coo.shape

    # 推荐用新版API，避免警告
    sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))

    return sparse_tensor

def top_k(loc_pred, loc_true, topk):
    loc_pred = torch.FloatTensor(loc_pred)  # (batch_size * output_dim)
    val, index = torch.topk(loc_pred, topk, 1)
    index = index.numpy()  # (batch_size * topk)
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):  # i->batch, p->(topk,)
        target = loc_true[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg