import os
from datetime import datetime as dt
from datetime import date, timedelta
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math, wandb, time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
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
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        wandb.log({"learning_rate": lr})


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
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


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


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
    date_list = [(datetime1 + timedelta(day)).strftime('%Y%m%d') for day in range(days)]
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