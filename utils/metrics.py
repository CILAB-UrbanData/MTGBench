import numpy as np
import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sims = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(sims, labels) + F.cross_entropy(sims.t(), labels)) / 2.0

def mtp_loss(mtp_logits, true_nodes, mask):
    B,L,N = mtp_logits.shape
    logits = mtp_logits.view(B*L, N)
    targets = true_nodes.view(B*L)
    maskf = mask.view(B*L)
    if maskf.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(logits[maskf], targets[maskf])

def mtp_time_loss(pred_time, true_time, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_time.device, requires_grad=True)
    diff = pred_time[mask] - true_time[mask]
    return F.mse_loss(diff, torch.zeros_like(diff))

def match_loss(traj_repr, node_repr_avg, temperature=0.1):
    return nt_xent_loss(traj_repr, node_repr_avg, temperature)

def next_state_loss(pred, true):
    return torch.nn.functional.l1_loss(pred, true)

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
