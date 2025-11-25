import argparse
import os
import torch
import torch.backends
from exp.exp_prediction import ExpPrediction
from exp.exp_trackpre import ExpTRACKPre
from utils.print_args import print_args
import random
import wandb
import numpy as np

os.environ["WANDB_MODE"] = "offline"

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Traffic-Benchmark')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='TrafficPrediction',
                        help='task name, options:[TrafficPrediction, TrafficLSTM]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='MDTP',
                        help='model name, options: [MDTP, Trajnet, TrGNN]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--normalization', type=bool, default=True, help='whether to normalize the data')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='location of log files')
    parser.add_argument('--cache_dir', type=str, default='./cache/', help='location of cache files')

    # Prediction task
    parser.add_argument('--pre_steps', type=int, default=6, help='the predictions time steps')
    parser.add_argument('--seq_len', type=int, default=4, help='input sequence length')
    parser.add_argument('--NumofRoads', type=int, default=19621, help='number of nodes in the graph')
    parser.add_argument('--shp_file', type=str, default=None, help='shapefile path for edge information')
    parser.add_argument('--traj_file', type=str, default=None, help='standard order traj file path')
    parser.add_argument('--time_interval', type=int, default=10, help='time interval for flow binning (minutes)')
    parser.add_argument('--min_flow_count', type=int, default=1000, help='minimum flow count for filtering')

    # model define
    '''MDTP's args'''
    parser.add_argument('--in_feats', type=int, default=2, help='input feature dimension')
    parser.add_argument('--gcn_hidden', type=int, default=128, help='hidden dimension of GCN')
    parser.add_argument('--lstm_hidden', type=int, default=256, help='hidden dimension of LSTM')
    parser.add_argument('--fusion', type=str, default='sum', help='fusion method for traffic prediction, options:[sum, concat]')
    parser.add_argument('--N_regions', type=int, help='number of nodes in the graph')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the model')
    parser.add_argument('--S', type=int, default=24, help='sequence length for traffic prediction')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    '''Trajnet's args'''
    parser.add_argument('--T1', type=int, default=6, help='number of recent time steps')
    parser.add_argument('--T2', type=int, default=2, help='number of daily time steps')
    parser.add_argument('--T3', type=int, default=2, help='number of weekly time steps')
    parser.add_argument('--kernel_size', type=int, default=2, help='kernel size for convolution')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of layers in the model')
    parser.add_argument('--outChannel_1', type=int, default=16, help='output channel for the first convolution layer')
    parser.add_argument('--adj', type=str, default='adjacency_1000.pkl')
    parser.add_argument('--tstride', type=int, default=15, help='temporal stride for segmenting time series')
    
    '''TrGNN's args'''
    parser.add_argument('--demand_hop', type=int, default=75, help='GCN output feature size')
    parser.add_argument('--status_hop', type=int, default=3, help='GCN input feature size')
    parser.add_argument('--start_date', type=str, default='20080517')
    parser.add_argument('--end_date', type=str, default='20080610')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--min_lr_ratio', type=float, default=5e-5, help='minimum learning rate ratio')
    parser.add_argument('--preprocess_path', type=str, default=None, help='preprocess file path for TrGNN')
    
    '''TRACK args'''
    parser.add_argument('--static_feat_dim', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=32, help='Dimension of the model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--traffic_seq_len', type=int, default=7)
    parser.add_argument('--traj_max_len', type=int, default=30, help='Max length of trajectory for pretraining')
    parser.add_argument('--contrast_temp', type=float, default=0.1, help='Temperature for contrastive loss')
    parser.add_argument('--mtp_mask_ratio', type=float, default=0.15, help='Mask ratio for MTP task')
    parser.add_argument('--lambda_traj', type=float, default=1.0, help='Weight for trajectory losses')
    parser.add_argument('--lambda_traf', type=float, default=1.0, help='Weight for traffic view losses')
    parser.add_argument('--lambda_match', type=float, default=1.0, help='Weight for matching loss')
    parser.add_argument('--load_pretrained', action='store_true', help='whether to load pretrained model', default=False)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=10, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='L1', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--lr_scheduler', type=str, default=None, help='type of learning rate scheduler')
    parser.add_argument('--lr_istorch', action='store_true', help='whether to use the learning rate scheduler provided by Pytorch', default=False)
    parser.add_argument('--learner', type=str, default='adam', help='optimizer type')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--adamw_beta1', type=float, default=0.9, help='beta1 for AdamW optimizer')
    parser.add_argument('--adamw_beta2', type=float, default=0.999, help='beta2 for AdamW optimizer')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping value')
    parser.add_argument('--adamw_weight_decay', type=float, default=0.01, help='weight decay for AdamW optimizer')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    args = parser.parse_args()

    wandb.init(
    project="Traffic-Benchmark",
    config=vars(args),
    dir=args.log_dir
    )

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'TrafficPrediction':
        Exp = ExpPrediction
    elif args.task_name == 'TRACK_pretrain':
        Exp = ExpTRACKPre

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_lr_{}_ba_{}_epo_{}_itr_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.learning_rate,
                args.batch_size,
                args.train_epochs,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_lr_{}_ba_{}_epo_{}_itr_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.learning_rate,
            args.batch_size,
            args.train_epochs,
            ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
