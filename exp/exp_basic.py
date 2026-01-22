import os
import torch
from models import MDTP, MDTPSingle, Trajnet, TrGNN, TRACK

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MDTP': MDTP,
            'MDTPsingle': MDTPSingle,
            'Trajnet': Trajnet,
            'TrGNN': TrGNN,
            'TRACK': TRACK,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.model_optim = self._select_optimizer()
        self.lr_istorch = args.lr_istorch
        self.lr_scheduler = None
        if self.lr_istorch:
            self._build_torch_scheduler()

    def _select_optimizer(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_torch_scheduler(self):
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
