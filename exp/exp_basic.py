import os
import torch
from models import MDTP, MDTPSingle, Trajnet, TrGNN

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MDTP': MDTP,
            'MDTPmini': MDTPSingle,
            'Trajnet': Trajnet,
            'TrGNN': TrGNN
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
        if self.args.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)                                        
        elif self.args.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)  
        elif self.args.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.args.learning_rate)  
        elif self.args.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)  
        elif self.args.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.args.learning_rate)  
        elif self.args.learner.lower() == 'adamw':
            beta1 = getattr(self.args, "adamw_beta1", 0.9)
            beta2 = getattr(self.args, "adamw_beta2", 0.999)
            weight_decay = getattr(self.args, "adamw_weight_decay", 0.01)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)  
        return optimizer

    def _build_model(self):
        raise NotImplementedError
        return None

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
