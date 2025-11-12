import torch, os, wandb
import time
import numpy as np
import torch.nn as nn
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import TRACK_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

class ExpTRACKPre(Exp_Basic):
    def __init__(self, args):
        super(ExpTRACKPre, self).__init__(args)

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

    def _build_torch_scheduler(self):
        if self.args.lr_scheduler == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.model_optim,
                step_size=self.args.lr_step,
                gamma=self.args.lr_decay)
        elif self.args.lr_scheduler == 'MultiStepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.model_optim,
                milestones=self.args.lr_milestones,
                gamma=self.args.lr_decay)
        elif self.args.lr_scheduler == "cosine":
            # 自动 warmup：线性从 0 -> base_lr，再余弦到 min_lr
            warmup_epochs = getattr(self.args, "warmup_epochs", 5)
            min_lr_ratio = getattr(self.args, "min_lr_ratio", 0.03)  # min_lr = base_lr * ratio
            # 以 epoch 为单位 step（更常见）
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.model_optim,
                start_factor=1/3,
                end_factor=1.0,
                total_iters=max(1, warmup_epochs)
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.model_optim,
                T_max=max(1, self.args.train_epochs - warmup_epochs),
                eta_min=self.args.learning_rate * min_lr_ratio
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.model_optim, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            self.lr_scheduler = scheduler
        else:
            self._logger.warning('Received unrecognized lr scheduler, no lr scheduler will be used')
            self.lr_scheduler = None
     
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        criterion = TRACK_loss
        return criterion
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch, _) in enumerate(train_loader):
                iter_count += 1
                self.model_optim.zero_grad()

                if self.args.use_amp:
                    (r1, mtp_logits1, mtp_time1), (r2, mtp_logits2, mtp_time2), mask, v1, times, node_avg, pred_next_mask, true_next_mask, mask_T = \
                        self.model.forward_pretrain(batch)  # TRACK的上游任务过多，似乎只能这样写得更清晰一些，考虑把这个EXP变成TRACK私有的
                    loss = criterion(r1, r2, mtp_logits1, v1, mask, mtp_time1, times, pred_next_mask, true_next_mask, mask_T, node_avg, self.args)
                    train_loss.append(loss.item())
                else:
                    (r1, mtp_logits1, mtp_time1), (r2, mtp_logits2, mtp_time2), mask, v1, times, node_avg, pred_next_mask, true_next_mask, mask_T = \
                        self.model.forward_pretrain(batch)  # TRACK的上游任务过多，似乎只能这样写得更清晰一些，考虑把这个EXP变成TRACK私有的
                    loss = criterion(r1, r2, mtp_logits1, v1, mask, mtp_time1, times, pred_next_mask, true_next_mask, mask_T, node_avg, self.args)
                    train_loss.append(loss.item())  

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    self.model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            wandb.log({
                "train_loss": train_loss,
                "vali_loss": vali_loss,
                "test_loss": test_loss
            })
            
            if self.lr_istorch:
                adjust_learning_rate(self.model_optim, epoch + 1, self.args, self.lr_scheduler) 
            else:
                adjust_learning_rate(self.model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model   

    def vali(self, data_set, data_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if self.args.use_amp:
                    (r1, mtp_logits1, mtp_time1), (r2, mtp_logits2, mtp_time2), mask, v1, times, node_avg, pred_next_mask, true_next_mask, mask_T = \
                        self.model.forward_pretrain(batch)
                    loss = criterion(r1, r2, mtp_logits1, v1, mask, mtp_time1, times, pred_next_mask, true_next_mask, mask_T, node_avg, self.args)
                    total_loss.append(loss.item())
                else:
                    (r1, mtp_logits1, mtp_time1), (r2, mtp_logits2, mtp_time2), mask, v1, times, node_avg, pred_next_mask, true_next_mask, mask_T = \
                        self.model.forward_pretrain(batch)
                    loss = criterion(r1, r2, mtp_logits1, v1, mask, mtp_time1, times, pred_next_mask, true_next_mask, mask_T, node_avg, self.args)
                    total_loss.append(loss.item())
        avg_loss = np.mean(total_loss)
        print(f'Validation Loss: {avg_loss:.7f}')
        return avg_loss
    
    def test(self, setting, test=0):
        print("This is the pretrain step, no test process!")