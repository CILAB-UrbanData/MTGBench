from datetime import datetime
import pandas as pd
import os
import time
import tqdm
import torch
import wandb
import numpy as np
import torch.nn as nn
from functools import partial
import utils.losses as loss
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import ensure_dir, EarlyStopping, adjust_learning_rate, CosineLRScheduler, top_k

class Exp_trllm(Exp_Basic):
    def __init__(self, args):
        super(Exp_trllm, self).__init__(args)
        self.learner = args.get('learner', 'adamw')
        self.learning_rate = args.get('learning_rate', 1e-4)
        self.weight_decay = args.get('weight_decay', 0.01)
        self.lr_beta1 = args.get('lr_beta1', 0.9)
        self.lr_beta2 = args.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = args.get('lr_alpha', 0.99)
        self.lr_epsilon = args.get('lr_epsilon', 1e-8)
        self.lr_momentum = args.get('lr_momentum', 0)
        self.grad_accmu_steps = args.get('grad_accmu_steps', 1)
        self.test_every = args.get('test_every', 10)

        self.lr_decay = args.get('lr_decay', True)
        self.lr_scheduler_type = args.get('lr_scheduler', 'cosinelr')
        self.lr_decay_ratio = args.get('lr_decay_ratio', 0.1)
        self.milestones = args.get('steps', [])
        self.step_size = args.get('step_size', 10)
        self.lr_lambda = args.get('lr_lambda', lambda x: x)
        self.lr_T_max = args.get('lr_T_max', 30)
        self.lr_eta_min = args.get('lr_eta_min', 0)
        self.lr_patience = args.get('lr_patience', 10)
        self.lr_threshold = args.get('lr_threshold', 1e-4)
        self.lr_warmup_epoch = args.get("lr_warmup_epoch", 5)
        self.lr_warmup_init = args.get("lr_warmup_init", 1e-6)
        self.t_in_epochs = args.get("t_in_epochs", True)

        self.clip_grad_norm = args.get('clip_grad_norm', False)
        self.max_grad_norm = args.get('max_grad_norm', 1.)
        self.hyper_tune = args.get('hyper_tune', False)
        self.l2_reg = args.get('l2_reg', None)
        
        self.batch_size = args.get("batch_size", 64)
        self.n_views = args.get("n_views", 2)
        self.similarity = args.get("similarity", 'cosine')  # or inner
        self.temperature = args.get("temperature", 0.05)
        self.contra_loss_type = args.get("contra_loss_type", 'simclr').lower()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.data_augment1 = args.get("data_augment1", [])
        self.data_augment2 = args.get("data_augment2", [])

        self.align_w = args.get("align_w", 1.)
        self.unif_w = args.get("unif_w", 1.)
        self.align_alpha = args.get("align_alpha", 2)
        self.unif_t = args.get("unif_t", 2)
        self.train_align_uniform = args.get("train_align_uniform", False)
        self.test_align_uniform = args.get("test_align_uniform", True)
        self.norm_align_uniform = args.get("norm_align_uniform", False)
                
        self.criterion_mask = nn.NLLLoss(ignore_index=0, reduction='none')
        self.time_loss_ratio = args.get("time_loss_ratio", 1)
        self.mlm_loss_ratio = args.get("mlm_loss_ratio", 1)
        self.cont_loss_ratio = args.get("cont_loss_ratio", 1)  
        self.topk = args.get('topk', [1])  
        self.args = args

        self.evaluator_clear()

    def _cal_loss(self, pred, targets, targets_mask):
        batch_loss_list = self.criterion_mask(pred.transpose(1, 2), targets)
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        return mean_loss, batch_loss, num_active

    def _cal_acc(self, pred, targets, targets_mask):
        mask_label = targets[targets_mask]  # (num_active, )
        lm_output = pred[targets_mask].argmax(dim=-1)  # (num_active, )
        correct_l = mask_label.eq(lm_output).sum().item()
        return correct_l
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self):
        data_set, train_loader, eval_loader, test_loader, data_feature = data_provider(self.args)
        return data_set, train_loader, eval_loader, test_loader, data_feature

    def _select_optimizer(self):
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _contrastive_loss(self, z1, z2, loss_type):
        if loss_type == 'simsce':
            return self._contrastive_loss_simsce(z1, z2)
        elif loss_type == 'simclr':
            return self._contrastive_loss_simclr(z1, z2)
        elif loss_type == 'consert':
            return self._contrastive_loss_consert(z1, z2)
        else:
            raise ValueError('Error contrastive loss type {}!'.format(loss_type))

    def _build_lr_scheduler(self):
        self.lr_scheduler_type = self.args.get('lr_scheduler', 'cosinelr')
        if self.lr_scheduler_type.lower() == 'multisteplr':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
        elif self.lr_scheduler_type.lower() == 'steplr':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
        elif self.lr_scheduler_type.lower() == 'exponentiallr':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.lr_decay_ratio)
        elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
        elif self.lr_scheduler_type.lower() == 'lambdalr':
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=self.lr_lambda)
        elif self.lr_scheduler_type.lower() == 'reducelronplateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=self.lr_patience,
                factor=self.lr_decay_ratio, threshold=self.lr_threshold)
        elif self.lr_scheduler_type.lower() == 'cosinelr':
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init, t_in_epochs=self.t_in_epochs)

    def vali(self, eval_loader):
        self.model.eval()
        total_loss_list = []
        with torch.no_grad():
            for batch in eval_loader:
                contra_view1, contra_view2, padding_masks1, padding_masks2, \
                    X, targets, target_masks, padding_masks = batch
                contra_view1 = contra_view1.to(self.device)
                contra_view2 = contra_view2.to(self.device)
                padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
                padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore

                predictions_l, predictions_t, z1, z2 = self.model(
                    contra_view1, padding_masks1, contra_view2, padding_masks2,
                    X, padding_masks, batch_temporal_mat=None,
                )
                # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
                targets_time = targets[..., 1].float() / 60.0  # (B, seq_len)
                mean_loss_t = 0
                if predictions_t is not None:
                    mean_loss_t = loss.masked_mae_torch(predictions_t, targets_time, 0)
                targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
                mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
                mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

                mean_loss = self.time_loss_ratio * mean_loss_t + \
                            self.mlm_loss_ratio * mean_loss_l + \
                            self.cont_loss_ratio * mean_loss_con

                if self.test_align_uniform or self.train_align_uniform:
                    align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                    if self.train_align_uniform:
                        mean_loss += align_uniform_loss  

                total_loss = mean_loss
                if self.l2_reg is not None:
                    total_loss += self.l2_reg * loss.l2_reg_loss(self.model)

                total_loss_list.append(total_loss.item())

        total_loss = np.mean(total_loss_list)
        return total_loss

    def test(self, setting, test):
        data_set, train_loader, eval_loader, test_loader, data_feature = self._get_data()
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                contra_view1, contra_view2, padding_masks1, padding_masks2, \
                    X, targets, target_masks, padding_masks = batch
                contra_view1 = contra_view1.to(self.device)
                contra_view2 = contra_view2.to(self.device)
                padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
                padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore

                predictions_l, predictions_t, z1, z2 = self.model(
                    contra_view1, padding_masks1, contra_view2, padding_masks2,
                    X, padding_masks, batch_temporal_mat=None,
                )
                # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
                targets_time = targets[..., 1].float() / 60.0  # (B, seq_len)
                mean_loss_t = 0
                if predictions_t is not None:
                    mean_loss_t = loss.masked_mae_torch(predictions_t, targets_time, 0)
                targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
                mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
                mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

                mean_loss = self.time_loss_ratio * mean_loss_t + \
                            self.mlm_loss_ratio * mean_loss_l + \
                            self.cont_loss_ratio * mean_loss_con

                if self.test_align_uniform or self.train_align_uniform:
                    align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                    if self.train_align_uniform:
                        mean_loss += align_uniform_loss 

                evaluate_input = {
                        'loc_true': targets_l[target_masks_l].reshape(-1, 1).squeeze(-1).cpu().numpy(),  # (num_active, )
                        'loc_pred': predictions_l[target_masks_l].reshape(-1, predictions_l.shape[-1]).cpu().numpy()  # (num_active, n_class)
                    }
                self.collect(evaluate_input) 
                
            folder_path = './test_results/' + setting + '/'
            self.save_result(save_path=folder_path, filename='test_result')

    def evaluator_clear(self):
        self.result = {}
        self.intermediate_result = dict()
        self.intermediate_result['total'] = 0
        for inter in ['hit']:
            for k in self.topk:
                self.intermediate_result[inter + '@' + str(k)] = 0
        for inter in ['rank', 'dcg']:
            for k in self.topk:
                self.intermediate_result[inter + '@' + str(k)] = 0.0

    def collect(self, evaluate_input):
        total = len(evaluate_input['loc_true'])
        self.intermediate_result['total'] += total
        for k in self.topk:
            hit, rank, dcg = top_k(evaluate_input['loc_pred'], evaluate_input['loc_true'], k)
            self.intermediate_result['hit@' + str(k)] += hit
            self.intermediate_result['rank@' + str(k)] += rank
            self.intermediate_result['dcg@' + str(k)] += dcg

    def evaluate(self):
        for k in self.topk:
            precision = self.intermediate_result['hit@{}'.format(k)] / (self.intermediate_result['total'] * k)
            self.result['Precision@{}'.format(k)] = precision
            recall = self.intermediate_result['hit@{}'.format(k)] / self.intermediate_result['total']
            self.result['Recall@{}'.format(k)] = recall
            self.result['F1@{}'.format(k)] = \
                0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

            self.result['MRR@{}'.format(k)] = \
                self.intermediate_result['rank@{}'.format(k)] / self.intermediate_result['total']
            self.result['MAP@{}'.format(k)] = \
                self.intermediate_result['rank@{}'.format(k)] / self.intermediate_result['total']
            self.result['NDCG@{}'.format(k)] = \
                self.intermediate_result['dcg@{}'.format(k)] / self.intermediate_result['total']
        return self.result
    
    def save_result(self, save_path, filename=None):
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            filename = str(self.config['exp_id']) + '_' + \
                       datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']
        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for metric in self.metrics:
                for k in self.topk:
                    dataframe[metric].append(self.result[metric + '@' + str(k)])
            dataframe = pd.DataFrame(dataframe, index=self.topk)
            path = os.path.join(save_path, '{}.csv'.format(filename))
            dataframe.to_csv(path, index=False)
            self._logger.info('Evaluate result is saved at ' + path)
            self._logger.info("\n" + str(dataframe))
        return dataframe

    def train(self, setting):
        data_set, train_loader, eval_loader, test_loader, data_feature = self._get_data()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            self.model_optim.zero_grad()

            batches_seen = epoch * train_steps      

            epoch_loss = []  # total loss of epoch
            epoch_loss_t = []
            total_correct_l = 0  # total top@1 acc for masked elements in epoch
            total_active_elements_l = 0  # total masked elements in epoch

            for i, batch in tqdm(enumerate(train_loader), desc="Train epoch={}".format(epoch), total=len(train_loader)):
                contra_view1, contra_view2, padding_masks1, padding_masks2, \
                    X, targets, target_masks, padding_masks = batch
                contra_view1 = contra_view1.to(self.device)
                contra_view2 = contra_view2.to(self.device)
                padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
                padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore  

                predictions_l, predictions_t, z1, z2 = self.model(
                    contra_view1, padding_masks1, contra_view2, padding_masks2,
                    X, padding_masks, batch_temporal_mat=None,
                )
                # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
                targets_time = targets[..., 1].float() / 60.0  # (B, seq_len)
                mean_loss_t = 0
                if predictions_t is not None:
                    mean_loss_t = loss.masked_mae_torch(predictions_t, targets_time, 0)
                targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
                mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
                mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

                mean_loss = self.time_loss_ratio * mean_loss_t + \
                            self.mlm_loss_ratio * mean_loss_l + \
                            self.cont_loss_ratio * mean_loss_con

                if self.test_align_uniform or self.train_align_uniform:
                    align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                    if self.train_align_uniform:
                        mean_loss += align_uniform_loss  

                total_loss = mean_loss
                if self.l2_reg is not None:
                    total_loss += self.l2_reg * loss.l2_reg_loss(self.model)

                total_loss = total_loss / self.grad_accmu_steps
                batches_seen += 1

                total_loss.backward()   

                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if batches_seen % self.grad_accmu_steps == 0:
                    self.model_optim.step()
                    if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:  
                        self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                    self.model_optim.zero_grad()

                with torch.no_grad():
                    total_correct_l += self._cal_acc(predictions_l, targets_l, target_masks_l)
                    total_active_elements_l += num_active_l.item()
                    epoch_loss.append(mean_loss.item())  # add total loss of batch
                    epoch_loss_t.append(mean_loss_l.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))                   
            train_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
            train_loss_t = np.mean(epoch_loss_t)  # average loss per element for whole epoch
            train_correct_l = total_correct_l / total_active_elements_l * 100.0

            vali_loss, vali_correct_l = self.vali(eval_loader)
            test_loss, test_correct_l = self.vali(test_loader)

            print("Epoch: {} | train_loss: {:.7f}, train_acc: {:.2f}% | vali_loss: {:.7f}, vali_acc: {:.2f}% | test_loss: {:.7f}, test_acc: {:.2f}%".format(
                epoch + 1, train_loss, train_correct_l, vali_loss, vali_correct_l, test_loss, test_correct_l))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_correct_l,
                "vali_loss": vali_loss,
                "vali_acc": vali_correct_l,
                "test_loss": test_loss,
                "test_acc": test_correct_l
            })

            adjust_learning_rate(self.model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
