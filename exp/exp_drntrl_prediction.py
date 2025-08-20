import os
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
from functools import partial
import utils.losses as loss
from libcity.utils import get_evaluator
from libcity.executor.drntrl_executor import DRNTRLExecutor
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import ensure_dir, EarlyStopping, adjust_learning_rate

class ExpDRNTRL(Exp_Basic):
    def __init__(self, args):
        super(ExpDRNTRL, self).__init__(args)
        self.args = args
        self.exp_id = getattr(args, 'exp_id', None)
        self.output_dim = getattr(args, 'output_dim', 1)
        self._scaler = data_feature.get("scaler")

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)    

        self.learner = getattr(args, 'learner', 'adamw')
        self.learning_rate = getattr(args, 'learning_rate', 0.001)
        self.weight_decay = getattr(args, 'weight_decay', 0.05)
        self.lr_beta1 = getattr(args, 'lr_beta1', 0.9)
        self.lr_beta2 = getattr(args, 'lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = getattr(args, 'lr_alpha', 0.99)
        self.lr_epsilon = getattr(args, 'lr_epsilon', 1e-8)
        self.lr_momentum = getattr(args, 'lr_momentum', 0)
        self.lr_decay = getattr(args, 'lr_decay', True)
        self.lr_scheduler_type = getattr(args, 'lr_scheduler', 'cosinelr')
        self.lr_decay_ratio = getattr(args, 'lr_decay_ratio', 0.1)
        self.milestones = getattr(args, 'steps', [5, 20, 40, 70])
        self.step_size = getattr(args, 'step_size', 10)
        self.lr_lambda = getattr(args, 'lr_lambda', lambda x: x)
        self.lr_T_max = getattr(args, 'lr_T_max', 30)
        self.lr_eta_min = getattr(args, 'lr_eta_min', 0)
        self.lr_patience = getattr(args, 'lr_patience', 10)
        self.lr_threshold = getattr(args, 'lr_threshold', 1e-4)
        self.lr_warmup_epoch = getattr(args, "lr_warmup_epoch", 5)
        self.lr_warmup_init = getattr(args, "lr_warmup_init", 1e-6)

        self.clip_grad_norm = getattr(args, 'clip_grad_norm', True)
        self.max_grad_norm = getattr(args, 'max_grad_norm', 5.0)
        self.grad_accmu_steps = getattr(args, 'grad_accmu_steps', 1)

        self.lape_dim = getattr(args, 'lape_dim', 8)
        self.random_flip = getattr(args, 'random_flip', True)

        self.graph_dict = {
            # 'out_lap_mx': self.out_lap_mx,
            # 'in_lap_mx': self.in_lap_mx,
            'outdegree': outdegree,
            'indegree': indegree,
            'node_features': node_features,
            'traj_edge_index': traj_edge_index,
            'traj_loc_trans_prob': traj_loc_trans_prob,
            'traj_t_loc_trans_prob': traj_t_loc_trans_prob,
        }

        self.task_level = getattr(args, "task_level", 0)
        self.step_size = getattr(args, 'step_size', 2500)
        self.output_window = getattr(args, "output_window", 6)
        self.use_curriculum_learning = getattr(args, 'use_curriculum_learning', True)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.args.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.args.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.args.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.args.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        elif self.args.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _select_criterion(self):
        criterion = loss.masked_mae_torch
        return criterion
    
    def test(self, setting, test=0):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

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

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (inputs, target) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                target = target.to(self.args.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, target)
                        train_loss.append(loss.item())

                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, target)
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
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

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
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            return self.model
        
    def vali(self, data_set, data_loader, criterion):
        self.model.eval()
        total_loss = []
        if self.args.model == "Trajnet":
            data_set.on_epoch_start()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(data_loader):
                target = target.to(self.args.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, target)
                        total_loss.append(loss.item())
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, target)
                    total_loss.append(loss.item())
        
        avg_loss = np.mean(total_loss)
        print(f'Validation Loss: {avg_loss:.7f}')
        return avg_loss

class DRNTRLTSPExecutor(DRNTRLExecutor):

    def __init__(self, args, model, data_feature):
        super().__init__(args, model, data_feature)
        self.evaluator = get_evaluator(args)
        self.task_level = getattr(args, "task_level", 0)
        self.step_size = getattr(args, 'step_size', 2500)
        self.output_window = getattr(args, "output_window", 6)
        self.use_curriculum_learning = getattr(args, 'use_curriculum_learning', True)
        self.args = args

    def evaluate(self, test_dataloader):
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                traf_X, traf_Y = batch
                traf_X = traf_X.to(self.device)  # (B, T, N, D)
                traf_Y = traf_Y.to(self.device)  # (B, T, N, D)
                # pred_traf = self.model(traf_X, self.out_lap_mx, self.in_lap_mx, graph_dict=self.graph_dict)
                pred_traf = self.model(traf_X, graph_dict=self.graph_dict)
                y_true = self._scaler.inverse_transform(traf_Y[..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(pred_traf[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses, batches_seen = self._train_epoch(train_dataloader, epoch_idx, batches_seen)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss = np.mean(losses)
            self._writer.add_scalar('training loss', train_loss, batches_seen)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, batches_seen)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            if epoch_idx % self.log_every == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, batches_seen, train_loss, val_loss, log_lr, end_time - start_time)
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self._save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            average_train_time = sum(train_time) / len(train_time)
            average_eval_time = sum(eval_time) / len(eval_time)
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), average_train_time, average_eval_time))
        if self.load_best_epoch:
            self._load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None):
        self.model.train()
        losses = []
        for batch in train_dataloader:
            traf_X, traf_Y = batch
            traf_X = traf_X.to(self.device)  # (B, T, N, D)
            traf_Y = traf_Y.to(self.device)  # (B, T, N, D)
            '''
            out_lap_pos_enc = self.out_lap_mx.to(self.device)
            in_lap_pos_enc = self.in_lap_mx.to(self.device)
            if self.random_flip:
                out_sign_flip = torch.rand(out_lap_pos_enc.size(1)).to(self.device)
                out_sign_flip[out_sign_flip >= 0.5] = 1.0
                out_sign_flip[out_sign_flip < 0.5] = -1.0
                out_lap_pos_enc = out_lap_pos_enc * out_sign_flip.unsqueeze(0)
                in_sign_flip = torch.rand(in_lap_pos_enc.size(1)).to(self.device)
                in_sign_flip[in_sign_flip >= 0.5] = 1.0
                in_sign_flip[in_sign_flip < 0.5] = -1.0
                in_lap_pos_enc = in_lap_pos_enc * in_sign_flip.unsqueeze(0)
            pred_traf = self.model(traf_X, out_lap_pos_enc, in_lap_pos_enc, graph_dict=self.graph_dict)
            '''
            pred_traf = self.model(traf_X, graph_dict=self.graph_dict)
            y_true = self._scaler.inverse_transform(traf_Y[..., :self.output_dim])
            y_pred = self._scaler.inverse_transform(pred_traf[..., :self.output_dim])
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info(f'Training: task_level increase from {self.task_level - 1} to {self.task_level}')
                self._logger.info('Current batches_seen is {}'.format(batches_seen))
            if self.use_curriculum_learning:
                traf_loss = loss.masked_mae_torch(y_pred[:, :self.task_level, :, :], y_true[:, :self.task_level, :, :], 0)
            else:
                traf_loss = loss.masked_mae_torch(y_pred, y_true, 0)

            self._logger.debug(traf_loss.item())
            losses.append(traf_loss.item())
            
            traf_loss = traf_loss / self.grad_accmu_steps
            batches_seen += 1

            traf_loss.backward()
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen=None):
        with torch.no_grad():
            self.model.eval()
            losses = []
            for batch in eval_dataloader:
                traf_X, traf_Y = batch
                traf_X = traf_X.to(self.device)  # (B, T, N, D)
                traf_Y = traf_Y.to(self.device)  # (B, T, N, D)
                # pred_traf = self.model(traf_X, self.out_lap_mx, self.in_lap_mx, graph_dict=self.graph_dict)
                pred_traf = self.model(traf_X, graph_dict=self.graph_dict)
                y_true = self._scaler.inverse_transform(traf_Y[..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(pred_traf[..., :self.output_dim])
                traf_loss = loss.masked_mae_torch(y_pred, y_true, 0)
                self._logger.debug(traf_loss.item())
                losses.append(traf_loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss
