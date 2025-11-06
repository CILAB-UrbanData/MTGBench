import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.losses as loss
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping


class Exp_drntrlCont(Exp_Basic):

    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.cont_loss_type = config.get("cont_loss_type", "simclr").lower()
        self.n_views = config.get("n_views", 2)
        self.similarity = config.get("similarity", 'cosine')  # or inner
        self.temperature = config.get("temperature", 0.05)
        self.criterion_cont = nn.CrossEntropyLoss(reduce="mean")

        self.align_w = config.get("align_w", 1.)
        self.unif_w = config.get("unif_w", 1.)
        self.align_alpha = config.get("align_alpha", 2)
        self.unif_t = config.get("unif_t", 2)
        self.train_align_uniform = config.get("train_align_uniform", False)
        self.test_align_uniform = config.get("test_align_uniform", True)
        self.norm_align_uniform = config.get("norm_align_uniform", False)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self):
        data_set, train_loader, eval_loader, test_loader, data_feature = data_provider(self.args)
        return data_set, train_loader, eval_loader, test_loader, data_feature

    def align_loss(self, x, y, alpha=2):
        if self.norm_align_uniform:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        if self.norm_align_uniform:
            x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def align_uniform(self, x, y):
        align_loss_val = self.align_loss(x, y, alpha=self.align_alpha)
        unif_loss_val = (self.uniform_loss(x, t=self.unif_t) + self.uniform_loss(y, t=self.unif_t)) / 2
        sum_loss = align_loss_val * self.align_w + unif_loss_val * self.unif_w
        return sum_loss, align_loss_val.item(), unif_loss_val.item()

    def _cal_cont_loss_simsce(self, z1, z2):
        assert z1.shape == z2.shape
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(z1, z2.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(z1, z2.T)
        similarity_matrix /= self.temperature

        labels = torch.arange(similarity_matrix.shape[0]).long().to(self.device)
        loss_res = self.criterion_cont(similarity_matrix, labels)
        return loss_res

    def _cal_match_loss(self, z1, z2):
        '''
        z1: (B, traj_B, D)
        z2: (B, traj_B, D)
        '''
        batch_size = z1.shape[0]
        z1 = z1.permute(1, 0, 2)  # (traj_B, B, D)
        z2 = z2.permute(1, 0, 2)  # (traj_B, B, D)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity_matrix = F.cosine_similarity(z1.unsqueeze(2), z2.unsqueeze(1), dim=-1)  # (traj_B, B, B)
        labels = torch.eye(batch_size, dtype=torch.bool).unsqueeze(0).expand_as(similarity_matrix)  # (traj_B, B, B)
        similarity_matrix = similarity_matrix.reshape(-1, batch_size)  # (traj_B * B, B)
        labels = labels.reshape(-1, batch_size)  # (traj_B * B, B)
        positives = similarity_matrix[labels].unsqueeze(-1)  # (traj_B * B, 1)
        negatives = similarity_matrix[~labels].reshape(-1, batch_size - 1)  # (traj_B * B, B - 1)
        logits = torch.cat([positives, negatives], dim=1)  # （traj * B, B)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device) 
        logits = logits / self.temperature

        return self.criterion_cont(logits, labels)


    def _cal_cont_loss_simclr(self, z1, z2):
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / self.temperature

        loss_res = self.criterion_cont(logits, labels)
        return loss_res

    def _cal_cont_loss_consert(self, z1, z2):
        assert z1.shape == z2.shape
        batch_size, _ = z1.shape

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        hidden1_large = z1
        hidden2_large = z2

        labels = torch.arange(0, batch_size).to(device=self.device)
        masks = F.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(
            device=self.device, dtype=torch.float)

        if self.similarity == 'inner':
            logits_aa = torch.matmul(z1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_bb = torch.matmul(z2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ab = torch.matmul(z1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ba = torch.matmul(z2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        elif self.similarity == 'cosine':
            logits_aa = F.cosine_similarity(z1.unsqueeze(1), hidden1_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_bb = F.cosine_similarity(z2.unsqueeze(1), hidden2_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_ab = F.cosine_similarity(z1.unsqueeze(1), hidden2_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_ba = F.cosine_similarity(z2.unsqueeze(1), hidden1_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
        else:
            logits_aa = torch.matmul(z1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_bb = torch.matmul(z2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ab = torch.matmul(z1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ba = torch.matmul(z2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * 1e9
        logits_bb = logits_bb - masks * 1e9
        loss_a = self.criterion_cont(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = self.criterion_cont(torch.cat([logits_ba, logits_bb], dim=1), labels)
        return loss_a + loss_b

    def _cal_cont_loss(self, z1, z2, loss_type):
        if loss_type == 'simsce':
            return self._cal_cont_loss_simsce(z1, z2)
        elif loss_type == 'simclr':
            return self._cal_cont_loss_simclr(z1, z2)
        elif loss_type == 'consert':
            return self._cal_cont_loss_consert(z1, z2)
        else:
            raise ValueError('Error contrastive loss type {}!'.format(loss_type))

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None):
        self.model.train()
        traf_epoch_loss = 0  # total traf loss of epoch
        traj_epoch_loss = 0  # total traj loss of epoch
        time_epoch_loss = 0  # total time loss of epoch
        cont_epoch_loss = 0  # total cont loss of epoch
        match_epoch_loss = 0  # total match loss of epoch
        total_correct = 0  # total top@1 acc for masked elements in epoch
        total_active_elements = 0  # total masked elements in epoch
        for i, batch in tqdm(enumerate(train_dataloader), desc=f"Train epoch={epoch_idx}", total=len(train_dataloader)):
            traj, targets, target_masks, padding_masks,\
                traj1, padding_masks1, traj2, padding_masks2, traf_X, traf_Y = batch
            traj = traj.to(self.device)  # (B, traj_B, seq_len, feat_dim)
            targets = targets.to(self.device)  # (B, traj_B, seq_len, feat_dim)
            target_masks = target_masks.to(self.device)  # (B, traj_B, seq_len, feat_dim)
            padding_masks = padding_masks.to(self.device)  # (B, traj_B, seq_len)
            traj1 = traj1.to(self.device)  # (B, traj_B, seq_len, feat_dim)
            padding_masks1 = padding_masks1.to(self.device)  # (B, traj_B, seq_len)
            traj2 = traj2.to(self.device)  # (B, traj_B, seq_len, feat_dim)
            padding_masks2 = padding_masks2.to(self.device)  # (B, traj_B, seq_len)
            traf_X = traf_X.to(self.device)  # (B, T, N, D)
            traf_Y = traf_Y.to(self.device)  # (B, T, N, D)
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

            pred_next_traf, pred_masked_traj, pred_masked_time, pool_out_traj1, pool_out_traj2, pool_drn_out_traj = self.model(
                traf_X, out_lap_pos_enc, in_lap_pos_enc,
                traj1, padding_masks1, traj2, padding_masks2,
                traj, padding_masks, batch_temporal_mat=None,
                graph_dict=self.graph_dict,
            )  # (B, N, 1), (B, traj_B, seq_len, vocab_size), (B, traj_B, seq_len)

            true_next_traf = self._scaler.inverse_transform(traf_Y[..., :self.output_dim].squeeze(1))
            pred_next_traf = self._scaler.inverse_transform(pred_next_traf[..., :self.output_dim])
            traf_loss = loss.masked_mae_torch(pred_next_traf, true_next_traf, 0)

            targets_time = targets[..., 1].float() / 60.0  # (B, traj_B, seq_len)
            time_loss = torch.zeros(1, dtype=torch.float32).to(self.device)
            if pred_masked_time is not None:
                time_loss = loss.masked_mae_torch(pred_masked_time, targets_time, 0)

            targets = targets[..., 0]  # (B, traj_B, seq_len)
            target_masks = target_masks[..., 0]  # (B, traj_B, seq_len)
            targets = targets.view(-1, targets.shape[-1])  # (B * traj_B, seq_len)
            target_masks = target_masks.view(-1, target_masks.shape[-1])  # (B * traj_B, seq_len)
            # (B * traj_B, seq_len, vocab_size)
            pred_masked_traj = pred_masked_traj.view(-1, pred_masked_traj.shape[-2], pred_masked_traj.shape[-1])
            traj_loss_list = self.criterion(pred_masked_traj.transpose(1, 2), targets)
            traj_batch_loss = torch.sum(traj_loss_list)
            num_active = target_masks.sum()
            traj_loss = traj_batch_loss / num_active

            match_loss = torch.zeros(1, dtype=torch.float32).to(self.device)
            if pool_drn_out_traj is not None and pool_out_traj2.shape[0] > 1:
                match_loss = self._cal_match_loss(pool_out_traj2, pool_drn_out_traj)

            pool_out_traj1 = pool_out_traj1.view(-1, pool_out_traj1.shape[-1])
            pool_out_traj2 = pool_out_traj2.view(-1, pool_out_traj2.shape[-1])
            cont_loss = self._cal_cont_loss(pool_out_traj1, pool_out_traj2, self.cont_loss_type)

            total_loss = traf_loss + traj_loss + time_loss + cont_loss + match_loss

            if self.test_align_uniform:
                align_uniform_loss, align_loss, uniform_loss = self.align_uniform(pool_out_traj1, pool_out_traj2)

            if self.l2_reg is not None:
                total_loss += self.l2_reg * loss.l2_reg_loss(self.model)

            total_loss = total_loss / self.grad_accmu_steps
            batches_seen += 1

            total_loss.backward()
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()

            with torch.no_grad():
                mask_label = targets[target_masks]
                lm_output = pred_masked_traj[target_masks].argmax(dim=-1)
                assert mask_label.shape == lm_output.shape
                correct = mask_label.eq(lm_output).sum().item()
                total_correct += correct
                total_active_elements += num_active.item()
                traj_epoch_loss += traj_batch_loss.item()
                traf_epoch_loss += traf_loss.item()
                time_epoch_loss += time_loss.item()
                cont_epoch_loss += cont_loss.item()
                match_epoch_loss += match_loss.item()

            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "traf_loss": traf_loss.item(),
                "traj_loss": traj_loss.item(),
                "time_loss": time_loss.item(),
                "cont_loss": cont_loss.item(),
                "match_loss": match_loss.item(),
                "acc(%)": total_correct / total_active_elements * 100,
            }
            if self.test_align_uniform:
                post_fix['align_loss'] = align_loss
                post_fix['uniform_loss'] = uniform_loss
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))

        traf_epoch_loss = traf_epoch_loss / len(train_dataloader)
        traj_epoch_loss = traj_epoch_loss / total_active_elements
        time_epoch_loss = time_epoch_loss / len(train_dataloader)
        cont_epoch_loss = cont_epoch_loss / len(train_dataloader)
        match_epoch_loss = match_epoch_loss / len(train_dataloader)
        total_correct = total_correct / total_active_elements * 100.0
        self._logger.info(
            f"Train: expid = {self.exp_id}, Epoch = {epoch_idx}, traf_avg_loss = {traf_epoch_loss},\
            traj_avg_loss = {traj_epoch_loss}, time_avg_loss = {time_epoch_loss}, cont_avg_loss = {cont_epoch_loss},\
            match_avg_loss = {match_epoch_loss}, total_acc = {total_correct}%."
        )
        epoch_loss = traf_epoch_loss + traj_epoch_loss + time_epoch_loss + cont_epoch_loss + match_epoch_loss
        self._writer.add_scalar('Train loss', epoch_loss, epoch_idx)
        self._writer.add_scalar('Train acc', total_correct, epoch_idx)
        return epoch_loss, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        self.model.eval()
        traf_epoch_loss = 0  # total traf loss of epoch
        traj_epoch_loss = 0  # total traj loss of epoch
        time_epoch_loss = 0  # total time loss of epoch
        cont_epoch_loss = 0  # total cont loss of epoch
        match_epoch_loss = 0  # total match loss of epoch
        total_correct = 0  # total top@1 acc for masked elements in epoch
        total_active_elements = 0  # total masked elements in epoch
        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc=f"{mode} epoch={epoch_idx}", total=len(eval_dataloader)):
                traj, targets, target_masks, padding_masks,\
                    traj1, padding_masks1, traj2, padding_masks2, traf_X, traf_Y = batch
                traj = traj.to(self.device)  # (B, traj_B, seq_len, feat_dim)
                targets = targets.to(self.device)  # (B, traj_B, seq_len, feat_dim)
                target_masks = target_masks.to(self.device)  # (B, traj_B, seq_len, feat_dim)
                padding_masks = padding_masks.to(self.device)  # (B, traj_B, seq_len)
                traj1 = traj1.to(self.device)  # (B, traj_B, seq_len, feat_dim)
                padding_masks1 = padding_masks1.to(self.device)  # (B, traj_B, seq_len)
                traj2 = traj2.to(self.device)  # (B, traj_B, seq_len, feat_dim)
                padding_masks2 = padding_masks2.to(self.device)  # (B, traj_B, seq_len)
                traf_X = traf_X.to(self.device)  # (B, T, N, D)
                traf_Y = traf_Y.to(self.device)  # (B, T, N, D)

                pred_next_traf, pred_masked_traj, pred_masked_time, pool_out_traj1, pool_out_traj2, pool_drn_out_traj = self.model(
                    traf_X, self.out_lap_mx, self.in_lap_mx,
                    traj1, padding_masks1, traj2, padding_masks2,
                    traj, padding_masks, batch_temporal_mat=None,
                    graph_dict=self.graph_dict,
                )

                true_next_traf = self._scaler.inverse_transform(traf_Y[..., :self.output_dim].squeeze(1))
                pred_next_traf = self._scaler.inverse_transform(pred_next_traf[..., :self.output_dim])
                traf_loss = loss.masked_mae_torch(pred_next_traf, true_next_traf, 0)

                targets_time = targets[..., 1].float() / 60.0  # (B, traj_B, seq_len)
                time_loss = torch.zeros(1, dtype=torch.float32).to(self.device)
                if pred_masked_time is not None:
                    time_loss = loss.masked_mae_torch(pred_masked_time, targets_time, 0)

                targets = targets[..., 0]  # (B, traj_B, seq_len)
                target_masks = target_masks[..., 0]  # (B, traj_B, seq_len)
                targets = targets.view(-1, targets.shape[-1])
                target_masks = target_masks.view(-1, target_masks.shape[-1])
                pred_masked_traj = pred_masked_traj.view(-1, pred_masked_traj.shape[-2], pred_masked_traj.shape[-1])
                traj_loss_list = self.criterion(pred_masked_traj.transpose(1, 2), targets)
                traj_batch_loss = torch.sum(traj_loss_list)
                num_active = target_masks.sum()
                traj_loss = traj_batch_loss / num_active

                match_loss = torch.zeros(1, dtype=torch.float32).to(self.device)
                if pool_drn_out_traj is not None and pool_out_traj2.shape[0] > 1:
                    match_loss = self._cal_match_loss(pool_out_traj2, pool_drn_out_traj)

                pool_out_traj1 = pool_out_traj1.view(-1, pool_out_traj1.shape[-1])
                pool_out_traj2 = pool_out_traj2.view(-1, pool_out_traj2.shape[-1])
                cont_loss = self._cal_cont_loss(pool_out_traj1, pool_out_traj2, self.cont_loss_type)

                if self.test_align_uniform:
                    align_uniform_loss, align_loss, uniform_loss = self.align_uniform(pool_out_traj1, pool_out_traj2)

                mask_label = targets[target_masks]
                lm_output = pred_masked_traj[target_masks].argmax(dim=-1)
                assert mask_label.shape == lm_output.shape
                correct = mask_label.eq(lm_output).sum().item()
                total_correct += correct
                total_active_elements += num_active.item()
                traj_epoch_loss += traj_batch_loss.item()
                traf_epoch_loss += traf_loss.item()
                time_epoch_loss += time_loss.item()
                cont_epoch_loss += cont_loss.item()
                match_epoch_loss += match_loss.item()

                post_fix = {
                    "mode": mode,
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "traf_loss": traf_loss.item(),
                    "traj_loss": traj_loss.item(),
                    "time_loss": time_loss.item(),
                    "cont_loss": cont_loss.item(),
                    "match_loss": match_loss.item(),
                    "acc(%)": total_correct / total_active_elements * 100,
                }
                if self.test_align_uniform:
                    post_fix['align_loss'] = align_loss
                    post_fix['uniform_loss'] = uniform_loss
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            traf_epoch_loss = traf_epoch_loss / len(eval_dataloader)
            traj_epoch_loss = traj_epoch_loss / total_active_elements
            time_epoch_loss = time_epoch_loss / len(eval_dataloader)
            cont_epoch_loss = cont_epoch_loss / len(eval_dataloader)
            match_epoch_loss = match_epoch_loss / len(eval_dataloader)
            total_correct = total_correct / total_active_elements * 100.0
            self._logger.info(
                f"{mode}: expid = {self.exp_id}, Epoch = {epoch_idx}, traf_avg_loss = {traf_epoch_loss},\
                traj_avg_loss = {traj_epoch_loss}, time_avg_loss = {time_epoch_loss}, cont_avg_loss = {cont_epoch_loss},\
                match_avg_loss = {match_epoch_loss}, total_acc = {total_correct}%."
            )
            epoch_loss = traf_epoch_loss + traj_epoch_loss + time_epoch_loss + cont_epoch_loss + match_epoch_loss
            self._writer.add_scalar(f'{mode} loss', epoch_loss, epoch_idx)
            self._writer.add_scalar(f'{mode} acc', total_correct, epoch_idx)
            return epoch_loss
    
    def train(self, setting):
        data_set, train_loader, eval_loader, test_loader, data_feature = self._get_data()
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        