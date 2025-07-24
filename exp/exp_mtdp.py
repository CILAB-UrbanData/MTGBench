from exp.exp_basic import Exp_Basic
from layers.lstm_init import init_state
from utils.tools import EarlyStopping, adjust_learning_rate
from data_provider.data_factory import data_provider
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch, os, time, wandb
import numpy as np
import torch.nn as nn

class ExpMTDP(Exp_Basic):
    def __init__(self, args):
        super(ExpMTDP, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.model == "MDTP":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        val_losses = []
        state_taxi_eval = None
        state_bike_eval = None

        with torch.no_grad():
            for taxi_seq, bike_seq, A_taxi, A_bike, y_taxi, y_bike in vali_loader:
                if state_taxi_eval is None:
                    state_taxi_eval = init_state(self.args)
                    state_bike_eval = init_state(self.args)
                taxi_seq, bike_seq = taxi_seq.to(self.device), bike_seq.to(self.device)
                A_taxi, A_bike     = A_taxi.to(self.device), A_bike.to(self.device)
                y_true = torch.cat([y_taxi, y_bike], dim=-1).to(self.device)
                # 前向并更新 hidden state
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred, (state_taxi_eval, state_bike_eval) = self.model(
                            taxi_seq, bike_seq, A_taxi, A_bike,
                            state_taxi_eval, state_bike_eval
                        )
                        val_losses.append(criterion(pred, y_true).item())
                else:   
                    pred, (state_taxi_eval, state_bike_eval) = self.model(
                        taxi_seq, bike_seq, A_taxi, A_bike,
                        state_taxi_eval, state_bike_eval
                    )
                    val_losses.append(criterion(pred, y_true).item())
        avg_val = np.mean(val_losses)
        return avg_val

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        state_taxi = None
        state_bike = None
        
        y_trues, y_preds = [], []
        with torch.no_grad():
            for taxi_seq, bike_seq, A_taxi, A_bike, y_taxi, y_bike in test_loader:
                if state_taxi is None:
                    state_taxi = init_state(self.args)
                    state_bike = init_state(self.args)
                taxi_seq, bike_seq = taxi_seq.to(self.device), bike_seq.to(self.device)
                A_taxi, A_bike     = A_taxi.to(self.device), A_bike.to(self.device)
                # 前向并更新 hidden state
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred, (state_taxi, state_bike) = self.model(
                            taxi_seq, bike_seq, A_taxi, A_bike,
                            state_taxi, state_bike
                        )
                else:   
                    pred, (state_taxi, state_bike) = self.model(
                        taxi_seq, bike_seq, A_taxi, A_bike,
                        state_taxi, state_bike
                    )
                pred = pred.cpu().numpy()
                true = np.concatenate([y_taxi, y_bike], axis=-1)
                y_preds.append(pred.reshape(-1,4))
                y_trues.append(true.reshape(-1,4))
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        print(f"Predictions shape: {y_preds.shape}, True values shape: {y_trues.shape}")
        print(f"Sample predictions: {y_preds[:5]}")
        print(f"Sample true values: {y_trues[:5]}")

        mae  = mean_absolute_error(y_trues, y_preds)
        rmse = mean_squared_error(y_trues, y_preds)
        mape = np.mean(np.abs((y_trues - y_preds) / (y_trues + 1e-3))) * 100
        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        print(f"Test MAE {mae:.4f}, RMSE {rmse:.4f}, MAPE {mape:.2f}%")
        return metrics

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
            scaler = torch.cuda.amp.GradScaler()
        
        if self.args.model == "MDTP":
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []
                self.model.train()
                epoch_time = time.time()
                state_taxi = None
                state_bike = None

                for i, (taxi_seq, bike_seq, A_taxi, A_bike, y_taxi, y_bike) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    if state_taxi is None:
                        state_taxi = init_state(self.args)
                        state_bike = init_state(self.args)
                    taxi_seq, bike_seq = taxi_seq.to(self.device), bike_seq.to(self.device)
                    A_taxi, A_bike     = A_taxi.to(self.device), A_bike.to(self.device)
                    y_true = torch.cat([y_taxi, y_bike], dim=-1).to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            pred, (state_taxi, state_bike) = self.model(
                                taxi_seq, bike_seq, A_taxi, A_bike,
                                state_taxi, state_bike
                            )

                            loss = criterion(pred, y_true)
                            train_loss.append(loss.item())
                    else:
                        pred, (state_taxi, state_bike) = self.model(
                            taxi_seq, bike_seq, A_taxi, A_bike,
                            state_taxi, state_bike
                        )

                        loss = criterion(pred, y_true)
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
                    # —— 截断梯度，防止跨 batch 反向传播 —— 
                    def detach_state(state):
                        h, c = state
                        return (h.detach(), c.detach())
                    state_taxi = detach_state(state_taxi)
                    state_bike = detach_state(state_bike)

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.mean(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                wandb.log({
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    "test_loss": test_loss
                })
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            return self.model
