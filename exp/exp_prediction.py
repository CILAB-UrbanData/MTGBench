import torch, os, time, wandb
import numpy as np
import torch.nn as nn
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

class ExpPrediction(Exp_Basic):
    def __init__(self, args):
        super(ExpPrediction, self).__init__(args)
    
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
        if self.args.model == "Trajnet":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion
    
    def vali(self, data_set, data_loader, criterion):
        self.model.eval()
        total_loss = []
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
        
        if self.args.model == "Trajnet":
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []
                self.model.train()
                epoch_time = time.time()
                train_data.on_epoch_start()

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
        
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()

        y_trues, y_preds = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.to(self.args.device)
                # 前向并更新 hidden state
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(inputs)                       
                else:   
                    pred = self.model(inputs)                  
                pred = pred.cpu().numpy().reshape(-1, self.args.pre_steps)
                true = targets.cpu().numpy().reshape(-1, self.args.pre_steps)
                y_preds.append(pred)
                y_trues.append(true)
                
        y_preds = np.vstack(y_preds)
        y_trues = np.vstack(y_trues)
        print(f"Predictions shape: {y_preds.shape}, True values shape: {y_trues.shape}")
        print(f"Sample predictions: {y_preds[:5]}")
        print(f"Sample true values: {y_trues[:5]}")

        mae  = mean_absolute_error(y_trues, y_preds)
        rmse = mean_squared_error(y_trues, y_preds)
        mape = np.mean(np.abs((y_trues - y_preds) / (y_trues + 1e-3))) * 100
        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        print(f"Test MAE {mae:.4f}, RMSE {rmse:.4f}, MAPE {mape:.2f}%")

        f = open(folder_path + 'result.txt', 'w')
        f.write('Test MAE: {:.4f}\n'.format(mae))
        f.write('Test RMSE: {:.4f}\n'.format(rmse))
        f.write('Test MAPE: {:.2f}%\n'.format(mape))
        f.close()

        np.savez(folder_path + 'pred.npz', y_preds)
        np.savez(folder_path + 'true.npz', y_trues)

        return metrics