# main code for time series explanation

from model import TSTransformerEncoderClf
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import HARDataset, Operator
from torch.utils.data import DataLoader
from transformer import model_pl
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ts_model import PL_wrap, GRU_ts
import pickle

from config import (
    HAR_MAX_LENGTH,
    HAR_NUM_CLASSES,
    HAR_NUM_FEATURES,
    SLEEP_NUM_CLASSES,
    SLEEP_NUM_FEATURES,
    SLEEP_MAX_LENGTH,
)


class TSExplainer(nn.Module):
    def __init__(self, ts_model, time_steps: int, features: int, hidden_state: int):
        super().__init__()
        self.ts_model = ts_model
        self.ts_model.eval()
        self.time_steps = time_steps
        self.features = features
        self.hidden_state = hidden_state
        self.operator = Operator(window_size=10, features=features, time_steps=time_steps)
        #self.temp = 2.0
        self.sigmoid = nn.Sigmoid()
        self.lmbda = 1
        self.mlp = nn.Sequential(
            nn.LayerNorm(time_steps*features),
            nn.Linear(time_steps*features, hidden_state),
            nn.ReLU(),
            #nn.LayerNorm(hidden_state),
            #nn.Linear(hidden_state, hidden_state),
            #nn.ReLU(),
            nn.LayerNorm(hidden_state),
            nn.Linear(hidden_state, time_steps*features)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

        self.mlp.apply(init_weights)

    def sample(self, mlp_out, temperature=1.0, bias=0.0):
        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(mlp_out.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = (gate_inputs + mlp_out) / temperature
        sample = torch.sigmoid(gate_inputs)
        return sample

    def forward(self, inp, mask=None, temp=1.0): # with gumbel softmax
        #print(mask)
        batch_size = inp.shape[0]
        inp = inp.reshape(batch_size, -1)
        out = self.mlp(inp)
        samp = self.sample(out, temperature=temp)
        samp = samp.reshape(batch_size, self.features, self.time_steps)
        inp = inp.reshape(batch_size, self.features, self.time_steps)
        #samp_inp = samp * inp
        #samp_inp = samp_inp.reshape(batch_size, self.features, self.time_steps)
        samp_inp = self.operator.hadamard(inp, samp)

        if mask is not None:
            samp_out = self.ts_model(samp_inp, mask)
        else:
            samp_out = self.ts_model(samp_inp)
        if type(samp_out) == tuple:
            samp_out = samp_out[0]
        return out, samp, samp_out

    def interpret(self, inp): # generate explanations
        self.mlp.eval()
        batch_size = inp.shape[0]
        inp = inp.reshape(batch_size, -1)
        out = self.mlp(inp)
        expl = torch.sigmoid(out)
        return expl


class Trainer:
    def __init__(self, model: TSExplainer):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_forecast = nn.MSELoss()
        self.ls = 0
        self.sample_size = 1
        self.budget = 75
        self.temp = (4.0, 1.0)
        self.epochs = 300
        self.temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))
        self.sparsity = 0
        self.lmbda = 1.0 # weight on the regularizer # l1 on the latent parameters # do not increase
        self.omega = 100.0 # weight on the loss -- predict the same class as the model to explain
        self.gamma = 300.0 # weight on elementwise entropy
        self.delta = 1.0 # weight on budget penalty
        self.loss_traj = []
        self.best_epoch = 0
        self.latent = []

    def train_local_w_budget(self, data):
        optimizer = torch.optim.AdamW(self.model.mlp.parameters(), lr=0.0001)
        X, y, mask = data
        mask = None
        for i in range(self.epochs):
            loss = 0.0
            temp = self.temp_schedule(i)
            for k in range(self.sample_size):
                out, samp, samp_out = self.model(X, mask, temp)
                #torch.save(self.model.state_dict(), f'Models/Explmodel/expl_budget_{i}.ckpt')
                loss_1 = self.omega * self.criterion(samp_out, y)  # when budget is specified
                loss_2 = self.delta * F.relu(torch.sum(samp) - self.budget)
                loss += loss_1 + loss_2
            self.loss_traj.append(loss.item())
            print(f"Loss at epoch {i}: {loss_1.item()}, {loss_2.item()}, {samp_out}")
            print("============================================================")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train_local(self, data, task='cls'):
        optimizer = torch.optim.AdamW(self.model.mlp.parameters(), lr=0.001)
        X, y, mask = data
        eps = 0.0001
        #mask = None
        for i in range(self.epochs):
            temp = self.temp_schedule(i)
            loss = torch.FloatTensor([0]).detach()
            for k in range(self.sample_size):
                out, samp, samp_out = self.model(X, mask, temp)
                #torch.save(self.model.state_dict(), f'Models/Explmodel/expl_local_{i}.ckpt')
                if task == 'cls':
                    loss_1 = self.omega * self.criterion(samp_out, y)
                elif task == 'forecast':
                    loss_1 = self.omega * self.criterion_forecast(samp_out, y)
                else: # add other tasks...
                    loss_1 = None
                loss_2 = self.lmbda * torch.sum(samp)
                loss_3 = self.gamma * (-1) * torch.mean(torch.log(samp+eps)*(samp + eps) +
                                              torch.log(1 - samp + eps)*(1 - samp + eps))

                loss += loss_1 + loss_2 + loss_3
            #print(f"Loss at epoch {i}: {loss_1}, {loss_2}, {loss_3}, {samp_out}")
            #print("============================================================")
            self.loss_traj.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #break

    def train_global(self, test_data, task='cls'):
        optimizer = torch.optim.AdamW(self.model.mlp.parameters(), lr=0.001, weight_decay=0.1)
        eps = 0.0001
        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
        for i in tqdm(range(self.epochs), position=0, desc="epochs", leave=False, colour='green'):
            temp = self.temp_schedule(i)
            #loss1 = []
            #loss2 = []
            #loss3 = []
            #loss = torch.FloatTensor([0]).detach()
            for data in tqdm(test_loader, position=1, desc="batch", leave=False, colour='red'):
                loss = torch.FloatTensor([0]).detach()
                X, y, mask = data
                #mask = None
                for k in range(self.sample_size):
                    out, samp, samp_out = self.model(X, mask, temp)
                    #loss_1 = self.omega * self.criterion(samp_out, y)
                    if task == 'cls':
                        loss_1 = self.omega * self.criterion(samp_out, y)
                    elif task == 'forecast':
                        loss_1 = self.omega * self.criterion_forecast(samp_out, y)
                    else:  # add other tasks...
                        loss_1 = None
                    loss_2 = self.lmbda * torch.sum(samp)
                    loss_3 = self.gamma * (-1) * torch.mean(torch.log(samp + eps) * (samp + eps) +
                                                            torch.log(1 - samp + eps) * (1 - samp + eps))

                    #loss1.append(loss_1.detach().item())
                    #loss2.append(loss_2.detach().item())
                    #loss3.append(loss_3.detach().item())
                    loss_d = loss_1 + loss_2 + loss_3
                    loss += loss_d

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #print(loss1)
            #print(f"Loss at epoch {i}: {np.mean(loss1)}, {np.mean(loss2)}, {np.mean(loss3)}")
            #self.loss_traj.append((np.mean(loss1), np.mean(loss2), np.mean(loss3)))
        torch.save(self.model.state_dict(), f'Models/Global_models/HAR/expl_global_HAR.ckpt')

    def train_local_forecasting(self, data):
        optimizer = torch.optim.AdamW(self.model.mlp.parameters(), lr=0.001)
        X, y = data
        eps = 0.0001
        mask = None
        for i in tqdm(range(self.epochs)):
            temp = self.temp_schedule(i)
            out, samp, samp_out = self.model(X, mask, temp)
            torch.save(self.model.state_dict(), f'Models/Explmodel/expl_local_{i}.ckpt')
            loss_1 = self.omega * self.criterion_forecast(samp_out, y)
            loss_2 = self.lmbda * torch.sum(samp)
            loss_3 = self.gamma * (-1) * torch.mean(torch.log(samp + eps) * (samp + eps) +
                                                    torch.log(1 - samp + eps) * (1 - samp + eps))

            loss = loss_1 + loss_2 + loss_3
            print(f"Loss at epoch {i}: {loss_1}, {loss_2}, {loss_3}, {samp_out}")
            # print("============================================================")
            self.loss_traj.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def plot_loss(self):
        ls_0 = np.array([self.loss_traj[i][0] for i in range(len(self.loss_traj))])
        ls_1 = np.array([self.loss_traj[i][1] for i in range(len(self.loss_traj))])
        ls_2 = np.array([self.loss_traj[i][2] for i in range(len(self.loss_traj))])
        x_axis = np.array([i for i in range(len(self.loss_traj))])
        plt.yscale('log')
        plt.plot(x_axis, ls_0)
        plt.plot(x_axis, ls_1)
        plt.plot(x_axis, ls_2)
        plt.show()


def interpret(model, inp, mask):
    expl, inf_e_cls = model.interpret(inp, mask)
    return expl, inf_e_cls


def test(ts_model):
    test_data = HARDataset('Datasets/HAR/test.pt')
    test_loader = DataLoader(test_data, batch_size=32)
    pred_cls = []
    gt_cls = []
    for X, y, mask in tqdm(test_loader):
        out = ts_model(X, mask)
        cls_ = torch.argmax(out, dim=1)
        pred_cls.extend(cls_.reshape(-1).numpy().tolist())
        gt_cls.extend(y.reshape(-1).numpy().tolist())

    print(accuracy_score(gt_cls, pred_cls))


def evaluate_expl(X, mask):
    ts_model = model_pl.load_from_checkpoint('Models/TimeSeries/epoch=18-step=3496.ckpt')
    ts_model.eval()
    model_best = TSExplainer(ts_model=ts_model, time_steps=128, features=3)
    # model_best.eval()
    for i in range(1000):
        model_best.load_state_dict(torch.load(f'Models/Explmodel/expl_local_{i}.ckpt'))
        model_best.eval()
        # print(X)
        expl, expl_score, inf_e_cls = model_best.interpret(X, mask)
        print(inf_e_cls, torch.sum(expl))
        # print(torch.max(expl))
        # print(torch.sum(expl))

    # print(trainer.loss_traj)


def load_ts_model_data(dataset):
    if dataset == 'HAR':
        ts_model = PL_wrap.load_from_checkpoint('Models/TimeSeries/GRU/epoch=13-step=2576.ckpt')
        ts_model.eval()

        # test(ts_model=ts_model)
        test_data = HARDataset('Datasets/HAR/test.pt')
        test_loader = DataLoader(test_data, batch_size=1)

    else:
        ts_model, test_loader = None, None

    return ts_model, test_loader


def local_training(dataset):
    ts_model, test_loader = load_ts_model_data(dataset)
    if dataset == 'HAR':
        exp_model = TSExplainer(ts_model=ts_model, time_steps=HAR_MAX_LENGTH, features=HAR_NUM_FEATURES, hidden_state=1024)
        trainer = Trainer(model=exp_model)
        for X, y in test_loader:
            mask = None
            cls_dist = ts_model(X).detach().softmax(dim=1)
            inf_cls = torch.argmax(cls_dist, dim=1)
            print(y, inf_cls)
            # exp_model(X, mask, 1.0)
            trainer.train_local((X, y, mask))
            #trainer.train_local_w_budget((X, y, mask), cls_dist)
            # print("===============================================")
            # evaluate_expl(X, mask)
            # out = exp_model(X, mask)
            # print(out.shape)
            break

    elif dataset == 'sleepEDF':
        exp_model = TSExplainer(ts_model=ts_model, time_steps=SLEEP_MAX_LENGTH, features=SLEEP_NUM_FEATURES,
                                hidden_state=1024)
        trainer = Trainer(model=exp_model)
        for X, y in test_loader:
            cls_dist = ts_model(X)[0].detach().softmax(dim=1)
            inf_cls = torch.argmax(cls_dist, dim=1)
            print(y.item(), inf_cls.item())
            trainer.train_local((X, y, None), cls_dist)
            break


def global_training():
    ts_model = PL_wrap.load_from_checkpoint('Models/TimeSeries/GRU/epoch=13-step=2576.ckpt')
    ts_model.eval()
    # train the explainer on the validation set....
    # test on the test set
    # test(ts_model=ts_model)
    val_data = HARDataset('Datasets/HAR/val.pt')
    exp_model = TSExplainer(ts_model=ts_model, time_steps=128, features=3, hidden_state=1024)
    trainer = Trainer(model=exp_model)
    trainer.train_global(val_data)
    #trainer.plot_loss()


if __name__ == '__main__':
   local_training(dataset='HAR')
   #global_training()

