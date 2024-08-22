# GRU based model for time series prediction
import torch
import torch.nn as nn
from utils import HARDataset, Sleep_EEDDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
from torch.optim import AdamW
import pickle

ckpt_callback = pl.callbacks.ModelCheckpoint(
    dirpath='Models/TimeSeries/GRU/Sleep/',
    monitor='val_loss',
    save_top_k=1
)

from config import (
    HAR_MAX_LENGTH,
    HAR_NUM_CLASSES,
    HAR_NUM_FEATURES,
    SLEEP_NUM_CLASSES,
    SLEEP_NUM_FEATURES,
    SLEEP_MAX_LENGTH,
)

def load_data(fname=None, batch_size=1):
    if fname is None:
        test_data = Sleep_EEDDataset('Datasets/sleepEDF/test.pt')
    else:
        test_data = Sleep_EEDDataset(fname)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return test_loader


class GRU_ts(nn.Module):
    def __init__(self, features: int, hidden_size: int, num_layers: int, num_class: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.i2h = nn.Linear(in_features=features, out_features=hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, num_class)

    def forward(self, inp):
        #print(inp)
        inp = torch.permute(inp, (0, 2, 1))
        #print(inp)
        inp_1 = self.i2h(inp) # inp -> batch x seqlen x features
        h_0 = self.initHidden(batch_size=inp.shape[0])
        _, h_n = self.gru(inp_1, h_0)
        #with open('intm.pkl', 'wb') as fs:
        #    pickle.dump({'inp': inp, 'inp1': inp_1, 'h_n': h_n}, fs)
        #h_n = torch.mean(h_n, dim=0)
        #print(h_n)
        out = self.h2o(h_n[-1])
        return out

    def initHidden(self, batch_size):
        return torch.zeros(self.layers, batch_size, self.hidden_size)


class PL_wrap(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.model = GRU_ts(features=3, hidden_size=100, num_layers=2, num_class=6)
        self.model = GRU_ts(features= HAR_NUM_FEATURES, hidden_size=100, num_layers=2, num_class=HAR_NUM_CLASSES)
        self.loss = nn.CrossEntropyLoss()
        self.accr = torchmetrics.Accuracy(task="multiclass", num_classes=HAR_NUM_CLASSES)
        self.microf1 = torchmetrics.F1Score(task="multiclass", num_classes=HAR_NUM_CLASSES)
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=1)

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999

        opt_t = AdamW(self.model.parameters(), lr=lr, betas=(b1, b2))
        return opt_t

    def forward(self, X):
        out = self.model(X)
        return out

    def training_step(self, batch, batch_idx):
        #X, y, _ = batch
        X, y = batch
        out = self.model(X)
        l = self.loss(out, y)
        return l

    def validation_step(self, batch, batch_idx):
        #X, y, _ = batch
        X, y = batch
        out = self.model(X)
        l = self.loss(out, y)
        self.log('val_loss', l)

    def test_step(self, batch, batch_idx):
        #X, y, _ = batch
        X, y = batch
        # padding_mask = torch.ones(X.shape[0], X.shape[1]).to(torch.int64)
        out = self.model(X)
        out = self.sigmoid(out)
        out = self.softmax(out)
        self.log('Test accuracy', self.accr(out, y.reshape(-1)))
        self.log('Test f1', self.microf1(out, y.reshape(-1)))


def train(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for X, y, _ in train_loader:
        #print(X.shape)
        out = model(X)
        loss = criterion(out, y)
        #print(out)
        print(f"Loss: {loss.detach().item()}")
        #inf_cls = torch.argmax(out.detach(), dim=1)
        #print(y, inf_cls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("------------------------------------")


def train_HAR():
    train_data = HARDataset('Datasets/HAR/train.pt')
    val_data = HARDataset('Datasets/HAR/val.pt')
    test_data = HARDataset('Datasets/HAR/test.pt')
    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    # model = GRU_ts(features=3, hidden_size=100, num_layers=2, num_class=6)

    pl_model = PL_wrap()
    trainer = pl.Trainer(max_epochs=15, callbacks=[ckpt_callback])
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')


def test_HAR():
    test_data = HARDataset('Datasets/HAR/test.pt', mask=False)
    test_loader = DataLoader(test_data, batch_size=1)
    model = PL_wrap.load_from_checkpoint('Models/TimeSeries/GRU/epoch=13-step=2576.ckpt')
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            for _ in range(10):
                out = model(X)
                print(out)
            break


def train_sleepEED():
    data_path = 'Datasets/sleepEDF/test.pt'
    test_loader = load_data(fname=data_path, batch_size=1)
    data_path = 'Datasets/sleepEDF/train.pt'
    train_loader = load_data(fname=data_path, batch_size=100)
    data_path = 'Datasets/sleepEDF/val.pt'
    val_loader = load_data(fname=data_path, batch_size=100)
    pl_model = PL_wrap()
    trainer = pl.Trainer(max_epochs=15, callbacks=[ckpt_callback])
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')


if __name__ == '__main__':
    test_HAR()
