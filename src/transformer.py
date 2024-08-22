#from utils import HARDataset, MIMICDataset
from model import TSTransformerEncoderClf
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from utils import load_data, HARDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm


ckpt_callback = pl.callbacks.ModelCheckpoint(
    dirpath='Models/TimeSeries/',
    monitor='val_loss',
    save_top_k=1
)


class model_pl(pl.LightningModule):
    def __init__(self):
        super(model_pl, self).__init__()
        #### for HAR ########

        self.ts_model = TSTransformerEncoderClf(feat_dim=3,
                                                max_len=206,
                                                d_model=128,
                                                n_heads=4,
                                                num_layers=4,
                                                dim_feedforward=1024,
                                                repr_dim=512,
                                                num_classes=6)
        '''
        #####################
        self.ts_model = TSTransformerEncoderClf(feat_dim=31,
                                                max_len=48,
                                                d_model=128,
                                                n_heads=4,
                                                num_layers=4,
                                                dim_feedforward=1024,
                                                repr_dim=512,
                                                num_classes=2)
        '''
        self.loss = nn.CrossEntropyLoss()
        self.accr = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=6)
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=1)

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999

        opt_t = AdamW(self.ts_model.parameters(), lr=lr, betas=(b1, b2))
        return opt_t

    def forward(self, X, mask):
        X = X.permute(0, 2, 1)
        #padding_mask = torch.ones(X.shape[0], X.shape[1]).to(torch.int64)
        out = self.ts_model(X, mask)
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        #print(X.shape, y.shape, padding_mask.shape)
        X = X.permute(0, 2, 1)
        padding_mask = torch.ones(X.shape[0], X.shape[1]).to(torch.int64)
        #print(X.shape, padding_mask.shape)
        out = self.ts_model(X, padding_mask)
        y = y.long()
        l = self.loss(out, y)
        return l

    def validation_step(self, batch, batch_idx):
        X, y, padding_mask = batch
        X = X.permute(0, 2, 1)
        #padding_mask = torch.ones(X.shape[0], X.shape[1]).to(torch.int64)
        out = self.ts_model(X, padding_mask)
        l = self.loss(out, y)
        self.log('val_loss', l)

    def test_step(self, batch, batch_idx):
        X, y, padding_mask = batch
        X = X.permute(0, 2, 1)
        #padding_mask = torch.ones(X.shape[0], X.shape[1]).to(torch.int64)
        out = self.ts_model(X, padding_mask)
        out = self.sigmoid(out)
        out = self.softmax(out)
        self.log('Test accuracy', self.accr(out, y.reshape(-1)))
        self.log('Test AUROC', self.auroc(out, y.reshape(-1)))


def train(model, data):
    loader = DataLoader(data, batch_size=32, shuffle=True)
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    optimizer = AdamW(model.parameters(), lr=lr, betas=(b1, b2))
    criterion = nn.CrossEntropyLoss()
    for X, y in loader:
        X = X.permute(0, 2, 1)
        padding_mask = torch.ones(X.shape[0], X.shape[1]).to(torch.int64)
        out = model(X, padding_mask)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break


def train_HAR():
    train_data = HARDataset('Datasets/HAR/train.pt')
    val_data = HARDataset('Datasets/HAR/val.pt')
    test_data = HARDataset('Datasets/HAR/test.pt')
    model = model_pl()
    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    # trainer = pl.Trainer(max_epochs=20, callbacks=[ckpt_callback], accelerator='gpu', device=4)
    trainer = pl.Trainer(max_epochs=1, callbacks=[ckpt_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')


def test_HAR():
    test_data = HARDataset('Datasets/HAR/test.pt', mask=True)
    test_loader = DataLoader(test_data, batch_size=32)
    model = model_pl.load_from_checkpoint('Models/TimeSeries/epoch=18-step=3496.ckpt')
    model.eval()
    pred_cls = []
    true_cls = []
    with torch.no_grad():
        for X, y, mask in tqdm(test_loader):
            out = model(X, mask)
            pred_cls.extend(torch.argmax(out, dim=1).reshape(-1).numpy().tolist())
            true_cls.extend(y.reshape(-1).numpy().tolist())

    print(accuracy_score(true_cls, pred_cls))


if __name__ == '__main__':
    test_HAR()
    '''
    mimic_path = '/home/nasr/GitHub/Xai_Proj/Dynamask/data/mimic/'
    p_data, train_loader, valid_loader, test_loader = load_data(
            batch_size=32, path=mimic_path, task="mortality", cv=0
        )
    feature_size = p_data.feature_size
    class_weight = p_data.pos_weight
    #train_HAR()
    #train_data = train_loader
    model = model_pl()
    #train_loader = DataLoader(train_data, batch_size=32)
    # trainer = pl.Trainer(max_epochs=20, callbacks=[ckpt_callback], accelerator='gpu', device=4)
    trainer = pl.Trainer(max_epochs=1, callbacks=[ckpt_callback])
    trainer.fit(model, train_dataloaders=train_loader)
    '''
