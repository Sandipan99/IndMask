from transformer import model_pl
from utils import HARDataset
from torch.utils.data import DataLoader
from tsexplain import TSExplainer, Trainer
import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import (
    HAR_MAX_LENGTH,
    HAR_NUM_CLASSES,
    HAR_NUM_FEATURES
)

test_data = HARDataset('Datasets/HAR/test.pt', mask=True)
val_data = HARDataset('Datasets/HAR/val.pt', mask=True)
ts_model = model_pl.load_from_checkpoint('Models/HAR/epoch=18-step=3496.ckpt')
ts_model.eval()


def local_explain_HAR():
    all_masks = []
    loader = DataLoader(test_data, batch_size=1, shuffle=False)
    for X, y, mask in tqdm(loader, position=0, desc="item", leave=False, colour='green'):
        expl_model = TSExplainer(ts_model=ts_model, time_steps=HAR_MAX_LENGTH, features=HAR_NUM_FEATURES,
                                 hidden_state=1024)
        trainer = Trainer(model=expl_model)
        trainer.epochs = 200
        trainer.lmbda = 1.0  # weight on the regularizer # l1 on the latent parameters # do not increase
        trainer.omega = 1000.0  # weight on the loss -- predict the same class as the model to explain
        trainer.gamma = 100.0  # weight on elementwise entropy
        trainer.delta = 1.0  # weight on budget penalty
        cls_dist = ts_model(X, mask).detach().softmax(dim=1)
        inf_cls = torch.argmax(cls_dist, dim=1)
        #print(y, inf_cls)
        trainer.train_local((X, y, mask))

        expl_model.eval()
        with torch.no_grad():
            expl = expl_model.interpret(X)
            #print(torch.sum(expl))
            expl = expl.reshape((X.shape[0], X.shape[1], X.shape[2]))

        all_masks.append({'x': X.cpu(), 'mask': expl.cpu()})

    with open('all_masks_local_HAR.pkl', 'wb') as fs:
        pickle.dump(all_masks, fs)


def compute_average():
    data = torch.load('Datasets/HAR/train.pt')
    # print(data['samples'].shape)
    feat_avg = torch.mean(data['samples'], dim=0)
    return feat_avg


def perturb(x, x_mask, x_avg):
    x_p = x*(1 - x_mask) + x_avg.unsqueeze(0)*x_mask
    return x_p


def evaluate_explainer(path):
    x_avg = compute_average()
    #print(x_avg.shape)
    all_results_HAR = []
    with open(path, 'rb') as fs:
        mask_data = pickle.load(fs)

    loss = []
    y_o = []
    y_p = []
    features = 0
    with torch.no_grad():
        for obj in tqdm(mask_data):
            #print(obj['x'].shape)
            #print(obj['mask'].shape)
            x_per = perturb(obj['x'], obj['mask'], x_avg)
            #print(x_per.shape)
            features += torch.sum(obj['mask'].reshape(-1)).item()
            mask = torch.ones((1, HAR_MAX_LENGTH)).to(torch.int32)
            out = ts_model(x_per, mask)
            #print(out)
            y = (torch.argmax(out, dim=1)).item()
            y_p.append(y)
            out_ = ts_model(obj['x'], mask)
            #print(out_)
            y_or = torch.argmax(out_, dim=1)
            y_o.append(y_or.item())
            ls = F.cross_entropy(out, y_or).item()
            loss.append(ls)
            all_results_HAR.append({'y_or': y_or.item(), 'y_or_dist': out_, 'y': y, 'y_dist': out})

    print(np.mean(loss))
    print(accuracy_score(y_o, y_p))
    print(precision_score(y_o, y_p, average='macro'))
    print(recall_score(y_o, y_p, average='macro'))
    print(features)

    #with open('all_results_HAR.pkl', 'wb') as fs:
    #    pickle.dump(all_results_HAR, fs)


def global_explain_HAR():
    expl_model = TSExplainer(ts_model=ts_model, time_steps=HAR_MAX_LENGTH, features=HAR_NUM_FEATURES, hidden_state=1024)
    trainer = Trainer(model=expl_model)
    trainer.epochs = 300
    trainer.lmbda = 1.0  # weight on the regularizer # l1 on the latent parameters # do not increase
    trainer.omega = 1000.0  # weight on the loss -- predict the same class as the model to explain
    trainer.gamma = 100.0  # weight on elementwise entropy
    trainer.delta = 1.0  # weight on budget penalty
    trainer.train_global(val_data)
    all_masks = []
    expl_model.eval()
    loader = DataLoader(test_data, batch_size=1, shuffle=True)
    with torch.no_grad():
        for X, y, mask in loader:
            expl = expl_model.interpret(X)
            # print(torch.sum(expl))
            expl = expl.reshape((X.shape[0], X.shape[1], X.shape[2]))
            all_masks.append({'x': X, 'mask': expl})


    with open('all_masks_global_HAR.pkl', 'wb') as fs:
        pickle.dump(all_masks, fs)


if __name__ == '__main__':
    local_explain_HAR()
    #global_explain_HAR()
    #evaluate_explainer(path='all_masks_global_HAR.pkl')