from GRUmimic_model import StateClassifier
from utils import MIMICDataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from tqdm import tqdm
from tsexplain import TSExplainer, Trainer
from utils import CPU_Unpickler
import numpy as np
import pickle
from config import (
    MIMIC_NUM_CLASSES,
    MIMIC_NUM_FEATURES,
    MIMIC_MAX_LENGTH,
)

ts_model = StateClassifier(feature_size=31, hidden_size=200, n_state=2)
ts_model.load_state_dict(torch.load('Models/MIMIC/mimic_model_0.pt', map_location=torch.device('cpu')))
ts_model.eval()
data = MIMICDataset('Datasets/MIMIC/inputs.pkl', 'Datasets/MIMIC/true_labels.pkl')


def filter_data():
    with open('Datasets/MIMIC/inputs_0.pkl', 'rb') as fs:
        data_X = CPU_Unpickler(fs).load()
    with open('Datasets/MIMIC/true_labels_0.pkl', 'rb') as fs:
        data_y = CPU_Unpickler(fs).load()

    pred_cls = []

    for i in tqdm(range(data_X.shape[0])):
        y = data_y[i].item()
        if y==1:
            x = data_X[i].unsqueeze(0)
            x = x.permute((0, 2, 1))
            out = ts_model(x)
            cls = torch.argmax(out).item()
            pred_cls.append(cls)

    print(sum(pred_cls)/len(pred_cls))


def evaluate():
    true_cls = []
    pred_cls = []
    loader = DataLoader(data, batch_size=32)
    with torch.no_grad():
        for X, y in tqdm(loader):
            out = ts_model(X)
            cls = torch.argmax(out, dim=1)
            pred_cls.extend(cls.reshape(-1).numpy().tolist())
            true_cls.extend(y.reshape(-1).numpy().tolist())


    print(accuracy_score(true_cls, pred_cls))
    print(roc_auc_score(true_cls, pred_cls))
    print(balanced_accuracy_score(true_cls, pred_cls))


def local_explain_MIMIC():
    loader = DataLoader(data, batch_size=1, shuffle=True)
    all_masks = []
    for X, y in loader:
        mask = None
        expl_model = TSExplainer(ts_model=ts_model, time_steps=MIMIC_MAX_LENGTH, features=MIMIC_NUM_FEATURES,
                                 hidden_state=1024)
        trainer = Trainer(model=expl_model)
        trainer.epochs = 200
        trainer.lmbda = 1.0  # weight on the regularizer # l1 on the latent parameters # do not increase
        trainer.omega = 10000.0  # weight on the loss -- predict the same class as the model to explain
        trainer.gamma = 1.0  # weight on elementwise entropy
        trainer.delta = 1.0  # weight on budget penalty
        cls_dist = ts_model(X).detach().softmax(dim=1)
        inf_cls = torch.argmax(cls_dist, dim=1)
        #print(y, inf_cls)
        trainer.train_local((X, y, mask))

        expl_model.eval()
        with torch.no_grad():
            expl = expl_model.interpret(X)

        #print(torch.sum(expl))
        expl = expl.reshape((X.shape[0], X.shape[1], X.shape[2]))
        all_masks.append({'x': X, 'mask': expl})


    with open('all_masks_local_MIMIC.pkl', 'wb') as fs:
        pickle.dump(all_masks, fs)


def global_explain_MIMIC(val_data=None):
    expl_model = TSExplainer(ts_model=ts_model, time_steps=MIMIC_MAX_LENGTH, features=MIMIC_NUM_FEATURES, hidden_state=1024)
    trainer = Trainer(model=expl_model)
    trainer.epochs = 300
    trainer.lmbda = 1.0  # weight on the regularizer # l1 on the latent parameters # do not increase
    trainer.omega = 1000.0  # weight on the loss -- predict the same class as the model to explain
    trainer.gamma = 100.0  # weight on elementwise entropy
    trainer.delta = 1.0  # weight on budget penalty
    trainer.train_global(val_data)

    all_masks = []
    expl_model.eval()
    loader = DataLoader(data, batch_size=1, shuffle=True)
    for X, y in loader:
        with torch.no_grad():
            expl = expl_model.interpret(X)

        expl = expl.reshape((X.shape[0], X.shape[1], X.shape[2]))
        all_masks.append({'x': X, 'mask': expl})

    with open('all_masks_global_MIMIC.pkl', 'wb') as fs:
        pickle.dump(all_masks, fs)


def compute_average():
    with open('Datasets/MIMIC/inputs_0.pkl', 'rb') as fs:
        data_X = CPU_Unpickler(fs).load()
    #print(data_X.shape)
    # compute this from the train data...
    data_X = data_X.permute(0, 2, 1)
    return torch.mean(data_X, dim=0)


def perturb(x, x_mask, x_avg):
    x_p = x*(1 - x_mask) + x_avg.unsqueeze(0)*x_mask
    return x_p


def evaluate_explainer(path):
    x_avg = compute_average()
    # print(x_avg.shape)
    features = 0
    all_results_MIMIC = []
    with open(path, 'rb') as fs:
        mask_data = CPU_Unpickler(fs).load()

    cnt = 0
    c_pred = 0
    loss = []
    with torch.no_grad():
        for obj in tqdm(mask_data):
            cnt += 1
            x_per = perturb(obj['x'], obj['mask'], x_avg)
            features+=torch.sum(obj['mask'].to(torch.int32)).item()
            #print(obj['x'].shape, obj['mask'].shape, x_avg.shape)
            #print(x_per.shape)
            #mask = None
            out = ts_model(x_per)
            # print(out)
            y = (torch.argmax(out, dim=1)).item()
            out_ = ts_model(obj['x'])
            # print(out_)
            y_or = torch.argmax(out_, dim=1)
            loss.append(F.cross_entropy(out, y_or).item())
            if y == y_or.item():
                c_pred += 1
            all_results_MIMIC.append({'y_or': y_or.item(), 'y_or_dist': out_, 'y': y, 'y_dist': out})

        print(c_pred / cnt)
        print(np.mean(loss))

        with open('all_results_MIMIC.pkl', 'wb') as fs:
            pickle.dump(all_results_MIMIC, fs)


def sparsity():
    with open('all_masks_local_MIMIC.pkl', 'rb') as fs:
        mask_data = CPU_Unpickler(fs).load()
    msk = []
    for i in range(100):
        msk.extend(mask_data[i]['mask'].reshape(-1).numpy().tolist())
    counts, bins = np.histogram(msk, density=True)
    #print(counts)
    plt.stairs(counts, bins)
    plt.show()


if __name__ == '__main__':
    #local_explain_MIMIC()
    #sparsity()
    #global_explain_MIMIC()
    evaluate_explainer(path='all_masks_local_MIMIC.pkl')
    #compute_average()