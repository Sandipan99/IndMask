import os
import pickle as pkl
import pickle
import warnings

from torch.utils.data import Dataset
import torch
import io
import pickle

import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import torch
import torch.utils.data as utils
from sklearn.metrics import classification_report, precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

#from fit.TSX.models import PatientData

import pytorch_lightning as pl

# np.set_printoptions(threshold=sys.maxsize)
# sns.set()

line_styles_map = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
marker_styles_map = [
    "o",
    "v",
    "^",
    "*",
    "+",
    "p",
    "8",
    "h",
    "o",
    "v",
    "^",
    "*",
    "+",
    "p",
    "8",
    "h",
    "o",
    "v",
    "^",
    "*",
    "+",
    "p",
    "8",
    "h",
]

# Ignore sklearn warnings caused by ill-defined precision score (caused by single class prediction)
warnings.filterwarnings("ignore")

intervention_list = [
    "vent",
    "vaso",
    "adenosine",
    "dobutamine",
    "dopamine",
    "epinephrine",
    "isuprel",
    "milrinone",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "colloid_bolus",
    "crystalloid_bolus",
    "nivdurations",
]



class PatientData:
    """Dataset of patient vitals, demographics and lab results
    Args:
        root: Root directory of the pickled dataset
        train_ratio: train/test ratio
        shuffle: Shuffle dataset before separating train/test
        transform: Preprocessing transformation on the dataset
    """

    def __init__(
        self, root, train_ratio=0.8, shuffle=False, random_seed="1234", transform="normalize", task="mortality"
    ):
        self.data_dir = os.path.join(root, "patient_vital_preprocessed.pkl")
        self.train_ratio = train_ratio
        self.random_seed = random.seed(random_seed)
        self.task = task
        self.pos_weight = None

        if not os.path.exists(self.data_dir):
            raise RuntimeError("Dataset not found")
        with open(self.data_dir, "rb") as f:
            self.data = pickle.load(f)

        if os.path.exists(os.path.join(root, "patient_interventions.pkl")):
            with open(os.path.join(root, "patient_interventions.pkl"), "rb") as f:
                self.intervention = pickle.load(f)

        self.n_train = int(self.train_ratio * len(self.data))
        if shuffle:
            inds = np.arange(len(self.data))
            random.shuffle(inds)
            self.data = self.data[inds]
            self.intervention = self.intervention[inds, :, :]

        if self.task == "mortality":
            X = np.array([x for (x, y, z) in self.data])
            self.train_data = X[0 : self.n_train]
            self.test_data = X[self.n_train :]
            self.train_label = np.array([y for (x, y, z) in self.data[0 : self.n_train]])
            self.test_label = np.array([y for (x, y, z) in self.data[self.n_train :]])
            self.train_missing = np.array([np.mean(z) for (x, y, z) in self.data[0 : self.n_train]])
            self.test_missing = np.array([np.mean(z) for (x, y, z) in self.data[self.n_train :]])

        elif self.task == "intervention":
            print("predicting intervention")
            if 0:  # suresh et al - predicts intervention state (onset, wean, stay off, stay on)
                X, y, z = self.__preprocess_predict_int__()
                choose_int = 0
                self.intervention = y[:, choose_int, :]
                self.pos_weight = np.sum(self.intervention) / np.sum(self.intervention, 0)
                self.pos_weight = 1 * self.pos_weight / self.pos_weight.sum()

            else:  # predicts interventions
                n = 3
                feat_hist = np.sum(np.sum(self.intervention, 2), 0)
                feat_idx = np.argsort(feat_hist)[::-1][:n]
                intervention_int = np.zeros((len(self.intervention), n + 1, self.intervention.shape[-1]))
                intervention_int[:, :n, :] = self.intervention[:, feat_idx, :]
                intervention_int[:, -1, :] = 1 - (np.sum(intervention_int[:, :n, :], 1) > 0).astype(int)
                self.intervention = intervention_int
                self.pos_weight = 1 / (
                    np.sum(np.sum(self.intervention, 2), 0) / (self.intervention.shape[0] * self.intervention.shape[-1])
                )
                X = np.array([x for (x, y, z) in self.data])
                z = np.array([z for (x, y, z) in self.data])

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=88)
            for train_idx, test_idx in sss.split(X[:, :, 0], self.intervention[:, :, 0]):
                self.train_data = X[train_idx]
                self.test_data = X[test_idx]
                self.train_intervention = self.intervention[train_idx]
                self.test_intervention = self.intervention[test_idx]
                self.train_label = self.train_intervention
                self.test_label = self.test_intervention
                missing = np.array([np.mean(zz) for zz in z])
                self.train_missing = missing[train_idx]
                self.test_missing = missing[test_idx]
        self.n_train = self.train_data.shape[0]
        self.n_test = self.test_data.shape[0]
        self.feature_size = len(self.data[0][0])
        self.time = len(self.data[0][0][0])
        self.len_of_stay = self.train_data.shape[-1]
        if transform == "normalize":
            self.normalize()

    def __getitem__(self, index):
        signals, target = self.data[index]
        return signals, target

    def __len__(self):
        return len(self.data)

    def __preprocess_predict_int__(self):
        "This replicates preprocessing of suresh et al for intervention prediction"
        X_orig = np.array([x for (x, y, z) in self.data])
        y_orig = self.intervention
        z_orig = np.array([z for (x, y, z) in self.data])

        X = []
        y = []
        z = []
        T = 24
        gaptime = 4
        window = 6
        stride = 6
        for h in range(0, self.time, 6):
            if h + T + gaptime + window >= self.time - 1:
                break

            X.append(X_orig[:, :, h : h + T])
            z.append(z_orig)
            y_t1 = y_orig[:, :, range(h + T + gaptime, h + T + gaptime + window)]
            n_ints = self.intervention.shape[1]
            y_label = np.zeros((X_orig.shape[0], n_ints, 4))
            for f in range(n_ints):
                onset_patients = np.where((y_t1[:, f, 0] == 0) & (y_t1[:, f, -1] == 1))[0]
                y_label[onset_patients, f, 0] = 1
                wean_patients = np.where((y_t1[:, f, 0] == 1) & (y_t1[:, f, -1] == 0))[0]
                y_label[wean_patients, f, 1] = 1
                stay_on_patients = np.where((y_t1[:, f, 0] == 1) & (y_t1[:, f, -1] == 1))[0]
                y_label[stay_on_patients, f, 2] = 1
                stay_off_patients = np.where((y_t1[:, f, 0] == 0) & (y_t1[:, f, -1] == 0))[0]
                y_label[stay_off_patients, f, 3] = 1
            y.append(y_label)

        X = np.array(X)
        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3]))
        y = np.array(y)
        y = y.reshape((y.shape[0] * y.shape[1], y.shape[2], y.shape[3]))
        z = np.array(z)
        z = z.reshape((z.shape[0] * z.shape[1], -1))
        return X, y, z

    def normalize(self):  # TODO: Have multiple normalization option or possibly take in a function for the transform
        """ Calculate the mean and std of each feature from the training set
        """
        d = [x.T for x in self.train_data]
        d = np.stack(d, axis=0)
        self.feature_max = np.tile(np.max(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        self.feature_min = np.tile(np.min(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        self.feature_means = np.tile(np.mean(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        self.feature_std = np.tile(np.std(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        self.train_data = np.array(
            [
                np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std)
                for x in self.train_data
            ]
        )
        self.test_data = np.array(
            [
                np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std)
                for x in self.test_data
            ]
        )
        # self.train_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.train_data])
        # self.test_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.test_data])


def load_data(batch_size, path="./data/", **kwargs):
    transform = kwargs["transform"] if "transform" in kwargs.keys() else "normalize"
    task = kwargs["task"] if "task" in kwargs.keys() else "mortality"
    p_data = PatientData(path, task=task, shuffle=False, transform=transform)
    test_bs = kwargs["test_bs"] if "test_bs" in kwargs.keys() else None
    train_pc = kwargs["train_pc"] if "train_pc" in kwargs.keys() else 1.0

    features = kwargs["features"] if "features" in kwargs.keys() else range(p_data.train_data.shape[1])
    p_data.train_data = p_data.train_data[:, features, :]
    p_data.test_data = p_data.test_data[:, features, :]

    p_data.feature_size = len(features)
    n_train = int(0.9 * p_data.train_data.shape[0])
    if "cv" in kwargs.keys():
        if task == "mortality":
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_idx, valid_idx = list(kf.split(p_data.train_data))[kwargs["cv"]]
        else:
            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=88)
            train_idx, valid_idx = list(sss.split(p_data.train_data[:, :, -1], p_data.train_label[:, :, -1]))[
                kwargs["cv"]
            ]
    else:
        if task == "mortality":
            train_idx = range(n_train)
            valid_idx = range(n_train, p_data.n_train)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=88)
            train_idx, valid_idx = list(sss.split(p_data.train_data[:, :, -1], p_data.train_label[:, :, -1]))[0]

    train_dataset = utils.TensorDataset(
        torch.Tensor(p_data.train_data[train_idx, :, :]), torch.Tensor(p_data.train_label[train_idx])
    )

    valid_dataset = utils.TensorDataset(
        torch.Tensor(p_data.train_data[valid_idx, :, :]), torch.Tensor(p_data.train_label[valid_idx])
    )
    test_dataset = utils.TensorDataset(torch.Tensor(p_data.test_data), torch.Tensor(p_data.test_label))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)  # p_data.n_train - int(0.8 * p_data.n_train))

    if test_bs is not None:
        test_loader = DataLoader(test_dataset, batch_size=test_bs)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(p_data.test_data))

    if task == "mortality":
        print(
            "Train set: ",
            np.count_nonzero(p_data.train_label[0 : int(0.8 * p_data.n_train)]),
            "patient who died out of %d total" % (int(0.8 * p_data.n_train)),
            "(Average missing in train: %.2f)" % (np.mean(p_data.train_missing[0 : int(0.8 * p_data.n_train)])),
        )
        print(
            "Valid set: ",
            np.count_nonzero(p_data.train_label[int(0.8 * p_data.n_train) :]),
            "patient who died out of %d total" % (len(p_data.train_label[int(0.8 * p_data.n_train) :])),
            "(Average missing in validation: %.2f)" % (np.mean(p_data.train_missing[int(0.8 * p_data.n_train) :])),
        )
        print(
            "Test set: ",
            np.count_nonzero(p_data.test_label),
            "patient who died  out of %d total" % (len(p_data.test_data)),
            "(Average missing in test: %.2f)" % (np.mean(p_data.test_missing)),
        )
    return p_data, train_loader, valid_loader, test_loader


def train_har(model, train_loader,device, optimizer, criterion, valid_loader, test_loader ):
    # Train the LSTM model
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs.squeeze(), labels.long())
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                val_loss = criterion(outputs.squeeze(), labels.long())
                val_losses.append(val_loss.item())
                
                _, predictions = torch.max(outputs, 1)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        eval_har(model, test_loader,device)
        
def eval_har(model,test_loader, device):
    # Evaluate the model on the test set
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            _, predictions = torch.max(outputs, 1)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_targets, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")



def logistic(x):
    return 1.0 / (1 + np.exp(-1 * x))


def top_risk_change(exp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    span = []
    testset = list(exp.test_loader.dataset)
    for i, (signal, label) in enumerate(testset):
        exp.risk_predictor.load_state_dict(torch.load("./ckpt/mimic/risk_predictor.pt"))
        exp.risk_predictor.to(device)
        exp.risk_predictor.eval()
        risk = []
        for t in range(1, 48):
            risk.append(exp.risk_predictor(signal[:, 0:t].view(1, signal.shape[0], t).to(device)).item())
        span.append((i, max(risk) - min(risk)))
    span.sort(key=lambda pair: pair[1], reverse=True)
    print([x[0] for x in span[0:300]])


def test_cond(mean, covariance, sig_ind, x_ind):
    x_ind = x_ind.unsqueeze(-1)
    mean_1 = torch.cat((mean[:, :sig_ind], mean[:, sig_ind + 1 :]), 1).unsqueeze(-1)
    cov_1_2 = torch.cat(([covariance[:, 0:sig_ind, sig_ind], covariance[:, sig_ind + 1 :, sig_ind]]), 1).unsqueeze(-1)
    cov_2_2 = covariance[:, sig_ind, sig_ind]
    cov_1_1 = torch.cat(([covariance[:, 0:sig_ind, :], covariance[:, sig_ind + 1 :, :]]), 1)
    cov_1_1 = torch.cat(([cov_1_1[:, :, 0:sig_ind], cov_1_1[:, :, sig_ind + 1 :]]), 2)
    mean_cond = mean_1 + torch.bmm(cov_1_2, (x_ind - mean[:, sig_ind]).unsqueeze(-1)) / cov_2_2
    covariance_cond = cov_1_1 - torch.bmm(cov_1_2, torch.transpose(cov_1_2, 2, 1)) / cov_2_2
    return mean_cond, covariance_cond


def shade_state_state_data(state_subj, t, ax, data="simulation"):
    cmap = plt.get_cmap("tab10")
    # Shade the state on simulation data plots
    for ttt in range(t[0], len(t)):
        if state_subj[ttt] == 0:
            ax.axvspan(ttt + 1, ttt, facecolor="blue", alpha=0.3)
        elif state_subj[ttt] == 1:
            ax.axvspan(ttt + 1, ttt, facecolor="green", alpha=0.3)
        elif state_subj[ttt] == 2:
            ax.axvspan(ttt + 1, ttt, facecolor="orange", alpha=0.3)


def shade_state(gt_importance_subj, t, ax, data="simulation"):
    cmap = plt.get_cmap("tab10")
    # Shade the state on simulation data plots
    if gt_importance_subj.shape[0] >= 3:
        gt_importance_subj = gt_importance_subj.transpose(1, 0)

    if not data == "simulation_spike":
        prev_color = "g" if np.argmax(gt_importance_subj[:, 1]) < np.argmax(gt_importance_subj[:, 2]) else "y"
        print("######################", t[1])
        for ttt in range(t[1], t[-1]):
            # state = np.argmax(gt_importance_subj[ttt, :])
            # ax.axvspan(ttt - 1, ttt, facecolor=cmap(state), alpha=0.3)
            if gt_importance_subj[ttt, 1] == 1:
                ax.axvspan(ttt - 1, ttt, facecolor="g", alpha=0.3)
                prev_color = "g"
            elif gt_importance_subj[ttt, 2] == 1:
                ax.axvspan(ttt - 1, ttt, facecolor="y", alpha=0.3)
                prev_color = "y"
            elif not prev_color is None:  # noqa: E714
                ax.axvspan(ttt - 1, ttt, facecolor=prev_color, alpha=0.3)


#___________________________________________________
def load_har_data(batch_size=100, datapath="", data_type="har", percentage=1.0, **kwargs):
    datapath = "/home/nasr/GitHub/Xai_Proj/data_dynmsk/har"
    with open(os.path.join(datapath, "x_train.pkl"), "rb") as f:
        x_train = pickle.load(f)  
    with open(os.path.join(datapath, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(datapath, "x_test.pkl"), "rb") as f:
        x_test = pickle.load(f)  
    with open(os.path.join(datapath, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    features = kwargs["features"] if "features" in kwargs.keys() else list(range(x_test.shape[1]))
    test_bs = kwargs["test_bs"] if "test_bs" in kwargs.keys() else None
    #--------------
    '''
    x_train = x_train[:50]
    y_train = y_train[:50]
    x_test = x_test[:50]
    y_test = y_test[:50]
    '''
    #-------------
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]
    
    total_sample_n = int(len(x_train) * percentage)
    x_train = x_train[:total_sample_n]
    y_train = y_train[:total_sample_n]
    n_train = int(0.8 * len(x_train))
    x_train = x_train[:, features, :]
    x_test = x_test[:, features, :]
    

    n_train = int(0.8 * len(x_train))
    if "cv" in kwargs.keys():
        print("cv : ", kwargs["cv"])
        kf = KFold(n_splits=5, shuffle=True, random_state=88)
        train_idx, valid_idx = list(kf.split(x_train))[kwargs["cv"]]
    else:
        train_idx = range(n_train)
        valid_idx = range(n_train, len(x_train))
        
    
    train_dataset = utils.TensorDataset(torch.Tensor.float(x_train[train_idx, :, :]), torch.Tensor.float(y_train[train_idx]))
    valid_dataset = utils.TensorDataset(torch.Tensor.float(x_train[valid_idx, :, :]), torch.Tensor.float(y_train[valid_idx]))
    
    print()
    test_dataset = utils.TensorDataset(torch.Tensor.float(x_test[:, :, :]), torch.Tensor.float(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=len(x_train) - int(0.8 * n_train))
    if test_bs is not None:
        test_loader = DataLoader(test_dataset, batch_size=test_bs)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(x_test))
    
    return np.concatenate([x_train, x_test]), train_loader, valid_loader, test_loader


def train_har(model, train_loader,device, optimizer, criterion, valid_loader, test_loader,cv ):
    # Train the LSTM model
    num_epochs = 100
    #Define early stopping parameters
    patience = 15
    best_loss = float("inf")
    current_patience = 0
    data_name = "har"
    

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs.squeeze(), labels.long())
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
            
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                val_loss = criterion(outputs.squeeze(), labels.long())
                val_losses.append(val_loss.item())
                
                _, predictions = torch.max(outputs, 1)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        #Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
               print("Early stopping triggered. Training stopped.")
               break
    #'./experiments/results/har/model_0.pt'
    torch.save(model.state_dict(), "./experiments/results/%s/%s_%d.pt" % (data_name, "model", cv) )
    eval_har(model, test_loader,device)
        
        
def eval_har(model,test_loader, device):
    # Evaluate the model on the test set
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            _, predictions = torch.max(outputs, 1)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_targets, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
##TF__________________________________________________________________________
def load_har_tf(batch_size=100, datapath="", data_type="har", percentage=1.0, **kwargs):
    #datapath = "C:/Users/l3s/Desktop/Med-ICU/Project3 Xai/Codes/HAR/"
    with open(os.path.join(datapath, "x_train.pkl"), "rb") as f:
        x_train = pickle.load(f)  
    with open(os.path.join(datapath, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(datapath, "x_test.pkl"), "rb") as f:
        x_test = pickle.load(f)  
    with open(os.path.join(datapath, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    features = kwargs["features"] if "features" in kwargs.keys() else list(range(x_test.shape[1]))
    test_bs = kwargs["test_bs"] if "test_bs" in kwargs.keys() else None
    #--------------
    '''
    x_train = x_train[:50]
    y_train = y_train[:50]
    x_test = x_test[:50]
    y_test = y_test[:50]
    '''
    #-------------
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]
    
    total_sample_n = int(len(x_train) * percentage)
    x_train = x_train[:total_sample_n]
    y_train = y_train[:total_sample_n]
    n_train = int(0.8 * len(x_train))
    x_train = x_train[:, features, :]
    x_test = x_test[:, features, :]
    

    n_train = int(0.8 * len(x_train))
    if "cv" in kwargs.keys():
        print("cv : ", kwargs["cv"])
        kf = KFold(n_splits=5, shuffle=True, random_state=88)
        train_idx, valid_idx = list(kf.split(x_train))[kwargs["cv"]]
    else:
        train_idx = range(n_train)
        valid_idx = range(n_train, len(x_train))
        
    # Convert the indices to integers using .astype(int)
    #train_idx = train_idx.astype(int)
    #valid_idx = valid_idx.astype(int)
    y_train = y_train.type(torch.LongTensor)
    #x_train = x_train.type(torch.FloatTensor)
    #x_test = x_test.type(torch.FloatTensor)
    train_dataset = utils.TensorDataset(torch.Tensor.float(x_train[train_idx, :, :]), torch.Tensor.long(y_train[train_idx]))
    valid_dataset = utils.TensorDataset(torch.Tensor.float(x_train[valid_idx, :, :]), torch.Tensor.long(y_train[valid_idx]))
    
    print()
    test_dataset = utils.TensorDataset(torch.Tensor.float(x_test[:, :, :]), torch.Tensor.long(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=len(x_train) - int(0.8 * n_train))
    if test_bs is not None:
        test_loader = DataLoader(test_dataset, batch_size=test_bs)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(x_test))
    
    return np.concatenate([x_train, x_test]), train_loader, valid_loader, test_loader


#_________________________________________________
ckpt_callback = pl.callbacks.ModelCheckpoint(
    dirpath='./experiments/results/har/',
    filename= "model_0",
    monitor='val_loss',
    save_top_k=1
)

def train_model_TF(model, train_loader, valid_loader, test_loader, optimizer, n_epochs, device, experiment, data, cv=0):
    trainer = pl.Trainer(max_epochs=1, accelerator='gpu')
    #trainer = pl.Trainer(max_epochs=10, callbacks=[ckpt_callback])

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # After training, get the state dictionary of the model
    state_dict = model.state_dict()
    torch.save(state_dict, "./experiments/results/%s/%s_%d.pt" % (data, "model", cv) )
    trainer.test(dataloaders=test_loader)
    
    

#_____________________________________________________

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class HARDataset(Dataset):
    def __init__(self, fname, mask=False):
       self.data = torch.load(fname)
       self.X = self.data['samples']
       self.y = self.data['labels'].to(torch.int64)
       self.mask = mask

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        #print(self.X[item].shape)
        #x = self.X[item][:, :128] # for RNN model
        #x = self.X[item] # for transformer model
        if self.mask:
            mask = torch.ones(self.X[item].shape[1]).to(torch.int32)
            return self.X[item], self.y[item], mask

        return self.X[item][:, :128], self.y[item]


class MIMICDataset(Dataset):
    def __init__(self, X_fname, y_fname):
        with open(X_fname, 'rb') as fs:
            self.X = CPU_Unpickler(fs).load()
            self.X = self.X.permute(0, 2, 1)

        with open(y_fname, 'rb') as fs:
            self.y = CPU_Unpickler(fs).load()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        # print(self.X[item].shape)
        x = self.X[item]
        #mask = torch.ones(x.shape[1]).to(torch.int32)
        return x, self.y[item]


class Sleep_EEDDataset(Dataset):
    def __init__(self, fname):
        self.data = torch.load(fname)
        self.X = self.data['samples']
        self.y = self.data['labels'].to(torch.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        x = self.X[item]
        return x, self.y[item]


class Operator:
    def __init__(self, window_size, features, time_steps, batch_size=1):
        self.window_size = window_size
        self.features = features
        self.time_steps = time_steps
        self.batch_size = batch_size # only applicable to local explanations, so batch_size = 1

    def hadamard(self, X, mask):
        #X = X.reshape(self.batch_size, self.features, self.time_steps)
        #mask = mask.reshape(self.batch_size, self.features, self.time_steps)
        return X * mask

    def compute_moving_average(self, X):
        self.avg_X = torch.clone(X)
        for i in range(self.time_steps):
            self.avg_X[:, :, i] = torch.mean(X[:, :, max(0, i-self.window_size):i+self.window_size], dim=2)

    def historic_average(self, X):
        self.avg_X = torch.clone(X)
        for i in range(self.time_steps):
            self.avg_X[:, :, i] = torch.mean(X[:, :, max(0, i - self.window_size):i], dim=2)

    def mean_operator(self, X, mask):
        X = X.reshape(self.batch_size, self.features, self.time_steps)
        mask = mask.reshape(self.batch_size, self.features, self.time_steps)
        return X * mask + self.avg_X * (1 - mask)


if __name__ == '__main__':
    '''
    data = HARDataset('Datasets/HAR/test.pt')
    #print(data.X.shape)
    operator = Operator(window_size=10, features=3, time_steps=206)
    test_loader = DataLoader(data, batch_size=1)
    for X, y in test_loader:
        print(X)
        operator.compute_moving_average(X)
        print(operator.avg_X)
        break
    
    data = MIMICDataset('Datasets/MIMIC/inputs_0.pkl', 'Datasets/MIMIC/true_labels_0.pkl')
    loader = DataLoader(data, batch_size=8)
    for X, y in loader:
        print(X.shape)
        #print(y.shape)
        print(y)
        break
    '''
    with open('Datasets/MIMIC/inputs_0.pkl', 'rb') as fs:
        X = CPU_Unpickler(fs).load()
        print(X.shape)

    X_s = X[:10, :, :]
    print(X_s.shape)
    with open('Datasets/MIMIC/inputs.pkl', 'wb') as fs:
        pickle.dump(X_s, fs)

    with open('Datasets/MIMIC/true_labels_0.pkl', 'rb') as fs:
        y = CPU_Unpickler(fs).load()
        print(y.shape)

    y_s =y[:10]
    print(y_s.shape)
    with open('Datasets/MIMIC/true_labels.pkl', 'wb') as fs:
        pickle.dump(y_s, fs)
    #print(data.X.shape)
    #print(data.y.shape)
    #data = Sleep_EEDDataset('Datasets/sleepEDF/test.pt')
    #print(data.X.shape)
