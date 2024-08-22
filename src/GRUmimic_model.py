import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class StateClassifier(nn.Module):
    def __init__(
        self,
        feature_size,
        n_state,
        hidden_size,
        rnn="GRU",
        regres=True,
        bidirectional=False,
        return_all=False,
        seed=random.seed("2019"),
        data= "mimic", #Change here into another data
    ):
        super(StateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        self.data = data
        #print(data)
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.n_state),
        )
        # nn.Softmax(-1))

    def forward(self, input, past_state=None, **kwargs):
        input = input.permute(2, 0, 1).to(self.device)
        self.rnn.to(self.device)
        self.regressor.to(self.device)
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, input.shape[1], self.hidden_size]).to(self.device)
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(input, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))
        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                reshaped_encodings = all_encodings.view(all_encodings.shape[1] * all_encodings.shape[0], -1)
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0], -1))
        else:
            return encoding.view(encoding.shape[1], -1)
