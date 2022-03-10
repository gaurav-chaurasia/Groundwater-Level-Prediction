import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

import io
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms 

# load model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, class_size, dropout=0.5, rnn_type='lstm'):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_size = class_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_size, self.class_size) # FC layer in our paper

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        r_out, _ = self.rnn(x, (h0, c0))

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(self.dropout((r_out[:, time_step, :]))))
        return torch.stack(outs, dim=1)


model = LSTM(input_size=3, hidden_size=5, num_layers=1, class_size=1, dropout=0.5, rnn_type='lstm')

PATH = "streamflow_model.pth"
model.load_state_dict(torch.load(PATH))
model.eval()


def get_prediction():
    pass