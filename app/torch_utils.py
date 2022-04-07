from calendar import month
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


streamflow = LSTM(input_size=3, hidden_size=5, num_layers=1, class_size=1, dropout=0.5, rnn_type='lstm')
groundwater_level = LSTM(input_size=5, hidden_size=40, num_layers=1, class_size=1, dropout=0.5, rnn_type='lstm')

PATH = "app/streamflow_model.pth"
streamflow.load_state_dict(torch.load(PATH))
streamflow.eval()

PATH1 = "app/groundwater_level.pth"
groundwater_level.load_state_dict(torch.load(PATH1))
groundwater_level.eval()


def get_groundwater_prediction(m, i, r, t, e):

    # download dataset
    # main utils
    ss_X = StandardScaler()
    ss_y = StandardScaler()

    url='https://raw.githubusercontent.com/gaurav-chaurasia/Groundwater-Level-Prediction/master/data/demo.csv'
    data = pd.read_csv(url)
    # data.head()


    Inputs = data.drop('Year', axis=1).drop('Depth', axis=1)
    Outputs = data['Depth']
    Inputs = Inputs.values

    Outputs = Outputs.values.reshape(-1, 1)

    # 

    input_data = [[int(m), float(i), float(r), float(t), float(e)]]
    input_data = pd.DataFrame(input_data, columns = ['Month', 'Irrigation', 'Rainfall', 'Tem', 'Evaporation'])
    # input_data = input_data.set_index('Date')

    # print(input_data)
    # print(data)
    input_data = input_data.values
    # print(input_data)

    Inputs = np.concatenate([Inputs, input_data], axis=0)

    # defining some variables
    training_percentage = 0.8
    testing_percentage = 1 - training_percentage

    # this shape of numpy array give the dimension of the matrix and zero'th index element gives number of element in matrix
    data_size = Inputs.shape[0] | Outputs.shape[0]

    count_for_training = round(training_percentage * data_size)
    count_for_testing = data_size - count_for_training

    X_train = Inputs[0:count_for_training]
    y_train = Outputs[0:count_for_training]


    X_test = Inputs[count_for_training:]
    y_test = Outputs[count_for_training:]

    X = np.concatenate([X_train, X_test], axis=0)


    # Standardization
    X = ss_X.fit_transform(X)

    # Training set of data
    X_train_standardized = X[0:count_for_training]
    X_train_standardized = np.expand_dims(X_train_standardized, axis=0)
    # print(X_train_standardized, "\n")


    y_train_standardized = ss_y.fit_transform(y_train)
    y_train_standardized = np.expand_dims(y_train_standardized, axis=0)
    # print(y_train_standardized, "\n")


    X_test_standardized  = X
    X_test_standardized = np.expand_dims(X_test_standardized, axis=0)

    # Transfer to Pytorch Variable
    X_train_standardized = Variable(torch.from_numpy(X_train_standardized).float())
    y_train_standardized = Variable(torch.from_numpy(y_train_standardized).float())
    X_test_standardized = Variable(torch.from_numpy(X_test_standardized).float())

    groundwater_level.eval()

    # print("sfsf",X_test_standardized.shape)
    # this y_prediction is prediction made by model on X_test_standardized data
    y_pred_total = groundwater_level(X_test_standardized).detach().numpy()
    y_pred_total = ss_y.inverse_transform(y_pred_total[0, :])

    return y_pred_total[-1][0]

def get_streamflow_prediction(p, tmax, tmin):

    # download dataset
    # main utils
    ss_X = StandardScaler()
    ss_y = StandardScaler()

    url='https://raw.githubusercontent.com/sachin-saroha/Data/main/APHRODITE_deep_learning%20(1).csv'
    data = pd.read_csv(url, index_col='Date')
    # data.head()

    input_data = [['2014-12-30', float(p), float(tmax), float(tmin)]]
    input_data = pd.DataFrame(input_data, columns = ['Date', 'p', 'tmax', 'tmin'])
    input_data = input_data.set_index('Date')

    # print(input_data)
    # print(data)
    input_data = input_data.values
    # print(input_data)


    Inputs = data.drop('Q', axis=1)
    Outputs = data['Q']
    Inputs = Inputs.values

    Outputs = Outputs.values.reshape(-1, 1)
    # print(Inputs.shape)

    Inputs = np.concatenate([Inputs, input_data], axis=0)

    # defining some variables
    training_percentage = 0.7
    testing_percentage = 1 - training_percentage

    # this shape of numpy array give the dimension of the matrix and zero'th index element gives number of element in matrix
    data_size = Inputs.shape[0] | Outputs.shape[0]

    count_for_training = round(training_percentage * data_size)
    count_for_testing = data_size - count_for_training

    X_train = Inputs[0:count_for_training]
    y_train = Outputs[0:count_for_training]


    X_test = Inputs[count_for_training:]
    y_test = Outputs[count_for_training:]

    X = np.concatenate([X_train, X_test], axis=0)


    # Standardization
    X = ss_X.fit_transform(X)

    # Training set of data
    X_train_standardized = X[0:count_for_training]
    X_train_standardized = np.expand_dims(X_train_standardized, axis=0)
    # print(X_train_standardized, "\n")


    y_train_standardized = ss_y.fit_transform(y_train)
    y_train_standardized = np.expand_dims(y_train_standardized, axis=0)
    # print(y_train_standardized, "\n")


    X_test_standardized  = X
    X_test_standardized = np.expand_dims(X_test_standardized, axis=0)

    # Transfer to Pytorch Variable
    X_train_standardized = Variable(torch.from_numpy(X_train_standardized).float())
    y_train_standardized = Variable(torch.from_numpy(y_train_standardized).float())
    X_test_standardized = Variable(torch.from_numpy(X_test_standardized).float())

    streamflow.eval()

    # print("sfsf",X_test_standardized.shape)
    # this y_prediction is prediction made by model on X_test_standardized data
    y_pred_total = streamflow(X_test_standardized).detach().numpy()
    y_pred_total = ss_y.inverse_transform(y_pred_total[0, :])

    return y_pred_total[-1][0]