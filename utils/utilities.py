## EXTERNAL
import pandas as pd 
import numpy as np 
import pickle
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import sklearn
import time
import os
import random 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def prepare_data_lstm(x_encoded, y_close, time_steps, log_return=True, train=True):
    ct = 0
    data = []
    for i in range(len(x_encoded)-time_steps):
        ct +=1
        if train:
            x_train = x_encoded[i:i+time_steps]
        else:
            x_train = x_encoded[:i+time_steps]

        data.append(x_train)

    if log_return==False:
        y_close = y_close.pct_change()[1:]
    else:
        y_close = (np.log(y_close) - np.log(y_close.shift(1)))[1:] # the log return, i.e. ln(y_t/y_(t-1))

    if train:           
        y = y_close[time_steps-1:]
    else:
        y=y_close

    return data, y


class ExampleDataset(Dataset):

    def __init__(self, x, y, batchsize):
        self.datalist = x
        self.target = y
        self.batchsize = batchsize
        self.length = 0
        self.length = len(x)

    def __len__(self):
        return int(self.length/self.batchsize+1)

    def __getitem__(self, idx):
        x = self.datalist[idx*self.batchsize:(idx+1)*self.batchsize]
        y = self.target[idx*self.batchsize:(idx+1)*self.batchsize]
        sample = {'x': x, 'y': y}

        return sample


def evaluate_lstm(dataloader, model, criterion):

    pred_val = []
    target_val = []
    model.eval()
    # do evaluation
    loss_val = 0
    sample_cum_x = [None]

    for j in range(len(dataloader)):

        sample = dataloader[j]
        sample_x = sample["x"]

        if len(sample_x) != 0:

            sample_x = np.stack(sample_x)
            input = Variable(torch.FloatTensor(sample_x), requires_grad=False)
            input = torch.transpose(input, 0, 1)
            target = Variable(torch.FloatTensor(sample["y"].as_matrix()), requires_grad=False)

            out = model(input)

            loss = criterion(out, target)

            loss_val += float(loss.data.numpy())
            pred_val.extend(out.data.numpy().flatten().tolist())
            target_val.extend(target.data.numpy().flatten().tolist())

    return loss_val, pred_val, target_val


def backtest(predictions, y):

    trans_cost = 0.0001
    real = [1]
    index = [1]
    for r in range(len(predictions)):
        rets= y.as_matrix().flatten().tolist()
        ret = rets[r]
        real.append(real[-1]*(1+ret))

        if predictions[r]>0.0:
            # buy
            ret = rets[r] - 2*trans_cost
            index.append(index[-1]*(1+ret))

        elif predictions[r]<0.0:
            # sell
            ret = -rets[r] - 2*trans_cost
            index.append(index[-1]*(1+ret))
        else:
            #print("no trade")
            # don't trade
            index.append(index[-1])

    return index, real


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', name="checkpoint"):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(name) + 'model_best.pth.tar')

