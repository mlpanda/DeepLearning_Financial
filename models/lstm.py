import pandas as pd 
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Sequence(nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, nb_layers=5, dropout=0.5):
        super(Sequence, self).__init__()
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm = nn.LSTM(self.nb_features, self.hidden_size, self.nb_layers, dropout=dropout)
        self.lin = nn.Linear(self.hidden_size,1)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        #print(type(h0))
        c0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        #print(type(c0))
        output, hn = self.lstm(input, (h0, c0))
        #output = F.relu(self.lin(output))
        out = self.lin(output[-1])
        return out