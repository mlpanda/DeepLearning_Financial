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

##INTERNAL
from models import Autoencoder
from models import Sequence
from models import waveletSmooth

from utils import prepare_data_lstm, ExampleDataset, save_checkpoint, evaluate_lstm, backtest

# ---------------------------------------------------------------------------
# --------------------------- STEP 0: LOAD DATA -----------------------------
# ---------------------------------------------------------------------------

path = "./data/S&P500IndexData-Table1.csv"
data_master = pd.read_csv(path, sep=";")

# 600 is a bit more than 2 years of data
num_datapoints = 600
# roll by approx. 60 days - 3 months of trading days
step_size = int(0.1 * num_datapoints)
# calculate number of iterations we can do over the entire data set
num_iterations = int(np.ceil((len(data_master)-num_datapoints)/step_size))+2

y_test_lst = []
preds = []
ct = 0

for n in range(num_iterations):
    print(n)
    data = data_master.iloc[n*step_size:num_datapoints+n*step_size,:]
    data.columns = [col.strip() for col in data.columns.tolist()]
    print(data.shape)
    ct +=1

    feats = data.iloc[:,2:]

    # This is a scaling of the inputs such that they are in an appropriate range    
    feats["Close Price"].loc[:] = feats["Close Price"].loc[:]/1000
    feats["Open Price"].loc[:] = feats["Open Price"].loc[:]/1000
    feats["High Price"].loc[:] = feats["High Price"].loc[:]/1000
    feats["Low Price"].loc[:] = feats["Low Price"].loc[:]/1000
    feats["Volume"].loc[:] = feats["Volume"].loc[:]/1000000
    feats["MACD"].loc[:] = feats["MACD"].loc[:]/10
    feats["CCI"].loc[:] = feats["CCI"].loc[:]/100
    feats["ATR"].loc[:] = feats["ATR"].loc[:]/100
    feats["BOLL"].loc[:] = feats["BOLL"].loc[:]/1000
    feats["EMA20"].loc[:] = feats["EMA20"].loc[:]/1000
    feats["MA10"].loc[:] = feats["MA10"].loc[:]/1000
    feats["MTM6"].loc[:] = feats["MTM6"].loc[:]/100
    feats["MA5"].loc[:] = feats["MA5"].loc[:]/1000
    feats["MTM12"].loc[:] = feats["MTM12"].loc[:]/100
    feats["ROC"].loc[:] = feats["ROC"].loc[:]/10
    feats["SMI"].loc[:] = feats["SMI"].loc[:] * 10
    feats["WVAD"].loc[:] = feats["WVAD"].loc[:]/100000000
    feats["US Dollar Index"].loc[:] = feats["US Dollar Index"].loc[:]/100
    feats["Federal Fund Rate"].loc[:] = feats["Federal Fund Rate"].loc[:]
    
    data_close = feats["Close Price"].copy()
    data_close_new = data_close

    # Split in train, test and validation set

    test = feats[-step_size:]
    validate = feats[-2*step_size:-step_size]
    train = feats[:-2*step_size]

    y_test = data_close_new[-step_size:].as_matrix()
    y_validate = data_close_new[-2*step_size:-step_size].as_matrix()
    y_train = data_close_new[:-2*step_size].as_matrix()
    feats_train = train.as_matrix().astype(np.float)
    feats_validate = validate.as_matrix().astype(np.float)
    feats_test = test.as_matrix().astype(np.float)

    # ---------------------------------------------------------------------------
    # ----------------------- STEP 2.0: NORMALIZE DATA --------------------------
    # ---------------------------------------------------------------------------

    # REMOVED THE NORMALIZATION AND MANUALLY SCALED TO APPROPRIATE VALUES ABOVE

    """
    scaler = StandardScaler().fit(feats_train)

    feats_norm_train = scaler.transform(feats_train)
    feats_norm_validate = scaler.transform(feats_validate)
    feats_norm_test = scaler.transform(feats_test)
    """
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(feats_train)

    feats_norm_train = scaler.transform(feats_train)
    feats_norm_validate = scaler.transform(feats_validate)
    feats_norm_test = scaler.transform(feats_test)
    """    
    data_close = pd.Series(np.concatenate((y_train, y_validate, y_test)))
    
    feats_norm_train = feats_train.copy()
    feats_norm_validate = feats_validate.copy()
    feats_norm_test = feats_test.copy()
    
    # ---------------------------------------------------------------------------
    # ----------------------- STEP 2.1: DENOISE USING DWT -----------------------
    # ---------------------------------------------------------------------------

    for i in range(feats_norm_train.shape[1]):
        feats_norm_train[:,i] = waveletSmooth(feats_norm_train[:,i], level=1)[-len(feats_norm_train):]

    # for the validation we have to do the transform using training data + the current and past validation data
    # i.e. we CAN'T USE all the validation data because we would then look into the future 
    temp = np.copy(feats_norm_train)
    feats_norm_validate_WT = np.copy(feats_norm_validate)
    for j in range(feats_norm_validate.shape[0]):
        #first concatenate train with the latest validation sample
        temp = np.append(temp, np.expand_dims(feats_norm_validate[j,:], axis=0), axis=0)
        for i in range(feats_norm_validate.shape[1]):
            feats_norm_validate_WT[j,i] = waveletSmooth(temp[:,i], level=1)[-1]

    # for the test we have to do the transform using training data + validation data + current and past test data
    # i.e. we CAN'T USE all the test data because we would then look into the future 
    temp_train = np.copy(feats_norm_train)
    temp_val = np.copy(feats_norm_validate)
    temp = np.concatenate((temp_train, temp_val))
    feats_norm_test_WT = np.copy(feats_norm_test)
    for j in range(feats_norm_test.shape[0]):
        #first concatenate train with the latest validation sample
        temp = np.append(temp, np.expand_dims(feats_norm_test[j,:], axis=0), axis=0)
        for i in range(feats_norm_test.shape[1]):
            feats_norm_test_WT[j,i] = waveletSmooth(temp[:,i], level=1)[-1]
    
    # ---------------------------------------------------------------------------
    # ------------- STEP 3: ENCODE FEATURES USING STACKED AUTOENCODER -----------
    # ---------------------------------------------------------------------------

    num_hidden_1 = 10
    num_hidden_2 = 10
    num_hidden_3 = 10
    num_hidden_4 = 10

    n_epoch=100#20000

    # ---- train using training data
    
    # The n==0 statement is done because we only want to initialize the network once and then keep training
    # as we move through time 

    if n == 0:
        auto1 = Autoencoder(feats_norm_train.shape[1], num_hidden_1)
    auto1.fit(feats_norm_train, n_epoch=n_epoch)

    inputs = torch.autograd.Variable(torch.from_numpy(feats_norm_train.astype(np.float32)))

    if n == 0:
        auto2 = Autoencoder(num_hidden_1, num_hidden_2)
    auto1_out = auto1.encoder(inputs).data.numpy()
    auto2.fit(auto1_out, n_epoch=n_epoch)

    if n == 0:
        auto3 = Autoencoder(num_hidden_2, num_hidden_3)
    auto1_out = torch.autograd.Variable(torch.from_numpy(auto1_out.astype(np.float32)))
    auto2_out = auto2.encoder(auto1_out).data.numpy()
    auto3.fit(auto2_out, n_epoch=n_epoch)

    if n == 0:
        auto4 = Autoencoder(num_hidden_3, num_hidden_4)
    auto2_out = torch.autograd.Variable(torch.from_numpy(auto2_out.astype(np.float32)))
    auto3_out = auto3.encoder(auto2_out).data.numpy()
    auto4.fit(auto3_out, n_epoch=n_epoch)
    

    # Change to evaluation mode, in this mode the network behaves differently, e.g. dropout is switched off and so on
    auto1.eval()        
    auto2.eval()
    auto3.eval()
    auto4.eval()
    
    X_train = feats_norm_train
    X_train = torch.autograd.Variable(torch.from_numpy(X_train.astype(np.float32)))
    train_encoded = auto4.encoder(auto3.encoder(auto2.encoder(auto1.encoder(X_train))))
    train_encoded = train_encoded.data.numpy()

    # ---- encode validation and test data using autoencoder trained only on training data 
    X_validate = feats_norm_validate_WT   
    X_validate = torch.autograd.Variable(torch.from_numpy(X_validate.astype(np.float32)))
    validate_encoded = auto4.encoder(auto3.encoder(auto2.encoder(auto1.encoder(X_validate))))
    validate_encoded = validate_encoded.data.numpy()

    X_test = feats_norm_test_WT
    X_test = torch.autograd.Variable(torch.from_numpy(X_test.astype(np.float32)))
    test_encoded = auto4.encoder(auto3.encoder(auto2.encoder(auto1.encoder(X_test))))
    test_encoded = test_encoded.data.numpy()
    
    # switch back to training mode
    auto1.train()        
    auto2.train()
    auto3.train()
    auto4.train()

    
    # ---------------------------------------------------------------------------
    # -------------------- STEP 4: PREPARE TIME-SERIES --------------------------
    # ---------------------------------------------------------------------------

    # split the entire training time-series into pieces, depending on the number
    # of time steps for the LSTM

    time_steps = 4

    args = (train_encoded, validate_encoded, test_encoded)

    x_concat = np.concatenate(args)

    validate_encoded_extra = np.concatenate((train_encoded[-time_steps:], validate_encoded))
    test_encoded_extra = np.concatenate((validate_encoded[-time_steps:], test_encoded))

    y_train_input = data_close[:-len(validate_encoded)-len(test_encoded)]
    y_val_input = data_close[-len(test_encoded)-len(validate_encoded)-1:-len(test_encoded)]
    y_test_input = data_close[-len(test_encoded)-1:]

    x, y = prepare_data_lstm(train_encoded, y_train_input, time_steps, log_return=True, train=True)
    x_v, y_v = prepare_data_lstm(validate_encoded_extra, y_val_input, time_steps, log_return=False, train=False)
    x_te, y_te = prepare_data_lstm(test_encoded_extra, y_test_input, time_steps, log_return=False, train=False)


    x_test = x_te
    x_validate = x_v
    x_train = x 

    y_test = y_te 
    y_validate = y_v 
    y_train = y

    y_train = y_train.as_matrix()

    # ---------------------------------------------------------------------------
    # ------------- STEP 5: TIME-SERIES REGRESSION USING LSTM -------------------
    # ---------------------------------------------------------------------------

    batchsize = 60

    trainloader = ExampleDataset(x_train, y_train, batchsize)
    valloader = ExampleDataset(x_validate, y_validate, 1)
    testloader = ExampleDataset(x_test, y_test, 1)

    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # build the model
    if n == 0:
        seq = Sequence(num_hidden_4, hidden_size=100, nb_layers=3)

    resume = ""

    # if a path is given in resume, we resume from a checkpoint
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        seq.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in seq.parameters()])))

    # we use the mean squared error loss
    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=seq.parameters(), lr=0.0005)

    start_epoch = 0 
    epochs = 1#5000

    global_loss_val = np.inf
    #begin to train
    global_profit_val = -np.inf

    for i in range(start_epoch, epochs):
        seq.train()
        loss_train = 0

        # shuffle ONLY training set        
        combined = list(zip(x_train, y_train))
        random.shuffle(combined)
        x_train=[]
        y_train=[]
        x_train[:], y_train[:] = zip(*combined)
        
        # initialize trainloader with newly shuffled training data        
        trainloader = ExampleDataset(x_train, y_train, batchsize)

        pred_train = []
        target_train = []
        for j in range(len(trainloader)):
            sample = trainloader[j]
            sample_x = sample["x"]

            if len(sample_x) != 0:

                sample_x = np.stack(sample_x)
                input = Variable(torch.FloatTensor(sample_x), requires_grad=False)
                input = torch.transpose(input, 0, 1)
                target = Variable(torch.FloatTensor([x for x in sample["y"]]), requires_grad=False)

                optimizer.zero_grad()
                out = seq(input)
                loss = criterion(out, target)

                loss_train += float(loss.data.numpy())
                pred_train.extend(out.data.numpy().flatten().tolist())
                target_train.extend(target.data.numpy().flatten().tolist())

                loss.backward()

                optimizer.step()


        if i % 100 == 0:

            plt.plot(pred_train)
            plt.plot(target_train)
            plt.show()
            
            loss_val, pred_val, target_val = evaluate_lstm(dataloader=valloader, model=seq, criterion=criterion)
            
            plt.scatter(range(len(pred_val)), pred_val)
            plt.scatter(range(len(pred_val)), target_val)
            plt.show()

            index, real = backtest(pred_val, y_validate)

            print(index[-1])
            # save according to profitability
            if index[-1]>global_profit_val and i>200:
                print("CURRENT BEST")
                global_profit_val = index[-1]
                save_checkpoint({'epoch': i + 1, 'state_dict': seq.state_dict()}, is_best=True, filename='checkpoint_lstm.pth.tar')

            save_checkpoint({'epoch': i + 1, 'state_dict': seq.state_dict()}, is_best=False, filename='checkpoint_lstm.pth.tar')

            print("LOSS TRAIN: " + str(float(loss_train)))        
            print("LOSS VAL: " + str(float(loss_val)))
            print(i)

    # do the final test
    # first load the best checkpoint on the val set

    resume = "./runs/checkpoint/model_best.pth.tar"
    #resume = "./runs/HF/checkpoint_lstm.pth.tar"

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        seq.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    seq.eval()

    loss_test, preds_test, target_test = evaluate_lstm(dataloader=testloader, model=seq, criterion=criterion)

    print("LOSS TEST: " + str(float(loss_test)))

    temp2 = y_test.as_matrix().flatten().tolist()
    y_test_lst.extend(temp2)
        
    plt.plot(preds_test)
    plt.plot(y_test_lst)
    plt.scatter(range(len(preds_test)), preds_test)
    plt.scatter(range(len(y_test_lst)), y_test_lst)
    plt.savefig("test_preds.pdf")

    # ---------------------------------------------------------------------------
    # ------------------ STEP 6: BACKTEST (ARTICLE WAY) -------------------------
    # ---------------------------------------------------------------------------

    index, real = backtest(preds_test, pd.DataFrame(y_test_lst))

    plt.close()
    plt.plot(index, label="strat")
    plt.plot(real, label="bm")
    plt.legend()
    plt.savefig("performance_article_way.pdf")
    plt.close()

