#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader,Dataset

class LSTM(nn.Module):
    def __init__(self, embedding_dim=32, h_dim=32
                 , mlp_dim=1024, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.embedding = nn.Linear(2, self.embedding_dim)
        self.decoder = nn.Linear(self.embedding_dim, 16)
        
    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim),
            torch.zeros(self.num_layers, batch, self.h_dim)
        )


    def forward(self, X):
        batch = X.shape[1]
        X_embed=self.embedding(X.contiguous().view(-1, 2))
        X_embed=X_embed.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.lstm(X_embed, state_tuple)
        final_h = state[0]
        final_h=final_h.view(-1, batch, self.embedding_dim)
        final_hh=self.decoder(final_h).view(8,batch,2)
        return final_hh


class Encoder(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            2, h_dim, num_layers, dropout=dropout
        )

        self.linear = nn.Linear(h_dim, 32)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim),
            torch.zeros(self.num_layers, batch, self.h_dim)
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj, state_tuple)
        final_h = state[0]
        final_h = final_h.view(batch, -1)
        final_hh = self.linear(final_h)
        final_hh = final_hh.view(batch, -1, 2)
        final_hh = final_hh.permute(1,0,2)
        return final_hh


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len = 8, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.decoder = nn.LSTM(
            2, h_dim, num_layers, dropout=dropout
        )
        self.linear = nn.Linear(h_dim, 2*self.seq_len)
        self.num_layers = num_layers

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim),
            torch.zeros(self.num_layers, batch, self.h_dim)
        )

    def forward(self, obs_traj):
        batch = obs_traj.size(1)
        state_tuple = self.init_hidden(batch)
        output, state = self.decoder(obs_traj, state_tuple)
        final_h = state[0]
        final_h = final_h.view(batch, -1)
        final_hh = self.linear(final_h)
        final_hh = final_hh.view(batch, -1, 2)
        final_hh = final_hh.permute(1, 0, 2)
        return final_hh


class NET(nn.Module):
    def __init__(
            self, embedding_dim=64, encoder_h_dim=64, decoder_h_dim=128, mlp_dim=1024, num_layers=1,
            dropout=0.0, seq_len=8
    ):
        super(NET, self).__init__()
        self.seq_len = seq_len
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.num_layers = num_layers
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = Decoder(
            embedding_dim = embedding_dim,
            h_dim = decoder_h_dim,
            mlp_dim = mlp_dim,
            num_layers = num_layers,
            dropout = dropout,
            seq_len = seq_len
        )

    def forward(self, obs_traj):
        obs_enco = self.encoder(obs_traj)
        obs_deco = self.decoder(obs_enco)
        return obs_deco



class MyDataset(Dataset):
    def __init__(self,dataset):
        self.x, self.y=dataset
        self.len = self.x.size(1)

    def __getitem__(self, index):
        return self.x[:, index, :], self.y[:, index, :]

    def __len__(self):
        return self.len





def create_inout_sequences(input_data, tw):
    inout_seq = []
    len = input_data.shape[0]
    for i in range(len-tw-7):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+8]
        inout_seq.append((train_seq, train_label))
    return inout_seq
'''把每个人连续的数据转变为每十六个一组，前八个用于输入，后八个作为输出'''

'''
def test_accuracy(test_batch_size, testing_data, test_net):
    test_num = testing_data.
    test_loss = 0
    for batch in range(0, test_batch_size):
        flag = False
        idd = float(idlist[batch])
        idddata = (testing_data[testing_data.ID == idd].iloc[:, 2:4])
        my_array = np.array(idddata)
        test_torch = torch.tensor(my_array, dtype=torch.float32)
        for test_data, test_label in create_inout_sequences(test_torch, 8):
            test_data = torch.unsqueeze(test_data, 1)
            test_label = torch.unsqueeze(test_label, 1)
            if flag == False:
                X_test = test_data
                y_test = test_label
                flag = True
            else:
                X_test = torch.cat((X_test, test_data), dim=1)
                y_test = torch.cat((y_test, test_label), dim=1)
        y_pred = test_net(X_test)
        test_loss += loss(y_pred.squeeze(), y_test.squeeze())
    return test_loss.item()/test_batch_size
'''

def read_data(train_data_set):
    flag = False
    num_people=int(train_data_set.iloc[:,1].max())
    for idd in range(1, num_people+1):
        idddata = (train_data_set[train_data_set.ID == idd].iloc[:, 2:4])
        my_array = np.array(idddata)
        my_torch = torch.tensor(my_array, dtype=torch.float32)
        for training_data, training_label in create_inout_sequences(my_torch, 8):
            if flag == False:
                x_train = torch.unsqueeze(training_data, 1)
                y_train = torch.unsqueeze(training_label, 1)
                flag = True
            else:
                x_train = torch.cat((x_train, torch.unsqueeze(training_data, 1)), dim=1)
                y_train = torch.cat((y_train, torch.unsqueeze(training_label, 1)), dim=1)

    return (x_train, y_train)
'''datatype from pd to torch'''



net = NET()
loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01,momentum=0.05)
batch_size = 64
num_epoc = 300
lr = 0.01
momen = 0.05
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen)


data2 = pd.read_csv('biwi_hotel_train.txt',    sep="\t",  names=['zhen', 'ID', 'x', 'y'])
data1 = pd.read_csv('crowds_zara02_train.txt', sep="\t",  names=['zhen', 'ID', 'x', 'y'])
data3 = pd.read_csv('crowds_zara03_train.txt', sep="\t",  names=['zhen', 'ID', 'x', 'y'])
data4 = pd.read_csv('students001_train.txt',   sep="\t",  names=['zhen', 'ID', 'x', 'y'])
data5 = pd.read_csv('students003_train.txt',   sep="\t",  names=['zhen', 'ID', 'x', 'y'])
data6 = pd.read_csv('uni_examples_train.txt',  sep="\t",  names=['zhen', 'ID', 'x', 'y'])
data7 = pd.read_csv('biwi_eth_train.txt',  sep="\t",  names=['zhen', 'ID', 'x', 'y'])
test_rawdata = pd.read_csv('crowds_zara01.txt',   sep="\t",  names=['zhen', 'ID', 'x', 'y'])
train_data1, train_label1 = read_data(data1)
train_data2, train_label2 = read_data(data2)
train_data3, train_label3 = read_data(data3)
train_data4, train_label4 = read_data(data4)
train_data5, train_label5 = read_data(data5)
train_data6, train_label6 = read_data(data6)
train_data7, train_label7 = read_data(data7)
test_data, test_label = read_data(test_rawdata)
train_data = torch.cat((train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7), dim=1)
train_label= torch.cat((train_label1, train_label2, train_label3, train_label4, train_label5, train_label6, train_label7),dim=1)
train_datas = (train_data, train_label)
test_datas = (test_data, test_label)
train_dataset = MyDataset(train_datas)
test_dataset = MyDataset(test_datas)
train_length = train_dataset.len
test_length = test_dataset.len
print(train_length)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
for epoc in range(50):
    trainADE_loss=0
    testADE_loss=0
    trainFDE_loss=0
    testFDE_loss=0
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train = x_train.permute(1,0,2)
        y_train = y_train.permute(1,0,2)
        length = x_train.shape[1]
        optimizer.zero_grad()
        y_pred = net(x_train)
        single_loss = loss(y_pred, y_train)
        single_loss.backward()
        optimizer.step()
        FDE_loss = loss(y_pred[-1, :, :], y_train[-1, :, :])
        trainADE_loss += single_loss.item() * length
        trainFDE_loss += FDE_loss.item() * length



    for i, (x_test,y_test) in enumerate(test_loader):
        x_test = x_test.permute(1,0,2)
        y_test = y_test.permute(1,0,2)
        length = x_test.shape[1]
        y_test_pred = net(x_test)
        testADE_loss += loss(y_test_pred, y_test).item() *length
        testFDE_loss += loss(y_test_pred[-1, :, :], y_test[-1, :, :]).item()*length


    print("ADEtrain loss:", trainADE_loss/train_length)
    print("ADEtest loss:", testADE_loss/test_length)
    print("FDEtrain_loss:", trainFDE_loss/train_length)
    print("FDEtest_loss:", testFDE_loss/test_length)






'''
for epoc in range(num_epoc):

    if epoc % 20 == 10:
        print("train error:", loss1, " ", loss2, " ", loss3, " ", loss4, " ", loss5, " ", loss6)
        print('test error', test_accuracy(20, test_data, net))
        lr *= 0.8
        momen *= 0.8


'''