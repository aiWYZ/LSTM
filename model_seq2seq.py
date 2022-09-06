"""
lstm(Encoder) -> fc -> lstm(Decoder) -> fc
"""

import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

INPUT_SIZE = 2
OUTPUT_SIZE = 2
INPUT_SEQ = 8
OUTPUT_SEQ = 8
HIDDEN_SIZE = 32
EPOCHS = 50
VAL_ITR = 10
BATCH_SIZE = 64
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, input):
        output, (h, c) = self.lstm(input)
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, h0, c0):
        lstm_out, (h, c) = self.lstm(input, (h0, c0))
        pred = self.fc(lstm_out)
        pred = pred[:, -1, :]
        return pred


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, input_seq, output_seq):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size, output_size)
    
    def forward(self, input):
        pred = torch.zeros(input.shape[0], self.input_seq, self.output_size).to(DEVICE)
        h, c = self.encoder(input)
        for frame in range(self.input_seq):
            _input = input[:, frame, :]
            _input = _input.view(-1, 1, self.input_size)
            output = self.decoder(_input, h, c)
            pred[:, frame, :] = output
        return pred



def train(train_pth, model_pth, input_size, output_size, input_seq, output_seq, hidden_zize, device):
    train_data = torch.load(train_pth, map_location=device)

    net = Net(input_size, output_size, hidden_zize, input_seq, output_seq)
    net = net.to(device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    overall_loss_log = []

    for epoch in range(EPOCHS):
        loss_log = []
        for input, truth in train_data:
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, truth)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
        epoch_loss = np.average(loss_log)
        overall_loss_log.append(epoch_loss)

        if (epoch%10==9):
            print('Epoch {} AveLoss = {}'.format(epoch, epoch_loss))

    torch.save(net.state_dict(), model_pth)

    print("-----------------------")

    # plt.plot(range(EPOCHS), overall_loss_log, label='Loss per epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('Loss_per_epoch')
    # plt.show()



if __name__ == '__main__':
    data_form = 'Linear'
    for dir in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
        if (os.path.exists('./models/{}/{}'.format(data_form, dir)) == False):
            os.mkdir('./models/{}/{}'.format(data_form, dir))
        TRAIN_PTH = './dataset/{}/{}/train_8-8'.format(data_form, dir)
        MODEL_PTH = './models/{}/{}/model_8-8.pth'.format(data_form, dir)
        train(TRAIN_PTH, MODEL_PTH, INPUT_SIZE, OUTPUT_SIZE, INPUT_SEQ, OUTPUT_SEQ, HIDDEN_SIZE, DEVICE)