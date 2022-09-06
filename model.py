"""
lstm -> fc
"""

import os
from turtle import forward
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

INPUT_SIZE = 2
OUTPUT_SIZE = 2
INPUT_SEQ = 8
OUTPUT_SEQ = 8
HIDDEN_SIZE = 16
EPOCHS = 50
VAL_ITR = 10
BATCH_SIZE = 64
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EuclidLossFunc(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, y_hat):
        loss = torch.mean(torch.sqrt(torch.square(y[:,:,0]-y_hat[:,:,0])+torch.square(y[:,:,1]-y_hat[:,:,1])))
        return loss

class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, output_seq) -> None:
        super(Net, self).__init__()
        self.output_size = output_size
        self.output_seq = output_seq
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size*output_seq)

    def forward(self, input):
        lstm_out, (h, c) = self.lstm(input)
        pred = self.fc(lstm_out)
        # pred(batch_size, input_seq, output_size*output_seq)
        pred = pred[:, -1, :]
        # pred(batch_size, output_size*output_seq)
        pred = pred.view(pred.shape[0], self.output_seq, self.output_size)
        # pred(batch_size, output_seq, output_size)
        return pred



def train(train_pth, model_pth, input_size, output_size, input_seq, output_seq, hidden_zize, device):
    train_data = torch.load(train_pth, map_location=device)

    net = Net(input_size, output_size, hidden_zize, output_seq)
    # net.load_state_dict(torch.load(model_pth, map_location=device))
    net = net.to(device=device)

    # criterion = EuclidLossFunc()
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
    for dir in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
        if (os.path.exists('./models/{}'.format(dir)) == False):
            os.mkdir('./models/{}'.format(dir))
        TRAIN_PTH = './dataset/{}/train_8-8'.format(dir)
        MODEL_PTH = './models/{}/model_8-8.pth'.format(dir)
        train(TRAIN_PTH, MODEL_PTH, INPUT_SIZE, OUTPUT_SIZE, INPUT_SEQ, OUTPUT_SEQ, HIDDEN_SIZE, DEVICE)