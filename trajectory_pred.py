import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class Params():
    def __init__(self, lr:float=1e-3, batch:int=1, epochs:int=280, hidden_size:int=100, 
                 data_dimension:int=1, seq_length:int=18, use_gpu:bool=True,
                 eval_itv:int=10):
        self._lr = lr
        self._batch_size = batch
        self._epochs = epochs
        self._features = hidden_size
        self._dim = data_dimension
        self._seq_length = seq_length
        self._eval_interval = eval_itv
        self._device = torch.device("cuda" if use_gpu else "cpu")
        self._path = (f'LSTM/test_outs/{self._batch_size}_{self._features}_' 
                + f'{self._seq_length}_{self._epochs}/')

class LSTM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self._params = params
        self._features = params._features
        self.lstm = nn.LSTM(params._dim, params._features, batch_first=True)
        self.linear = nn.Linear(params._features, params._dim)
        self.cell = (torch.zeros(1, params._batch_size, params._features, device=params._device), 
                     torch.zeros(1, params._batch_size, params._features, device=params._device))
    
    def forward(self, input):
        # input is [_batch, length, _dim], output is [_batch, length, _features]
        lstm_out, self.cell = self.lstm(input, self.cell)
        # go through linear to make output [_batch, length, _dim]
        predictions = self.linear(lstm_out)
        # the last one in length dimension is the result
        return predictions[:, -1, :]
    
    def clear_cells(self, isTraining:bool=True, batch_size:int=-1):
        params = self._params
        temp_batch = params._batch_size if isTraining else batch_size
        self.cell = (torch.zeros(1, temp_batch, params._features, device=params._device),
                     torch.zeros(1, temp_batch, params._features, device=params._device))

def plot_loss(input:list, params:Params, text:str, avg_num:int):
    input_array = np.array(input)
    sum_array = np.zeros(len(input) + avg_num)
    for i in range(avg_num):
        sum_array[i:i+len(input)] = sum_array[i:i+len(input)] + input_array
    avg_array = sum_array[avg_num:-avg_num] / 10
    log_avg_array = np.log10(avg_array)
    x = np.arange(avg_array.size)
    plt.plot(x, avg_array, linewidth=1)
    plt.savefig(params._path + f'{text}_loss.jpg')
    plt.clf()
    plt.plot(x, log_avg_array, linewidth=1)
    plt.savefig(params._path + f'{text}_logloss.jpg')
    plt.clf()
    plt.close()

def plot_eloss(train_loss:list, val_loss:list, params:Params, text:str, avg_num:int):
    input_array = np.array(train_loss)
    sum_array = np.zeros(len(train_loss) + avg_num)
    for i in range(avg_num):
        sum_array[i:i+len(train_loss)] = sum_array[i:i+len(train_loss)] + input_array
    avg_array = sum_array[avg_num:-avg_num] / avg_num
    log_avg_array = np.log10(avg_array)
    x = np.arange(avg_array.size)
    val_array = np.array(val_loss)
    log_val_array = np.log10(val_array)
    plt.plot(x, avg_array, linewidth=1)
    plt.scatter([i for i in range(0, params._epochs, params._eval_interval)], val_array)
    plt.savefig(params._path + f'{text}_loss.jpg')
    plt.clf()
    plt.plot(x, log_avg_array, linewidth=1)
    plt.scatter([i for i in range(0, params._epochs, params._eval_interval)], log_val_array)
    plt.savefig(params._path + f'{text}_logloss.jpg')
    plt.clf()
    plt.close()

def eval(model_path:str, data_path:str, params:Params):
    model = LSTM(params).to("cuda")
    model.load_state_dict(torch.load(model_path))
    data = torch.load(data_path)
    loss_func = nn.MSELoss(reduction='mean')
    losses = []
    for val_input, val_output in data:
        with torch.no_grad():
            model.clear_cells()
            loss_item = loss_func(model(val_input), val_output).item()
            losses.append(loss_item)
    return np.average(np.array(losses))

def train(model:LSTM, optimizer:torch.optim.Adam, data:list, params:Params, val_data_path:str, model_idx:int):
    epochs = params._epochs
    loss_func = nn.MSELoss(reduction='mean')
    losses = []
    epoch_losses = []
    val_losses = []
    val_params = Params(
        batch = 32,
        epochs = params._epochs,
        hidden_size = params._features,
        data_dimension = params._dim,
        seq_length = params._seq_length,
    )

    for i in range(epochs):
        time_start = time.time()
        temp_losses = []
        for train_input, train_output in data:
            optimizer.zero_grad()
            model.clear_cells(isTraining=True)
            prediction = model(train_input)

            loss_item = loss_func(prediction, train_output)
            losses.append(loss_item.item())
            temp_losses.append(loss_item.item())
            
            loss_item.backward()
            optimizer.step()
        epoch_loss = np.average(np.array(temp_losses))
        epoch_losses.append(epoch_loss)
        time_end = time.time()
        print(f'epoch {i:3} use {time_end - time_start} seconds, loss {epoch_loss}')
        if i % 10 == 0:
            model_path = params._path + f'{model_idx}_e{i}'
            torch.save(model.state_dict(), model_path)
            val_losses.append(eval(model_path, val_data_path, val_params))

    torch.save(model.state_dict(), params._path + f'{model_idx}')
    return losses, epoch_losses, val_losses

def run(train_data_path:str, val_data_path:str, params:Params, model_idx:int=0):
    data = torch.load(train_data_path)
    print(len(data))
    if not os.path.exists(params._path): os.mkdir(params._path)
    model = LSTM(params).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), params._lr)
    losses, epoch_losses, val_losses = train(model, optimizer, data, params, val_data_path, model_idx)
    plot_loss(losses, params, 'real', 100)
    plot_eloss(epoch_losses, val_losses, params, 'epoch', 1)