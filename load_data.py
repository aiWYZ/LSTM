import os
import numpy as np
import torch
import random

def read_file(_path, delim='\t')->np.ndarray:
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def load_data(data_type:str, dir_path:str, out_path:str, in_length:int, out_length:int, batch_size:int):
    files = os.listdir(dir_path)
    files = [os.path.join(dir_path, _path) for _path in files]
    init_data = []
    for path in files:
        data = read_file(path)
        pedestrains = np.unique(data[:, 1]).tolist()
        for ped in pedestrains:
            ped_x = data[ped == data[:, 1], 2]
            ped_y = data[ped == data[:, 1], 3]
            for itr in range(ped_x.shape[0] - in_length - out_length + 1):
                local_x, local_y = ped_x[itr:itr+in_length+out_length], ped_y[itr:itr+in_length+out_length]
                x_avg, y_avg = np.average(local_x), np.average(local_y)
                x_std, y_std = np.std(local_x), np.std(local_y)
                norm_x = (((local_x - x_avg) / x_std) if x_std != 0 else (local_x - x_avg))
                norm_y = (((local_y - y_avg) / y_std) if y_std != 0 else (local_y - y_avg))
                train_seq, train_result = np.zeros((in_length, 2)), np.zeros((out_length, 2))
                train_seq[:, 0], train_seq[:, 1] = norm_x[0:in_length], norm_y[0:in_length]
                train_result[:, 0], train_result[:, 1] = norm_x[in_length:], norm_y[in_length:]
                init_data.append((train_seq, train_result))
    random.shuffle(init_data)
    train_data = []
    for i in range(0, len(init_data) // batch_size):
        train_matrix = torch.zeros(batch_size, in_length, 2, device="cuda")
        result_matrix = torch.zeros(batch_size, out_length, 2, device="cuda")
        for batch_itr in range(batch_size):
            init_index = i * batch_size + batch_itr
            train_matrix[batch_itr] = torch.from_numpy(init_data[init_index][0])
            result_matrix[batch_itr] = torch.from_numpy(init_data[init_index][1])
        train_data.append((train_matrix, result_matrix))
    torch.save(train_data, f'{out_path}/{data_type}_b{batch_size}_{in_length}on{out_length}')

load_data('test', 'D:\\AI\\Python\\.vscode\\LSTM\\datasets\\eth\\test', 'LSTM/data', 7, 7, 1)