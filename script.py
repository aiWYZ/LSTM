from trajectory_pred import *
from load_data import *
from trajectory import *

train_params = Params(
    batch = 128,
    epochs = 200,
    hidden_size = 100,
    data_dimension = 2,
    seq_length = 8,
    eval_itv = 10,
)
train_data_path = 'LSTM/data/train_data_128'
val_data_path = 'LSTM/data/val_data_32'
test_data_path = 'LSTM/data/test_data_1'
model_idx=0
run(train_data_path, val_data_path, train_params, model_idx)
test_model(train_params, model_idx, test_data_path)