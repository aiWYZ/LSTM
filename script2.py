from traj import *
from load_data import *

train_params = Params(
    batch = 128,
    epochs = 200,
    hidden_size = 100,
    data_dimension = 2,
    in_length = 7,
    out_length = 7,
    eval_itv = 10,
)

train_data_path = 'LSTM/data/train_b128_7on7'
val_data_path = 'LSTM/data/val_b32_7on7'
test_data_path = 'LSTM/data/test_b1_7on7'
model_idx = 0
run(train_data_path, val_data_path, 32, train_params, model_idx)
test_model(train_params, model_idx, 1, test_data_path)