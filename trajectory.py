import torch
import random
import os
from trajectory_pred import *

def test_model(train_params:Params, model_index:int, data_path:str):
    test_params = Params(
        batch = 1,
        epochs = train_params._epochs,
        hidden_size = train_params._features,
        data_dimension = train_params._dim,
        seq_length = train_params._seq_length,
    )
    model = LSTM(test_params).to("cuda")
    model.load_state_dict(torch.load(f'{train_params._path}{model_index}'))
    data = torch.load(data_path)
    print(len(data))
    outfig_path = f'{train_params._path}{model_index}_test_fig/'
    if not os.path.exists(outfig_path): os.mkdir(outfig_path)
    for i in range(100):
        test_input, test_output = random.choice(data)
        model.clear_cells()
        test_pred = model(test_input).view(-1)

        input_data = test_input.view(-1, 2).detach().cpu().numpy()
        output_data = test_output.view(-1).detach().cpu().numpy()
        pred_data = test_pred.detach().cpu().numpy()

        plt.clf()
        plt.scatter(input_data[:, 0], input_data[:, 1], color='red')
        plt.scatter(pred_data[0], pred_data[1], color='green')
        plt.text(pred_data[0], pred_data[1], 'pred')
        plt.scatter(output_data[0], output_data[1], color='blue')
        plt.text(output_data[0], output_data[1], 'real')
        plt.savefig(f'{train_params._path}{model_index}_test_fig/{i}.jpg')