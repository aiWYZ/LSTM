import torch
from trajectory_pred import *

params = Params(
        batch = 1,
        epochs = 10,
        hidden_size = 100,
        data_dimension=2,
        seq_length = 8,
    )
for i in range(0, 80, 10):
    model = LSTM(params).to("cuda")
    model.load_state_dict(torch.load(f'LSTM/test_outs/128_100_8_80/1_e{i}'))
    data = torch.load('LSTM/val_data_norm_1')
    loss_func = nn.MSELoss(reduction='mean')
    losses = []
    for val_input, val_output in data:
        with torch.no_grad():
            pred = model(val_input)
            loss_item = loss_func(pred, val_output)
            if (loss_item > 100):
                #print(val_input, val_output, pred)
                continue
            losses.append(loss_item.item())
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(np.arange(len(losses)), losses, linewidth=1)
    plt.savefig(f'LSTM/test_outs/128_100_8_80/1_e{i}_val.jpg')
    print(np.average(np.array(losses)))