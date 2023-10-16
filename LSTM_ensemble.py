import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path

from data_process import X_test,X_train,X_valid,y_test,y_train,y_valid,Labels
from train_and_test import train_and_test, device

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



X_train = np.concatenate((X_train, X_valid), axis=0)

y_train = np.concatenate((y_train, y_valid), axis=0)

def train_and_test_brain_region(input_region_index, X_train, y_train, X_test, y_test, Labels):
    # global variables
    num_workers = 0
    hidden_size = 16 # 隐藏层维度
    input_size = np.shape(X_train)[2]
    output_size = len(Labels)
    seq_len = np.shape(X_train)[1]
    batch_size = 2
    num_layers = 3
    num_directions = 2
    num_epochs = 300
    lr, momentum, weight_decay = 1e-4, 0.9, 0.001

    X_train = torch.Tensor(X_train).to(device)

    y_train = torch.Tensor(y_train).to(device)

    X_test = torch.Tensor(X_test).to(device)

    y_test = torch.Tensor(y_test).to(device)

    # 在X_train中提取指定脑区数据
    X_train_region = torch.Tensor(X_train[:, input_region_index, :]).to(device)
    X_train_region = torch.unsqueeze(X_train_region, dim=1)

    # 在X_test中提取指定脑区数据
    X_test_region = torch.Tensor(X_test[:, input_region_index, :]).to(device)
    X_test_region = torch.unsqueeze(X_test_region, dim=1)

    train_dataset_region = torch.utils.data.TensorDataset(X_train_region, y_train)
    test_dataset_region = torch.utils.data.TensorDataset(X_test_region, y_test)

    class MLLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(MLLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, input, hidden):
            batch_size = input.shape[0]
            hidden = self.initHidden(batch_size)
            output, _ = self.lstm(input, hidden)
            output = self.fc(output)
            output = self.dropout(output)
            return torch.squeeze(output[:, -1:, :], dim=1)

        def initHidden(self, batch_size):
            # both hidden and cell states
            return (torch.ones(self.num_layers * 1, batch_size, self.hidden_size).to(device),
                    torch.ones(self.num_layers * 1, batch_size, self.hidden_size).to(device))

    
    net_region = MLLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    optimizer_region = torch.optim.RMSprop(net_region.parameters(), lr=lr, alpha=0.9, weight_decay=weight_decay)

    loss = nn.CrossEntropyLoss().to(device)

    init_hidden_region = torch.zeros(num_layers * num_directions, batch_size, hidden_size).to(device)

    losses, train_accs, test_accs = train_and_test(net_region, num_epochs, train_dataset_region, test_dataset_region, batch_size, optimizer_region, loss, init_hidden_region)

    return losses, train_accs, test_accs, net_region

all_losses = []
all_train_accs = []
all_test_accs = []
all_models = []

for region_index in range(np.shape(X_train)[1]):
    losses, train_accs, test_accs, model = train_and_test_brain_region(region_index, X_train, y_train, X_test, y_test, Labels)
    all_losses.append(losses)
    all_train_accs.append(train_accs)
    all_test_accs.append(test_accs)
    all_models.append(model)
    print(f"all_losses={all_losses},\nall_train_accs={all_train_accs},\nall_test_accs={all_test_accs},\nall_models={all_models}")

def ensemble_predict(models, input):
    predictions = []
    for model in models:
        # 将模型设置为评估模式
        model.eval()
        with torch.no_grad():
            output = model(input)
            predictions.append(output)
    predictions = torch.stack(predictions)
    ensemble_output = torch.mean(predictions, dim=0)
    _, predicted_labels = torch.max(ensemble_output, dim=1)
    return predicted_labels

# 使用集成模型进行预测
X_test_ensemble = torch.Tensor(X_test).to(device)
X_test_ensemble = torch.unsqueeze(X_test_ensemble, dim=1)
predicted_labels = ensemble_predict(all_models, X_test_ensemble)
