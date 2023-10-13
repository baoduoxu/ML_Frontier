# 用 LSTM 模型来处理时序序列
import sys, os
from data_process import X_test, X_train, y_test, y_train, Labels
from train_and_test import train_and_test, device

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # 添加上层目录到 sys.path



X_train = torch.Tensor(np.mean(X_train, axis=2))

y_train = torch.Tensor(y_train)

X_test = torch.Tensor(np.mean(X_test, axis=2))

y_test = torch.Tensor(y_test)

# 将X_train从2维张量变成3维张量
X_train = torch.unsqueeze(X_train, dim=2)

# 将X_test从2维张量变成3维张量
X_test = torch.unsqueeze(X_test, dim=2)

# global variables
num_workers = 0
hidden_size = 16  # 隐藏层维度
input_size = np.shape(X_train)[2]
output_size = len(Labels)
seq_len = np.shape(X_train)[1]
batch_size = 2
num_layers = 3
num_directions = 2

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

print(train_dataset[0][0].shape, train_dataset[0][1])


class MLLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=False)
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
        return (torch.ones(self.num_layers * 1, batch_size, self.hidden_size),
                torch.ones(self.num_layers * 1, batch_size, self.hidden_size))


net = MLLSTM(input_size, hidden_size, num_layers, output_size)
print(net)
num_epochs = 600
lr, momentum, weight_decay = 0.0008, 0.9, 0.001
optimizer = torch.optim.RMSprop(net.parameters(),
                                lr=lr,
                                alpha=0.9,
                                weight_decay=weight_decay)
# optimizer=torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer=torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay)
loss = nn.CrossEntropyLoss()
init_hidden = torch.zeros(num_layers * num_directions, batch_size,
                          hidden_size).to(device)
losses, train_accs, test_accs = train_and_test(net, num_epochs, train_dataset,
                                               test_dataset, batch_size,
                                               optimizer, loss, init_hidden)
