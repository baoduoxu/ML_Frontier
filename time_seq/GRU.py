# 用 GRU 模型来处理时序序列
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path

from data_process import X_test,X_train,y_test,y_train,Labels
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# global variables
num_workers = 0
hidden_size = 16 # 隐藏层维度
input_size = np.shape(X_train)[2]
output_size = len(Labels)
seq_len=np.shape(X_train)[1]
batch_size=1
num_layers=3


X_train = torch.Tensor(X_train)

y_train = torch.Tensor(y_train)

X_test = torch.Tensor(X_test)

y_test = torch.Tensor(y_test)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

print(train_dataset[0][0].shape, train_dataset[0][1])
# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)
#         # self.softmax = nn.LogSoftmax(dim=2)

#     def forward(self, input, hidden):
#         # print(input.shape, hidden.shape)
#         output, hidden = self.gru(input, hidden) 
#         # output: (seq_len, batch, num_directions * hidden_size), hidden: (num_layers * num_directions, batch, hidden_size),其中num_directions为RNN的方向数,单向为1,双向为2, output是每个时间步GRU的输出, hidden包含了每个隐藏层最后一个时间步的状态
#         output = self.fc(output)
#         # output = self.softmax(output)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, batch_size, self.hidden_size)

class MLGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout=nn.Dropout(p=0.5)
        # self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        # print(input.shape, hidden.shape)
        output, hidden = self.gru(input, hidden) 
        # output: (seq_len, batch, num_directions * hidden_size), hidden: (num_layers * num_directions, batch, hidden_size),其中num_directions为RNN的方向数,单向为1,双向为2, output是每个时间步GRU的输出, hidden包含了每个隐藏层最后一个时间步的状态
        output = self.fc(output)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

def train_and_test(net,num_epochs,train_dataset,test_dataset,batch_size,optimizer,loss): 
    # 参数为 网络结构 训练的总epoch数目 训练数据集 测试数据集 batch大小 优化器 损失函数 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) # 将模型移至GPU
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
    
    losses = [] # 存放训练loss的变化
    test_accs = [] # 存放测试集准确率的变化
    train_accs=[]
    n_train=len(train_dataset) # 训练集样本数目
    n_test=len(test_dataset)
    for epoch in range(num_epochs):
        train_l_sum, train_acc= 0.0, 0.0
        correct=0
        for X, y in train_iter: # 遍历测试集的每一个样本
            # print(y)
            y=y.long()
            X, y = X.to(device), y.to(device) # 数据移至GPU
            init_hidden=torch.zeros(num_layers,batch_size,hidden_size).to(device) # 初始化隐藏层
            # print(X.shape,y.shape)
            output=net(X,init_hidden)
            optimizer.zero_grad() # 清空梯度
            # print(type(output[-1]),type(y))
            # print(len(output),output[0][:, -1, :].shape,output[1].shape,y)
            l=loss(output[0][:, -1, :],y) 
            l.backward() # 反向传播
            train_l_sum+=l.item() # 计算总的loss
            optimizer.step() # 更新模型参数
            # 计算训练准确率和测试准确率
            _,predicted=torch.max(output[0][:, -1, :].data,1) # 找到最大值并返回最大值的索引
            correct+=((predicted==y).sum().item()) # 预测正确的数量
        train_acc=correct / n_train # 该 epoch 的测试准确率
        train_accs.append(train_acc)
        correct=0
        with torch.no_grad():
            for X,y in test_iter: # 测试集数据
                y=y.long()
                X, y = X.to(device), y.to(device) # 数据移至GPU
                init_hidden=torch.zeros(num_layers,batch_size,hidden_size).to(device)
                output=net(X,init_hidden)
                l=loss(output[0][:, -1, :],y)
                _,predicted=torch.max(output[0][:, -1, :].data,1)
                correct+=((predicted==y).sum().item()) # 正确的个数
            test_acc=correct/n_test
            test_accs.append(test_acc)
        losses.append(train_l_sum / n_train)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / n_train, train_acc, test_acc))
        torch.cuda.empty_cache() # unleash the memory of GPU
        
    return losses,train_accs,test_accs # 返回训练过程中的所有信息

# net=GRUModel(input_size,hidden_size,output_size)
net=MLGRU(input_size,hidden_size,num_layers,output_size)
print(net)
num_epochs=50
lr,momentum,weight_decay=0.003,0.9,0.001
# optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
optimizer=torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer=torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
loss=nn.CrossEntropyLoss()
losses,train_accs,test_accs=train_and_test(net,num_epochs,train_dataset,test_dataset,batch_size,optimizer,loss)