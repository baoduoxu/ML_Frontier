import sys,os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
# current_directory = os.path.abspath(os.path.curdir)
# parent_directory = os.path.dirname(current_directory)
# sys.path.append(parent_directory)
from data_process import X_test, X_train, X_valid, y_test_master
from data_process import y_test, y_train, y_train_master, y_valid_master, y_valid
from data_process import X_valid_MCI, X_train_MCI, y_valid_MCI, y_train_MCI, Labels
# from dimension_reduction import X_train,X_valid,X_test
from dimension_reduction_lda import dimension_reduction, visualization
from train_and_test import train_and_test, device
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

train_num = X_train.shape[0]
val_num = X_valid.shape[0]
test_num = X_test.shape[0]

def construct_knn_graph(X, y, k=5, metric='unsupervised'):
    '''
    Description: Construct the knn graph of the data.
    Input:
    - X: Train data and test data.
    - y: The label of train data.
    - k: The number of nearest neighbors.
    Return:
        - the knn graph of the data.
    '''
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    knn.fit(X)
    _, indices = knn.kneighbors(X) # Indices of the nearest points in the population matrix.
    adj_mat = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(k):
            adj_mat[i][indices[i][j]] = 1
            adj_mat[indices[i][j]][i] = 1
    edge_index = []
    for i in range(len(X)):
        for j in range(len(X)):
            if j>i and adj_mat[i][j]==1: # 对于训练集中的数据，只有两个是同一类才连边
                # edge_index.append([i,j])
                if i>=train_num or j>=train_num: # 如果两个点有一个不是训练集，就直接根据邻接矩阵连边
                    edge_index.append([i, j])
                if i<train_num and j<train_num and y[i]==y[j]:
                    edge_index.append([i, j])
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_index.to(device)
    X = X.astype(np.float32)
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)
    # print(X.shape,y.shape)
    data = pyg.data.Data(x=X, edge_index=edge_index, y=y)
    return data


def construct_random1_graph(X, y, num_neighbors=5, density=0.1):
    '''
    Description: Construct a random graph from the data.
    Input:
    - X: Data.
    - y: Labels.
    - num_neighbors: The number of neighbors for each node.
    - density: The edge density in the random graph.
    Return:
    - A random graph as a PyTorch Geometric Data object.
    '''

    num_nodes = len(X)
    adj_mat = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < density:
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1

    # Randomly select 'num_neighbors' neighbors for each node
    for i in range(num_nodes):
        neighbors = np.where(adj_mat[i] == 1)[0]
        if len(neighbors) > num_neighbors:
            random_neighbors = np.random.choice(neighbors, num_neighbors, replace=False)
            adj_mat[i] = 0
            adj_mat[i][random_neighbors] = 1

    edge_index = np.where(adj_mat == 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

    X = X.astype(np.float32)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    data = pyg.data.Data(x=X, edge_index=edge_index, y=y)

    return data

def construct_random2_graph(X, y, num_neighbors=5, prob_rewire=0.2):
    '''
    Description: Construct a random graph from the data.
    Input:
    - X: Data.
    - y: Labels.
    - num_neighbors: The number of neighbors for each node.
    - prob_rewire: The probability of rewiring edges in the Watts-Strogatz model.
    Return:
    - A random graph as a PyTorch Geometric Data object.
    '''

    # Create a Watts-Strogatz random graph
    G = nx.watts_strogatz_graph(len(X), num_neighbors, prob_rewire)

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t()
    edge_index = edge_index.to(device)

    X = X.astype(np.float32)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    data = pyg.data.Data(x=X, edge_index=edge_index, y=y)

    return data




X = np.concatenate((X_train, X_valid, X_test), axis=0)
# y=np.concatenate((y_train_master,y_valid_master,y_test_master),axis=0) # for ANDI
y = np.concatenate((y_train, y_valid, y_test), axis=0)
# for ADNI, not PPMI
# y[y==1]=0
# y[y==4]=1
# y[y==-1]=2
# print(X,y)
data = construct_knn_graph(X, y, k=50, metric='euclidean')
# data = construct_random1_graph(X, y, num_neighbors=50, density=0.1)
# data = construct_random2_graph(X, y, num_neighbors=5, prob_rewire=0.2)

train_idx = np.array(range(X_train.shape[0]))
val_idx = np.array(range(X_train.shape[0], X_train.shape[0]+X_valid.shape[0]))
train_and_val_idx = np.array(range(X_train.shape[0]+X_valid.shape[0]))
test_idx = np.array(range(X_train.shape[0]+X_valid.shape[0], X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]))
print(X.shape, y.shape)
all_f = np.zeros((X.shape[0],), dtype=np.bool)

all_f_tmp = all_f.copy()
all_f_tmp[train_idx] = True
train_mask = all_f_tmp

all_f_tmp = all_f.copy()
all_f_tmp[val_idx] = True
val_mask = all_f_tmp

all_f_tmp = all_f.copy()
all_f_tmp[train_and_val_idx] = True
train_and_val_mask = all_f_tmp

all_f_tmp = all_f.copy()
all_f_tmp[test_idx] = True
test_mask = all_f_tmp

print(y.shape, train_idx.shape, val_idx.shape, train_and_val_idx.shape, test_idx.shape)
print(train_mask.shape, val_mask.shape, train_and_val_mask.shape, test_mask.shape)
print(y[train_mask].shape, y[val_mask].shape, y[train_and_val_mask].shape, y[test_mask].shape)

data.train_mask = train_mask
data.val_mask = val_mask
data.train_and_val_mask = train_and_val_mask
data.test_mask = test_mask


print(data)

import argparse
import torch_geometric.transforms as T
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

num_features = X_train.shape[1]
num_classes = 2

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # GCNConv模型输入参数：输入结点特征维度，输出结点特征维度，是否cached和是否normalize；
        self.conv1 = pyg_nn.GCNConv(num_features, 16, cached=True, normalize=not args.use_gdc)
        self.conv2 = pyg_nn.GCNConv(16, num_classes, cached=True, normalize=not args.use_gdc)
        self.fc3 = nn.Linear(num_classes, 16)
        self.fc4 = nn.Linear(16, num_classes)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.bn = nn.BatchNorm1d(16)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = self.bn2(x)

        
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)

        return F.log_softmax(x, dim=1)  # 模型最后一层接上一个softmax和CNN类似


model, data = GCN().to(device), data.to(device)


# optimizer = torch.optim.SGD([
#     dict(params=model.reg_params, weight_decay=1e-4),
#     dict(params=model.non_reg_params, weight_decay=0)
# ], lr=1e-2, momentum=0.9, nesterov=True)

optimizer = torch.optim.AdamW([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=1e-2)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5000, gamma=0.5)


def train():
    model.train()
    optimizer.zero_grad()
    output = model()
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    return loss

# @torch.no_grad()

def test():
    model.eval()
    log, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = log[mask].max(1)[1]
        # print("pred", pred)
        # print("data", data.y[mask])
        acc = pred.eq(data.y[mask]).sum().item()/mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 30001):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    if epoch % 100 == 0:
        print(log.format(epoch, loss, train_acc, best_val_acc, test_acc))