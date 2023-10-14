import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
from data_process import X_test, X_train, X_valid, y_test_master
from data_process import y_test, y_train, y_train_master, y_valid_master, y_valid
from data_process import X_valid_MCI, X_train_MCI, y_valid_MCI, y_train_MCI, Labels
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv

# 设置随机种子
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random seed is set to {}.'.format(seed))

train_num = X_train.shape[0]
val_num = X_valid.shape[0]
test_num = X_test.shape[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                if i>=train_num+val_num or j>=train_num+val_num: # 如果两个点有一个不是训练集，就直接根据邻接矩阵连边
                    edge_index.append([i, j])
                if i<train_num+val_num and j<train_num+val_num and y[i]==y[j]:
                    edge_index.append([i, j])
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_index.to(device)
    X = X.astype(np.float32)
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)
    # print(X.shape,y.shape)
    data = pyg.data.Data(x=X, edge_index=edge_index, y=y)
    return data


def construct_random_graph(X, y, num_neighbors=5, density=0.1):
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


class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, num_classes)
        # self.conv1 = GATConv(num_node_features, 32, heads=2, dropout=0.5)
        # self.conv2 = GATConv(32*2, num_classes, heads=4, concat=False, dropout=0.5)
        self.norm = torch.nn.BatchNorm1d(64)

        self.linear1 = nn.Linear(num_node_features, 32)
        self.linear2 = nn.Linear(32, num_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)

        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
    
num_features = X_train.shape[1]
num_classes = 2

model = GCN(num_features, num_features).to(device)

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    for epoch in range(400):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))


def test():
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))




X = np.concatenate((X_train, X_valid, X_test), axis=0)
y = np.concatenate((y_train, y_valid, y_test), axis=0)
# data = construct_knn_graph(X, y, k=186, metric='euclidean')
data = construct_random_graph(X, y, num_neighbors=50, density=0.1)


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


train()
test()


# best_val_acc = test_acc = 0
# for epoch in range(1, 30001):
#     loss = train()
#     train_acc, val_acc, tmp_test_acc = test()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log = 'Epoch: {:03d}, Loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#     if epoch % 100 == 0:
#         print(log.format(epoch, loss, train_acc, best_val_acc, test_acc))