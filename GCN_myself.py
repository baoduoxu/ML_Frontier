import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch_geometric as pyg
import torch_geometric.data as pyg_data
from torch_geometric.nn import GCNConv
from data_process import seed, set_seed
from data_process import X_test, X_train, X_valid, y_test_master
from data_process import y_test, y_train, y_train_master, y_valid_master, y_valid
from data_process import X_valid_MCI, X_train_MCI, y_valid_MCI, y_train_MCI, Labels
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
import networkx as nx
from sklearn.decomposition import PCA


set_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Hyperparameters
run_epochs = 200
lr = 0.01
weight_decay = 5e-4
graph_type = 'knn'      # 'knn', 'random1', 'random2'
k_neighbors = 30
knn_metric = 'cosine'      # 'cosine', 'p', 'sqeucledian'
drop_ratio = 1
# 'seuclidean', 'p', 'sqeuclidean', 'mahalanobis', 'pyfunc', 'jaccard', 'nan_euclidean', 'cityblock', 'manhattan', 'precomputed', 'cosine', 'yule', 'infinity', 'sokalsneath', 'rogerstanimoto', 'euclidean', 'russellrao', 'canberra', 'haversine', 'correlation', 'l1', 'chebyshev', 'sokalmichener', 'braycurtis', 'dice', 'l2', 'hamming', 'minkowski'
density = 0.1




### Load data
train_num = X_train.shape[0]
val_num = X_valid.shape[0]
test_num = X_test.shape[0]
num_features = X_train.shape[1]
num_classes = len(Labels)


## Construct the graph
def construct_knn_graph(X, y, val_idx, k=5, metric='unsupervised', ratio=0.3, drop_class=1, density=0.1, drop_edges=24):
    '''
    Description: Construct the knn graph of the data.
    Input:
    - X: Train data and test data
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
            if i>=val_idx[0]:
                drop_class_idx = random.sample(range(len(Labels)), drop_class)
                if indices[i][j] < val_idx[0] and y[indices[i][j]] in drop_class_idx:
                    continue
            adj_mat[i][indices[i][j]] = 1
            adj_mat[indices[i][j]][i] = 1

    dist_valid=[]
    dist_test=[]
    for i in range(len(X)):  # 尝试一些删边手段，修改邻接矩阵
        for j in range(i+1,len(X)):
            if i >= val_idx[0] and i<test_idx[0] and j >= val_idx[0] and j<test_idx[0]: # 验证集内部进行删边
                if adj_mat[i][j]==1:
                    # print(np.linalg.norm(np.squeeze(X[i])-np.squeeze(X[j])))
                    dist_valid.append([np.linalg.norm(X[i]-X[j]),[i,j]])
            if i>=test_idx[0] and j>=test_idx[0]: # 测试集内部进行删边
                if adj_mat[i][j]==1:
                    dist_test.append([np.linalg.norm(X[i]-X[j]),[i,j]])
                # 测试集内部进行删边
                # drop_edge_idx_test = random.sample(range(val_idx[0],test_idx[0]), drop_edges)
                # if indices[i][j] >= test_idx[0] and y[indices[i][j]] in drop_edge_idx_test:
                #     continue

                # 随机选择 drop_class 个类，断开测试集和验证集中的样本与训练集中这些类的连边
            

                # 随机选择 k/2 个邻居连边
                # random_sample = random.sample(range(k), int(k*ratio))
                # if j in random_sample:
                #     adj_mat[i][indices[i][j]] = 1
                #     adj_mat[indices[i][j]][i] = 1
    dist_valid.sort(key=lambda x:x[0],reverse=True)
    dist_test.sort(key=lambda x:x[0],reverse=True) # 按照距离进行排序
    dist_valid=np.array(dist_valid)
    dist_test=np.array(dist_test)
    drop_valid=dist_valid[:drop_edges]
    drop_test=dist_test[:drop_edges]
    for _,idx in drop_valid:
        adj_mat[idx[0]][idx[1]]=0
        adj_mat[idx[1]][idx[0]]=0
    for _,idx in drop_test:
        adj_mat[idx[0]][idx[1]]=0
        adj_mat[idx[1]][idx[0]]=0
    for i in val_idx:
        for j in test_idx:
            adj_mat[i][j]=0
            adj_mat[j][i]=0
    # 将连边加入邻接矩阵
    edge_index = []
    for i in range(len(X)):
        for j in range(i+1,len(X)):
            # 随机断开验证集和测试集内的点与连边
            if adj_mat[i][j]==1: # 对于训练集中的数据，只有两个是同一类才连边
                # edge_index.append([i,j])
                if i>=train_num or j>=train_num: # 如果两个点有一个不是训练集，就直接根据邻接矩阵连边
                    edge_index.append([i, j])
                if i<train_num and j<train_num and y[i]==y[j]:
                    edge_index.append([i, j])
                # if i >= val_idx[0] and np.random.rand()<=density: # 对于验证集内部的点，随机断开一些连边
                # drop_edge_idx = random.sample(range(val_idx[0],test_idx[0]), drop_edges)
                # if 
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_index.to(device)
    X = X.astype(np.float32)
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)
    # print(X.shape,y.shape)
    data = pyg.data.Data(x=X, edge_index=edge_index, y=y)
    return data


# def construct_knn_graph(X, y, val_idx, k=5, metric='unsupervised', ratio=0.3, drop_class=0, density=0.1, drop_edges=0):
#     '''
#     Description: Construct the knn graph of the data.
    
#     Input:
#     - X: Train data and test data.
#     - y: The label of train data.
#     - k: The number of nearest neighbors.
    
#     Return:
#         - the knn graph of the data.
#     '''
#     knn=NearestNeighbors(n_neighbors=k, metric=metric)
#     knn.fit(X)
#     _, indices = knn.kneighbors(X) # Indices of the nearest points in the population matrix.
#     adj_mat=np.zeros((len(X),len(X)))
#     for i in range(len(X)):
#         for j in range(k):
#             adj_mat[i][indices[i][j]]=1
#     edge_index = []
#     for i in range(len(X_train)):
#         for j in range(len(X_train)):
#             if j>i and adj_mat[i][j]==1:
#                 edge_index.append([i,j])
#     edge_index = torch.tensor(edge_index).T
#     edge_index = edge_index.to(device)
#     X=X.astype(np.float32)
#     X = torch.tensor(X).float().to(device)
#     y = torch.tensor(y).long().to(device)
#     # print(X.shape,y.shape)
#     data = pyg.data.Data(x=X,edge_index=edge_index,y=y)
#     return data






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

    data = pyg_data.Data(x=X, edge_index=edge_index, y=y)

    return data


### Define the model
class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # self.conv1 = GCNConv(num_node_features, 64)
        # self.conv2 = GCNConv(64, num_classes)
        # self.norm = torch.nn.BatchNorm1d(64)
        self.conv1 = GATConv(num_node_features, 4, heads=8, dropout=0.5)
        self.conv2 = GATConv(4*8, num_classes, heads=8, concat=False, dropout=0.5)
        self.norm = torch.nn.BatchNorm1d(4*8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


### Train the model
def train():

    test_acc = 0
    best_test_acc = 0
    best_test_epoch = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(run_epochs):
        model.train()
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        train_acc, val_acc, test_acc = test()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch + 1
            print("Best Epoch {:03d}: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(epoch+1, train_acc, val_acc, test_acc))
        if (epoch+1) <100:
            log = 'Epoch: {:03d}, Loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}; Best Test: {:.4f} (epoch {:03d})'
            print(log.format(epoch+1, loss, train_acc, val_acc, test_acc, best_test_acc, best_test_epoch))
        if (epoch+1) % 100 == 0:
            log = 'Epoch: {:03d}, Loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}; Best Test: {:.4f} (epoch {:03d})'
            print(log.format(epoch+1, loss, train_acc, val_acc, test_acc, best_test_acc, best_test_epoch))


### Test the model
def test():
    model.eval()
    log, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = log[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs



if __name__ == '__main__':

    model = GCN(num_features, num_classes).to(device)

    ### Load and split the data
    X = np.concatenate((X_train, X_valid, X_test), axis=0)
    y = np.concatenate((y_train, y_valid, y_test), axis=0)

    train_idx = np.array(range(X_train.shape[0]))
    val_idx = np.array(range(X_train.shape[0], X_train.shape[0]+X_valid.shape[0]))
    train_and_val_idx = np.array(range(X_train.shape[0]+X_valid.shape[0]))
    test_idx = np.array(range(X_train.shape[0]+X_valid.shape[0], X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]))
    print(X.shape, y.shape)
    all_f = np.zeros((X.shape[0],), dtype=np.bool)

    if graph_type == 'knn':
        data = construct_knn_graph(X, y, val_idx=val_idx, k=k_neighbors, metric=knn_metric, ratio=drop_ratio, density=density)
    elif graph_type == 'random1':
        data = construct_random1_graph(X, y, num_neighbors=k_neighbors, density=density)
    elif graph_type == 'random2':
        data = construct_random2_graph(X, y, num_neighbors=k_neighbors, prob_rewire=density)

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

    ### Train the model and test the model
    train()