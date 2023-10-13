import sys,os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
# current_directory = os.path.abspath(os.path.curdir)
# parent_directory = os.path.dirname(current_directory)
# sys.path.append(parent_directory)
from data_process import X_test,X_train,X_valid,y_test_master
from data_process import y_test,y_train,y_train_master,y_valid_master,y_valid
from data_process import X_valid_MCI,X_train_MCI,y_valid_MCI,y_train_MCI,Labels
# from dimension_reduction import X_train,X_valid,X_test
from dimension_reduction_lda import dimension_reduction,visualization
from train_and_test import train_and_test,device
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

def construct_knn_graph(X,y, k=5, metric='unsupervised'):
    '''
    Description: Construct the knn graph of the data.
    
    Input:
    - X: Train data and test data.
    - y: The label of train data.
    - k: The number of nearest neighbors.
    
    Return:
        - the knn graph of the data.
    '''
    knn=NearestNeighbors(n_neighbors=k,metric=metric)
    knn.fit(X)
    _, indices = knn.kneighbors(X) # Indices of the nearest points in the population matrix.
    adj_mat=np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(k):
            adj_mat[i][indices[i][j]]=1
    edge_index = []
    for i in range(len(X_train)):
        for j in range(len(X_train)):
            if j>i and adj_mat[i][j]==1:
                edge_index.append([i,j])
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_index.to(device)
    X=X.astype(np.float32)
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)
    # print(X.shape,y.shape)
    data = pyg.data.Data(x=X,edge_index=edge_index,y=y)
    return data

X=np.concatenate((X_train,X_valid,X_test),axis=0)
# y=np.concatenate((y_train_master,y_valid_master,y_test_master),axis=0) # for ANDI
y=np.concatenate((y_train,y_valid,y_test),axis=0)
# for ADNI, not PPMI
# y[y==1]=0
# y[y==4]=1
# y[y==-1]=2
# print(X,y)
data=construct_knn_graph(X,y,k=10,metric='euclidean')

train_idx = np.array(range(X_train.shape[0]))
val_idx = np.array(range(X_train.shape[0],X_train.shape[0]+X_valid.shape[0]))
test_idx = np.array(range(X_train.shape[0]+X_valid.shape[0],X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]))
print(X.shape,y.shape)
all_f=np.zeros((X.shape[0],),dtype=np.bool)

all_f_tmp=all_f.copy()
all_f_tmp[train_idx]=True
train_mask=all_f_tmp

all_f_tmp=all_f.copy()
all_f_tmp[val_idx]=True
val_mask=all_f_tmp

all_f_tmp=all_f.copy()
all_f_tmp[test_idx]=True
test_mask=all_f_tmp

print(y.shape,train_idx.shape,val_idx.shape,test_idx.shape)
print(train_mask.shape,val_mask.shape,test_mask.shape)
print(y[train_mask].shape,y[val_mask].shape,y[test_mask].shape)

data.train_mask=train_mask
data.val_mask=val_mask
data.test_mask=test_mask


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

num_features=X_train.shape[1]
num_classes=2

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # GCNConv模型输入参数：输入结点特征维度，输出结点特征维度，是否cached和是否normalize；
        self.conv1 = pyg_nn.GCNConv(num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        # self.fc1=nn.Linear(16,8)
        # self.fc2=nn.Linear(8,4)
        self.conv2 = pyg_nn.GCNConv(16, num_classes, cached=True,
                             normalize=not args.use_gdc)
        self.fc3=nn.Linear(num_classes,16)
        self.fc4=nn.Linear(16,num_classes)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)  # dropout操作，避免过拟合
        
        # x = self.fc1(x)
        # x = F.relu(x)
        # x=F.dropout(x,training=self.training)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x=F.dropout(x,training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        
        x = self.fc3(x)
        x = F.relu(x)

        x=self.fc4(x)
        # x=F.dropout(x,training=self.training)

        return F.log_softmax(x, dim=1)  # 模型最后一层接上一个softmax和CNN类似

model,data=GCN().to(device),data.to(device)


optimizer = torch.optim.SGD([
    dict(params=model.reg_params, weight_decay=1e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=1e-2, momentum=0.9, nesterov=True)


def train():
    model.train()
    optimizer.zero_grad()
    output=model()
    loss=F.nll_loss(output[data.train_mask],data.y[data.train_mask])
    optimizer.step()
    return loss

@torch.no_grad()

def test():
    model.eval()
    log, accs= model(), []
    for _,mask in data('train_mask','val_mask','test_mask'):
        pred=log[mask].max(1)[1]
        # print(mask,data.y[mask].shape)
        acc=pred.eq(data.y[mask]).sum().item()/mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    loss=train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch,loss, train_acc, best_val_acc, test_acc))