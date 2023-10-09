import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
from data_process import X_test,X_train,X_valid,y_test_master
from data_process import y_test,y_train,y_train_master,y_valid_master,y_valid
from data_process import X_valid_MCI,X_train_MCI,y_valid_MCI,y_train_MCI
# from dimension_reduction import X_train,X_valid,X_test
from dimension_reduction_lda import dimension_reduction,visualization
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
# 主分类器
dim=2
def _svm(X_train,y_train,X_test,y_test):
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
        clf=svm.SVC(kernel='linear',C=1.0,random_state=0,decision_function_shape='ovr')
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(accuracy)
        # print(clf.coef_,clf.intercept_)
        return y_pred

def classifier(X_test): # 主分类器, 将1,2,3看作一类-1
    X_train_new=np.concatenate((X_train,X_valid),axis=0)
    y_train_master_new=np.concatenate((y_train_master,y_valid_master),axis=0) # 无需验证集, 将train和validation合并


    # X_train_reduced=dimension_reduction(X_train_new,y_train_master_new,dim=110,method='PCA')
    # # X_train_reduced=dimension_reduction(X_train_reduced,y_train_master_new,dim=60,method='Laplacian')
    # X_train_reduced,_=dimension_reduction(X_train_reduced,y_train_master_new,dim=dim,method='LDA')
    # visualization(X_train_reduced,y_train_master_new,dim=dim,method='LDA',class_num=3)

    # X_valid_reduced=dimension_reduction(X_valid,y_valid_master,dim=40,method='PCA')
    # # X_train_reduced=dimension_reduction(X_train_reduced,y_train_master_new,dim=60,method='Laplacian')
    # X_valid_reduced,proj=dimension_reduction(X_valid_reduced,y_valid_master,dim=dim,method='LDA')
    # visualization(X_valid_reduced,y_valid_master,dim=dim,method='LDA',class_num=3)
    
    
    # X_test_reduced=dimension_reduction(X_test,y_test_master,dim=40,method='PCA')
    # X_test_reduced=np.dot(X_test_reduced,proj)[:,:dim]
    # # X_test_reduced,_=dimension_reduction(X_test_reduced,y_test_master,dim=dim,method='LDA')
    # visualization(X_test_reduced,y_test_master,dim=dim,method='LDA',class_num=3)
    
    # # 对 X_train 进行降维
    # X_train_reduced,_=dimension_reduction(X_train_new,y_train_master_new,dim=2,method='LDA')
    # X_valid_reduced,_=dimension_reduction(X_valid,y_valid_master,dim=2,method='LDA')
    # _,proj=dimension_reduction(X_valid,y_valid_master,dim=2,method='LDA')
    # print(proj.shape)
    # X_test_reduced,_=dimension_reduction(X_test,y_test_master,dim=2,method='LDA')
    # # X_test_reduced=np.dot(X_test-np.mean(X_test),proj)[:,:dim] # 使用对训练集降维的线性变换对测试集进行降维
    y_pred=_svm(X_train_new,y_train_master_new,X_test,y_test_master) # 不降维直接用SVM
    print(y_pred)
    print(y_test_master)
    # y_pred=_svm(X_train_reduced,y_train_master_new,X_test_reduced,y_test_master)
    X_test_MCI=X_test[y_pred==-1,:] # 被分为-1的那些样本
    y_test_MCI=y_test[y_pred==-1]
    # # visualization(X_train_reduced,y_train_master_new,dim=dim,method='LDA',class_num=3)
    # # visualization(X_valid_reduced,y_valid_master,dim=dim,method='LDA',class_num=3)
    # # visualization(X_test_reduced,y_test_master,dim=dim,method='LDA',class_num=3)
    # # visualization(X_test_reduced,y_pred,dim=dim,method='LDA',class_num=3)



    X_train_MCI_new=np.concatenate((X_train_MCI,X_valid_MCI),axis=0)
    y_train_MCI_new=np.concatenate((y_train_MCI,y_valid_MCI),axis=0)


    # # 对 X_train_MCI 进行降维
    # X_train_MCI_reduced,proj=dimension_reduction(X_train_MCI_new,y_train_MCI_new,dim=2,method='LDA')
    # print(proj.shape)
    # X_test_MCI_reduced=np.dot(X_test_MCI-np.mean(X_test_MCI),proj)[:,:dim] # 使用对训练集降维的线性变换对测试集进行降维
    # # visualization(X_test_MCI_reduced,y_test_MCI,dim=dim,method='LDA',class_num=3)
    y_pred_MCI=_svm(X_train_MCI_new,y_train_MCI_new,X_test_MCI,y_test_MCI) # 不降维的版本
    # y_pred_MCI=_svm(X_train_MCI_reduced,y_train_MCI_new,X_test_MCI_reduced,y_test_MCI)
    # print(y_pred,np.shape(y_pred))
    y_pred[y_pred==-1]=y_pred_MCI # 将被分为-1的那些样本的预测结果替换为 y_pred_MCI
    print(y_test,np.shape(y_test))
    print(y_pred,np.shape(y_pred))
    print(y_pred_MCI,np.shape(y_pred_MCI))
    accuracy=accuracy_score(y_test,y_pred)
    return accuracy


print(classifier(X_test))

# def classifier(test_sample): # 对 test_sample做预测
#     # 第一步,将X_train与X_test一起降维 
#     pass
    

# 先通过 LDA 降维到2维, 此时已经将 MCI, MCIn, MCIp 看作一类
# dim=2
# # X_train=dimension_reduction(X_train,y_train_master,dim=dim,method='LDA')
# # visualization(X_train,y_train_master,dim=dim,method='LDA',class_num=3)
# # print(X_train,y_train_master)

# X_valid=dimension_reduction(X_valid,y_valid_master,dim=dim,method='LDA')
# # visualization(X_valid,y_valid_master,dim=dim,method='LDA',class_num=3)

# # X_train=dimension_reduction(X_train,y_train,dim=dim,method='LDA')
# # visualization(X_train,y_train,dim=dim,method='LDA',class_num=3)
# # # print(X_train,y_train_master)

# # X_valid=dimension_reduction(X_valid,y_valid,dim=dim,method='LDA')
# # visualization(X_valid,y_valid,dim=dim,method='LDA',class_num=3)

# X_test=dimension_reduction(X_test,y_test_master,dim=dim,method='LDA')
# # print(X_test)
# visualization(X_test,y_test_master,dim=dim,method='LDA',class_num=3)


# print(_svm(X_train,y_train_master,X_valid,y_valid_master))
# visualization(X_train,y_train_master,dim=dim,method='LDA',class_num=3)

# clf=svm.SVC(kernel='linear',C=1.0,random_state=0,decision_function_shape='ovr')
# clf.fit(X_train,y_train_master)
# # print(clf.coef_,clf.intercept_)
# P_train=clf.coef_
# B_train=clf.intercept_
# # print(P_train,B_train)
# clf.fit(X_valid,y_valid_master)
# P_valid=clf.coef_
# B_valid=clf.intercept_
# # print(P_valid,B_valid)

# P_train=torch.from_numpy(P_train)
# B_train=torch.from_numpy(B_train)
# P_valid=torch.from_numpy(P_valid)
# B_valid=torch.from_numpy(B_valid)

# def loss_function(W, b, P_train, P_valid, B_train, B_valid):
#     W=W.double()
#     b=b.double()
#     # b.reshape(2,1)
#     # print(W.shape,b.shape,P_train.shape,P_valid.shape,B_train.shape,B_valid.shape)
#     b=b.view(2, 1)
#     loss1 = torch.norm(torch.mm(P_train, W) - P_valid, 'fro')
#     loss2 = torch.norm(torch.mm(P_train, b) + B_train - B_valid, 2)
#     return loss1 + loss2

# # # initial_params = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
# # init=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

# # # 使用拟牛顿法进行优化
# # result = minimize(loss_function, init, args=(W,b), method='BFGS')
# # optimized_params = result.x

# # # 从优化后的参数中恢复 W 和 b
# # optimized_W = optimized_params[:4].reshape(2, 2)
# # optimized_b = optimized_params[4:].reshape(2, 1)

# # print("Optimized W:")
# # print(optimized_W)
# # print("Optimized b:")
# # print(optimized_b)

# # 初始化参数
# W = torch.randn((dim, dim), requires_grad=True)
# b = torch.randn((dim,), requires_grad=True)

# # 定义优化器
# optimizer = optim.Adam([W, b], lr=0.8,weight_decay=0.9)
# num_epochs = 1000
# # 迭代优化
# for epoch in range(num_epochs):
#     loss = loss_function(W, b, P_train, P_valid, B_train, B_valid)
#     # print(epoch,loss.detach().numpy())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # 最终的参数值
# optimal_W = W.detach().numpy()
# optimal_b = b.detach().numpy()
# print(optimal_W, optimal_b)
# # print(optimal_W.shape,optimal_b.shape)
# X_train=X_train.dot(optimal_W)+optimal_b[ np.newaxis,:]
# print(_svm(X_train,y_train_master,X_valid,y_valid_master))
# visualization(X_train,y_train_master,dim=dim,method='LDA',class_num=3)

# # class SimpleNet(nn.Module):
# #     def __init__(self,P_train,B_train,P_valid,B_valid):
# #         super(SimpleNet, self).__init__()
# #         self.P_train=P_train
# #         self.B_train=B_train
# #         self.P_valid=P_valid
# #         self.B_valid=B_valid
# #         self.fc = nn.Linear(in_features=dim, out_features=dim)
# #     def forward(self, x):
# #         x=self.fc(x)
# #         return x
# #     def loss_function(self):
# #         loss1 = torch.norm(torch.mm(self.P_train,self.fc.weight)-self.P_valid,'fro')
# #         loss2 = torch.norm(torch.mm(self.P_train, self.fc.bias) + self.B_train - self.B_valid, 2)
# #         return loss1+loss2
# #     def calculate_P_B(self):        
# #         pass



# # print(_svm(X_train,y_train_master,X_valid,y_valid_master))