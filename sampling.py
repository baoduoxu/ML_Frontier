import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde,multivariate_normal
import sys,os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
from data_process import X_test,X_train,X_valid,y_test_master
from data_process import y_test,y_train,y_train_master,y_valid_master,y_valid
from data_process import X_valid_MCI,X_train_MCI,y_valid_MCI,y_train_MCI
from data_process import normalize
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

def sampling_from_neighborhood(x, delta, N, proj_mat):
    """
    Given a 2D data x, a smaller neighborhood radius delta, and N samples, 
    sample N points from the circle centered at x with radius delta.
    """
    # Generate N random angles between 0 and 2*pi
    angles = np.random.uniform(0, 2*np.pi, N)
    
    # Generate N random radii between 0 and delta
    radii = np.random.uniform(0, delta, N)
    
    # Convert polar coordinates to cartesian coordinates
    x_neighbor = np.zeros((N, 2))
    x_neighbor[:, 0] = x[0] + radii * np.cos(angles)
    x_neighbor[:, 1] = x[1] + radii * np.sin(angles)
    
    # Project x and x_neighbor onto higher dimensional space
    x_proj = np.dot(proj_mat, x)
    x_neighbor_proj = np.dot(proj_mat, x_neighbor.T).T
    
    # Plot the points in the higher dimensional space
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_proj[0], x_proj[1], 'ro', label='x')
    ax.plot(x_neighbor_proj[:, 0], x_neighbor_proj[:, 1], 'bo', label='x_neighbor')
    ax.legend()
    plt.show()
    
    return x_neighbor

X_train_new=np.concatenate((X_train,X_valid),axis=0)
y_train_master_new=np.concatenate((y_train_master,y_valid_master),axis=0) # 无需验证集, 将train和validation合并

X_train_reduced,proj_mat=dimension_reduction(X_train_new,y_train_master_new,dim=2,method='LDA')

# Given X_train and y_train_master, three classes of data with labels -1, 0, 4
X_train = X_train_reduced
y_train_master = y_train_master_new


class_means = []
class_covs = []

for class_label in [-1, 0, 4]:
    # 选择特定类别的数据
    class_data = X_train[y_train_master == class_label]

    # 估计均值和协方差矩阵
    class_mean = np.mean(class_data, axis=0)
    class_cov = np.cov(class_data, rowvar=False)

    class_means.append(class_mean)
    class_covs.append(class_cov)

print(class_means, class_covs)

# 设置随机数种子以获得可重复的采样结果
np.random.seed(1881)

# 从估计的正态分布中进行随机采样
num_samples = 50
sampled_data = []

for i in range(3):
    class_samples = multivariate_normal.rvs(mean=class_means[i], cov=class_covs[i], size=num_samples)
    sampled_data.append(class_samples)

# 将采样数据合并为一个数组
# visualization(X_train,y_train_master,dim=2,method='LDA',class_num=3)
X_sampling = np.concatenate(sampled_data, axis=0)
y_sampling = np.repeat([-1, 0, 4], num_samples, axis=0)
# visualization(X_sampling,y_sampling,dim=2,method='LDA',class_num=3)

# X_sampling=np.concatenate((X_sampling,X_train),axis=0)
y_sampling=np.concatenate((y_sampling,y_train_master),axis=0)

# visualization(X_sampling,y_sampling,dim=2,method='LDA',class_num=3)

X_sampling=np.dot(X_sampling,proj_mat.T)

print(X_sampling.shape,y_sampling.shape)
X_sampling=np.concatenate((X_sampling,X_train_new),axis=0)
# X_sampling=normalize(X_sampling)

X_sampling=np.concatenate((X_sampling,X_test),axis=0)

X=dimension_reduction(X_sampling,y_sampling,dim=2,method='t-SNE')
visualization(X,np.array([2]*(3*num_samples)+[4]*X_train_new.shape[0]+[-1]*X_test.shape[0]),dim=2,method='PCA',class_num=3)
print(X.shape,X_test.shape[0])
print(X[-X_test.shape[0]:].shape)
visualization(X[-X_test.shape[0]:],y_test_master,dim=2,method='PCA',class_num=3)

# X_sampling 和 y_sampling 现在包含了合并后的采样数据和标签
# # Split the data into three classes
# data1 = X_train[y_train_master == -1]
# data2 = X_train[y_train_master == 0]
# data3 = X_train[y_train_master == 4]

# # Fit a normal distribution to each class of data
# pdf1 = multivariate_normal(loc=np.mean(data1,axis=0), scale=np.cov(data1,rowvar=False))
# pdf2 = multivariate_normal(loc=np.mean(data2,axis=0), scale=np.cov(data2,rowvar=False))
# pdf3 = multivariate_normal(loc=np.mean(data3,axis=0), scale=np.cov(data3,rowvar=False))

# # Generate 10000 random samples from each distribution
# samples1 = pdf1.rvs(size=10000)
# samples2 = pdf2.rvs(size=10000)
# samples3 = pdf3.rvs(size=10000)

# print(X_train.shape,samples1.shape, samples2.shape, samples3.shape)

# # Combine the samples with the original data
# X_sampling = np.concatenate((X_train, samples1, samples2, samples3))
# y_sampling = np.concatenate((y_train_master, np.full(10000, -1), np.full(10000, 0), np.full(10000, 4)))



