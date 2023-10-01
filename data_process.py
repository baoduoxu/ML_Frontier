# 读入.mat文件, 划分测试集与训练集

import scipy.io
import numpy as np
import sys

if len(sys.argv)!=2:
    print('Usage: python data_process.py <dataset_path>')
    sys.exit(1)

file_path=sys.argv[1]

mat_data=scipy.io.loadmat(file_path)

test_ratio=0.3
X_train=[]
X_test=[]
y_train=[]
y_test=[]
Labels=[]
for label,data in mat_data.items():
    if label.startswith('__'):
        continue
    # print(label)
    # print(np.shape(data))
    Labels.append(label)
    N=np.shape(data)[0]
    num_test_samples=int(N*test_ratio)
    test_index=np.random.choice(N,num_test_samples,replace=False)

    for i in range(N):
        if i in test_index:
            X_test.append(data[i].T)
            y_test.append(label)
        else:
            X_train.append(data[i].T)
            y_train.append(label)

dict_labels = {label: i for i, label in enumerate(Labels)}

X_train = np.array(X_train)
y_train=np.array([dict_labels[label] for label in y_train])
X_test = np.array(X_test)
y_test=np.array([dict_labels[label] for label in y_test])