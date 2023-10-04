# 读入.mat文件, 划分测试集与训练集

import scipy.io
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
if len(sys.argv)!=2:
    print('Usage: python data_process.py <dataset_path>')
    sys.exit(1)

file_path=sys.argv[1]
# file_path='dataset/ADNI.mat'
mat_data=scipy.io.loadmat(file_path)
# print(mat_data.keys())
test_ratio=0.3
X_train=[]
X_test=[]
y_train=[]
y_test=[]
Labels=[key for key in mat_data.keys() if not key.startswith('__')]
dict_labels = {label: i for i, label in enumerate(Labels)}
print(dict_labels)
scaler=StandardScaler()

X_train_MCI=[]
X_test_MCI=[]
y_train_MCI=[]
y_test_MCI=[]

for label,data in mat_data.items():
    if label.startswith('__'):
        continue
    N=np.shape(data)[0]
    num_test_samples=int(N*test_ratio)
    test_index=np.random.choice(N,num_test_samples,replace=False)

    for i in range(N):
        if i in test_index:
            # X_test.append(scaler.fit_transform(data[i].T))
            X_test.append(data[i].T)
            y_test.append(label)
            if dict_labels[label]>=1 and dict_labels[label]<=3:
                X_test_MCI.append(data[i])
                y_test_MCI.append(label)
        else:
            # X_train.append(scaler.fit_transform(data[i].T))
            X_train.append(data[i].T)
            y_train.append(label)
            if dict_labels[label]>=1 and dict_labels[label]<=3:
                X_train_MCI.append(data[i])
                y_train_MCI.append(label)




dict_labels_MCI = {label: i for i, label in enumerate(Labels[1:])}
X_train = np.array(X_train)
y_train=np.array([dict_labels[label] for label in y_train])
X_test = np.array(X_test)
y_test=np.array([dict_labels[label] for label in y_test])

X_train_MCI = np.array(X_train_MCI)
y_train_MCI=np.array([dict_labels_MCI[label] for label in y_train_MCI])
X_test_MCI = np.array(X_test_MCI)
y_test_MCI=np.array([dict_labels_MCI[label] for label in y_test_MCI])


def normalize(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    return X

X_train=normalize(X_train)
X_test=normalize(X_test)
X_train_MCI=normalize(X_train_MCI)
X_test_MCI=normalize(X_test_MCI)
