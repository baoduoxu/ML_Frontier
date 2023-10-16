import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
from data_process import X_test,X_train
from data_process import y_test,y_train,Labels
from data_process import X_test_MCI,X_train_MCI,y_test_MCI,y_train_MCI
# from dimension_reduction import X_train,X_test
from dimension_reduction_lda import dimension_reduction,visualization
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
dim=2
def _svm(X_train,y_train,X_test,y_test):
    clf=svm.SVC(kernel='linear',C=1.0,random_state=0,decision_function_shape='ovr')
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    # print(y_pred)
    accuracy=accuracy_score(y_test,y_pred)
    print(clf.coef_,clf.intercept_)
    return accuracy

if __name__=='__main__':
    print('降维前的svm的表现:')
    print(_svm(X_train,y_train,X_test,y_test))
    print('降维后svm的表现:')
    X_train_reduced=dimension_reduction(X_train,y_train,dim=2,method='PCA')
    # X_train_reduced=dimension_reduction(X_train_reduced,y_train,dim=30,method='Laplacian')
    X_train_reduced,proj_mat=dimension_reduction(X_train_reduced,y_train,dim=1,method='LDA')
    # visualization(X_train_reduced,y_train,dim=dim,method='PCA',class_num=2)
    X_test_reduced=dimension_reduction(X_test,y_test,dim=2,method='Isomap')
    # X_test_reduced=dimension_reduction(X_test_reduced,y_test,dim=30,method='Laplacian')
    # X_test_reduced=np.dot(X_test_reduced,proj_mat)
    visualization(X_test_reduced,y_test,dim=dim,method='PCA',class_num=2)
    print(_svm(X_train_reduced,y_train,X_test_reduced,y_test))

# 150,50/90,50,68.8