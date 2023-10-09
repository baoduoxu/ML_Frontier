import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
# from data_process import X_test,X_train
from data_process import y_test,y_train,Labels
from data_process import X_test_MCI,X_train_MCI,y_test_MCI,y_train_MCI
# from dimension_reduction import X_train,X_test
from dimension_reduction_lda import dimension_reduction,visualization
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
dim=2
if __name__=='__main__':
    # X_train_MCI=dimension_reduction(X_train_MCI,y_train_MCI,dim=2,method='LDA')
    # X_test_MCI=dimension_reduction(X_test_MCI,y_test_MCI,dim=2,method='LDA')
    
    X_train_reduced=dimension_reduction(X_train_MCI,y_train_MCI,dim=110,method='PCA')
    X_train_reduced=dimension_reduction(X_train_reduced,y_train_MCI,dim=60,method='Laplacian')
    X_train_reduced=dimension_reduction(X_train_reduced,y_train_MCI,dim=dim,method='LDA')

    visualization(X_train_reduced,y_train_MCI,dim=dim,method='LDA',class_num=3)

    # 测试集
    # X_train_reduced=dimension_reduction(X_train_MCI,y_train_MCI,dim=110,method='PCA')
    # X_train_reduced=dimension_reduction(X_train_reduced,y_train_MCI,dim=60,method='Laplacian')
    # X_train_reduced=dimension_reduction(X_train_reduced,y_train_MCI,dim=dim,method='LDA')

    # visualization(X_train_reduced,y_train_MCI,dim=dim,method='LDA',class_num=3)

    # 训练集
    X_test_reduced=dimension_reduction(X_test_MCI,y_test_MCI,dim=57,method='PCA')
    X_test_reduced=dimension_reduction(X_test_reduced,y_test_MCI,dim=30,method='Laplacian')
    X_test_reduced=dimension_reduction(X_test_reduced,y_test_MCI,dim=dim,method='LDA')

    def _svm(X_train,y_train,X_test,y_test):
        clf=svm.SVC(kernel='linear',C=1.0,random_state=0,decision_function_shape='ovr')
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(clf.coef_,clf.intercept_)
        return accuracy

    # print(_svm(X_train_MCI,y_train_MCI,X_test_MCI,y_test_MCI))
    print(_svm(X_train_reduced,y_train_MCI,X_test_reduced,y_test_MCI))