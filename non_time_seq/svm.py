import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
from data_process import X_test,X_train,y_test,y_train,Labels
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

clf=svm.SVC(kernel='rbf',C=1.0,random_state=0,decision_function_shape='ovr')

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)