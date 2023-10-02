import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
# from data_process import X_test,X_train
from data_process import y_test,y_train,Labels
from dimension_deduction import X_train,X_test
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

clf = MultinomialNB()

# 训练分类器
clf.fit(X_train, y_train)

# 使用分类器进行预测
y_pred = clf.predict(X_test)

# 评估分类器的性能
accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", confusion)
# print("Classification Report:\n", classification_rep)
