import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path

from data_process import X_test,X_train,y_test,y_train,Labels
from sktime.datasets import load_basic_motions
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based._arsenal import Arsenal
from sktime.classification.kernel_based._rocket_classifier import RocketClassifier
from sktime.classification.kernel_based._svc import TimeSeriesSVC
# from sktime.classification.kernel_based._rocket_classifier import ROCKETClassifier
# 创建ShapeletTransformClassifier分类器
# clf = HIVECOTEV2()
clf=RocketClassifier()

# 拟合模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算分类准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
