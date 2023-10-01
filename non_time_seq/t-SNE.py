import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
from data_process import X_test,X_train,y_test,y_train,Labels
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X=np.concatenate((X_train,X_test),axis=0)
y=np.concatenate((y_train,y_test),axis=0)

tsne=TSNE(n_components=2, learning_rate='auto',init='random', perplexity=20,early_exaggeration=15,random_state=0)
low_dim_embs=tsne.fit_transform(X)


plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], c=y)  # Labels是数据点的标签
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # 创建一个三维图形

# # 绘制散点图，用不同颜色标记不同类别的数据点
# scatter=ax.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], low_dim_embs[:, 2], c=y)
# # 设置图例
# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)

# ax.set_xlabel('t-SNE Dimension 1')
# ax.set_ylabel('t-SNE Dimension 2')
# ax.set_zlabel('t-SNE Dimension 3')
# ax.set_title('t-SNE 3D Visualization')
# plt.show()

