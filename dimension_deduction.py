# 对数据进行降维
from data_process import X_test,X_train,y_test,y_train,Labels
import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 数据归一化
# print(X_train[:,2])
# X_train=(X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)
# X_test=(X_test-np.mean(X_test,axis=0))/np.std(X_test,axis=0)

# 计算相关系数矩阵画热力图

# correlation = np.corrcoef(X_train.T) # X_train的形状为(214, 186)
# print(correlation)

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation, cmap="coolwarm", annot=False)  # 使用不同的颜色映射和关闭标注

# # 添加标签和标题
# plt.xlabel("Features")
# plt.ylabel("Features")
# plt.title("Correlation Heatmap")

# # 显示热力图
# plt.show()

# Q-Q图 验证是否符合正态分布, 除了极少数特殊的列, 基本符合
# for i in range(10):
#     normal_sample = np.random.normal(size=len(X_train))

#     stats.probplot(X_train[:, i], dist="norm", plot=plt)
#     plt.title("Q-Q Plot for Sample Dimension 0")
#     plt.xlabel("Theoretical Quantiles")
#     plt.ylabel("Sample Quantiles")

#     # 显示图形
#     plt.show()

# 降维

pca=PCA(n_components=23)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)



# 创建LLE模型
# lle = LocallyLinearEmbedding(n_neighbors=7, n_components=20, method='standard')

# # 使用LLE模型拟合和降维数据
# X_train = lle.fit_transform(X_train)
# X_test = lle.fit_transform(X_test)

# tsne=TSNE(n_components=3, learning_rate='auto',init='random', perplexity=17,early_exaggeration=15,random_state=0)
# X_train=tsne.fit_transform(X_train)
# X_test=tsne.fit_transform(X_test)



# explained_variance_ratio = pca.explained_variance_ratio_
# plt.plot(np.cumsum(explained_variance_ratio))
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance Ratio vs. Number of Principal Components')
# plt.grid(True)

# # # 寻找肘部点
# # def find_elbow_point(variance_ratio):
# #     for i in range(1, len(variance_ratio)):
# #         if variance_ratio[i] - variance_ratio[i - 1] < 0.02:
# #             return i

# # elbow_point = find_elbow_point(explained_variance_ratio)

# # # 在肘部点处绘制垂直线
# # plt.axvline(x=elbow_point, color='red', linestyle='--', label=f'Elbow Point ({elbow_point} components)')
# # plt.legend()

# # # 显示图形
# plt.show()

# # # 输出选择的维度
# # print(f'Selected Number of Principal Components: {elbow_point}')