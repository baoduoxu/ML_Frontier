# # 对数据进行降维
# from data_process import X_test,X_train,y_test,y_train,Labels
# from data_process import X_test_MCI,X_train_MCI,y_test_MCI,y_train_MCI
# import seaborn as sns
# import numpy as np
# import  matplotlib.pyplot as plt
# import scipy.stats as stats
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.manifold import LocallyLinearEmbedding
# from sklearn.preprocessing import StandardScaler


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X=np.concatenate((X_train,X_test),axis=0)
# y=np.concatenate((y_train,y_test),axis=0)

# # 数据归一化
# # print(X_train[:,2])
# # X_train=(X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)
# # X_test=(X_test-np.mean(X_test,axis=0))/np.std(X_test,axis=0)

# # 计算相关系数矩阵画热力图

# # correlation = np.corrcoef(X_train.T) # X_train的形状为(214, 186)
# # print(correlation)

# # plt.figure(figsize=(10, 8))
# # sns.heatmap(correlation, cmap="coolwarm", annot=False)  # 使用不同的颜色映射和关闭标注

# # # 添加标签和标题
# # plt.xlabel("Features")
# # plt.ylabel("Features")
# # plt.title("Correlation Heatmap")

# # # 显示热力图
# # plt.show()

# # Q-Q图 验证是否符合正态分布, 除了极少数特殊的列, 基本符合
# # for i in range(10):
# #     normal_sample = np.random.normal(size=len(X_train))

# #     stats.probplot(X_train[:, i], dist="norm", plot=plt)
# #     plt.title("Q-Q Plot for Sample Dimension 0")
# #     plt.xlabel("Theoretical Quantiles")
# #     plt.ylabel("Sample Quantiles")

# #     # 显示图形
# #     plt.show()

# # 降维

# # pca=PCA(n_components=23)
# # X_train=pca.fit_transform(X_train)
# # X_test=pca.fit_transform(X_test)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# # 创建LDA模型
# lda = LinearDiscriminantAnalysis(n_components=2)  # 你可以设置n_components为降维后的维度数

# # 拟合LDA模型并进行降维
# X_train = lda.fit_transform(X_train, y_train)
# X_test = lda.fit_transform(X_test,y_test)
# X_low_dimension = lda.fit_transform(X,y)
# # plt.plot(X_train, y_train, 'o')
# # plt.xlabel('LDA Component 1')
# # plt.ylabel('Target')
# # plt.title('LDA Dimensionality Reduction to 1D')
# # plt.show()
# # plt.plot(X_test, y_test, 'o')
# # plt.xlabel('LDA Component 1')
# # plt.ylabel('Target')
# # plt.title('LDA Dimensionality Reduction to 1D')
# # plt.show()


# # 绘制降维后的数据
# # 降到2维
# import matplotlib.pyplot as plt

# # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')

# color_mapping = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'm'}  # 颜色映射字典

# # 绘制散点图
# for label, color in color_mapping.items():
#     # 获取当前类别的数据点索引
#     class_indices = (y_train == label)
#     # 绘制当前类别的数据点
#     plt.scatter(X_train[class_indices, 0], X_train[class_indices, 1], c=color, label=f'Class {label}')

#     # 标注出颜色对应的类别
#     plt.text(X_train[class_indices, 0].mean(), X_train[class_indices, 1].mean(), f'Class {label}',
#              color='black', fontsize=12, ha='center', va='center')
# # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow')
# plt.xlabel('LDA Component 1')
# plt.ylabel('LDA Component 2')
# plt.title('LDA Dimensionality Reduction')
# plt.colorbar()
# plt.show()


# plt.scatter(X_low_dimension[:, 0], X_low_dimension[:, 1], c=y, cmap='rainbow')
# plt.xlabel('LDA Component 1')
# plt.ylabel('LDA Component 2')
# plt.title('LDA Dimensionality Reduction')
# plt.colorbar()
# plt.show()
# # # 创建LLE模型
# # lle = LocallyLinearEmbedding(n_neighbors=10, n_components=65, method='standard')

# # # # 使用LLE模型拟合和降维数据
# # X_train = lle.fit_transform(X_train)
# # X_test = lle.fit_transform(X_test)

# # # tsne=TSNE(n_components=3, learning_rate='auto',init='random', perplexity=17,early_exaggeration=15,random_state=0)
# # # X_train=tsne.fit_transform(X_train)
# # # X_test=tsne.fit_transform(X_test)



# # # explained_variance_ratio = pca.explained_variance_ratio_
# # # plt.plot(np.cumsum(explained_variance_ratio))
# # # plt.xlabel('Number of Principal Components')
# # # plt.ylabel('Explained Variance Ratio')
# # # plt.title('Explained Variance Ratio vs. Number of Principal Components')
# # # plt.grid(True)

# # # # # 寻找肘部点
# # # # def find_elbow_point(variance_ratio):
# # # #     for i in range(1, len(variance_ratio)):
# # # #         if variance_ratio[i] - variance_ratio[i - 1] < 0.02:
# # # #             return i

# # # # elbow_point = find_elbow_point(explained_variance_ratio)

# # # # # 在肘部点处绘制垂直线
# # # # plt.axvline(x=elbow_point, color='red', linestyle='--', label=f'Elbow Point ({elbow_point} components)')
# # # # plt.legend()

# # # # # 显示图形
# # # plt.show()

# # # # # 输出选择的维度
# # # # print(f'Selected Number of Principal Components: {elbow_point}')


# # # Laplacian
# # from sklearn.manifold import SpectralEmbedding
# # n_components = 2  # 降维后的维度
# # # embedding = SpectralEmbedding(n_components=n_components, affinity='rbf', gamma=None, random_state=2, eigen_solver=None, n_neighbors=None, n_jobs=None)
# # embedding = SpectralEmbedding(n_components=n_components)
# # # 使用模型进行降维
# # X_reduced = embedding.fit_transform(X_train)

# # # 绘制降维后的数据
# # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, cmap='rainbow')
# # plt.title('Laplacian Eigenmaps')
# # plt.show()

# # # isomap
# # from sklearn.manifold import Isomap
# # n_neighbors = 10  # 用于近邻搜索的邻居数
# # n_components = 2  # 降维后的维度
# # isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)

# # # 使用模型进行降维
# # X_reduced = isomap.fit_transform(X_train)

# # # 绘制降维后的数据
# # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, cmap='rainbow')
# # plt.title('Isomap')
# # plt.show()

