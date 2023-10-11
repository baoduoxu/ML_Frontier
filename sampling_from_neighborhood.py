import numpy as np
import matplotlib.pyplot as plt
import sys,os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
from data_process import X_test,X_train,X_valid,y_test_master
from data_process import y_test,y_train,y_train_master,y_valid_master,y_valid
from data_process import X_valid_MCI,X_train_MCI,y_valid_MCI,y_train_MCI
from data_process import normalize
# from dimension_reduction import X_train,X_valid,X_test
from dimension_reduction_lda import dimension_reduction,visualization

def sampling_from_neighborhood(x, delta, N, proj_mat,X_train_new):
    """
    Given a 2D data x, a smaller neighborhood radius delta, and N samples, 
    sample N points from the circle centered at x with radius delta.
    """
    # Generate N random angles between 0 and 2*pi
    angles = np.random.uniform(0, 2*np.pi, N)
    
    # Generate N random radii between 0 and delta
    radii = np.random.uniform(0, delta, N)
    
    # Convert polar coordinates to cartesian coordinates
    x_neighbor = np.zeros((N, 2))
    x_neighbor[:, 0] = x[0] + radii * np.cos(angles)
    x_neighbor[:, 1] = x[1] + radii * np.sin(angles)
    print(x,x_neighbor)
    # Project x and x_neighbor onto higher dimensional space
    print(x.shape,proj_mat.shape)
    print(x,np.dot(X_train_new[0],proj_mat))
    x_proj = np.dot(x,np.linalg.pinv(proj_mat))
    x_neighbor_proj = np.dot(x_neighbor,np.linalg.pinv(proj_mat))
    print(np.dot(np.linalg.pinv(proj_mat),proj_mat))
    print(np.dot(proj_mat,np.linalg.pinv(proj_mat)))
    # print(f'{sys.getsizeof(proj_mat[0][0])} bytes,{format(proj_mat[0][0],".300f")}')

    # x_neighbor_proj = x_neighbor_proj
    # Plot the points in the higher dimensional space
    plt.plot(range(186), X_train_new[0], label=f'Original Sample')
    plt.plot(range(186), x_proj, label=f'Original Sample after DR and DA')
    # for i in range(N):
    #     plt.plot(range(186), x_neighbor_proj[i, :], label=f'Sample {i + 1}')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title('Line Plots of Rows in X')

    # 显示折线图
    plt.show()
    
    return x_neighbor

if __name__=='__main__':
    X_train_new=np.concatenate((X_train,X_valid),axis=0)
    X_train_new=normalize(X_train_new)
    y_train_master_new=np.concatenate((y_train_master,y_valid_master),axis=0) # 无需验证集, 将train和validation合并
    # print(X_train_new[0])
    X_train_reduced,proj_mat=dimension_reduction(X_train_new,y_train_master_new,dim=2,method='LDA')

    X_train = X_train_reduced
    y_train_master = y_train_master_new
    # print(X_train)
    sampling_from_neighborhood(X_train[0],1e-6,1,proj_mat,X_train_new)