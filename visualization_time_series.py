# 可视化时间序列
from data_process import X_test, X_train
import numpy as np
import matplotlib.pyplot as plt
# print(np.shape(X_train))

# 创建横坐标
x_axis = range(1, 121)

# 创建一个空白的图像
plt.figure()

# 循环绘制十个折线图并叠加显示
for i in range(89):
    data_to_plot = X_train[1, :, i]

    # 强制y值在-100和100之间
    # data_to_plot = np.clip(data_to_plot, -100, 100)
    plt.plot(x_axis, data_to_plot, label=f'Sample {i + 1}, Column 2')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Line Plots of Sample Data, Column 2')
plt.legend()
plt.grid(True)

# 显示整个图像
plt.show()
