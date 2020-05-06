import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# x = np.linspace(0, 10, 1000)
# y = np.sin(x)*x
# z = np.cos(x)
#
# # GUI显示框的大小
# plt.figure(figsize=(8, 4))
# # 传入参数,定义图像的名称,b--代表蓝色，虚线
# plt.plot(x, y, label="sin(x)*x")  # 此处x,y传入是有顺序的，改变可能导致图像因xy轴不均匀而变化
# plt.plot(x, z, "r--", label="cos(x)")
# # 定义各个轴的名称，以及图表名称
# plt.xlabel("x axle")
# plt.ylabel("y axle")
# plt.title("my first matplotlib")
# # 限制y轴的范围
# plt.ylim(-5.0, 9.0)
# # 使得图例显示
# plt.legend()  # 用于显示 标示每条线所代表意义的小方框，就是图例
# plt.show()
#
# fig = plt.gcf()
# axes = plt.gca()
# print(fig, axes)
# # (get current axes/figures)分别获取当前axes和figures的对象
ax = plt.figure(figsize=(10, 10)).gca(projection="3d")
x = np.arange(-1, 1, 0.001)
y = np.arange(-1, 1, 0.001)
x, y = np.meshgrid(x, y)
z = x * y / (x + y)
ax.plot_surface(x, y, z, cmap='rainbow')
plt.show()
