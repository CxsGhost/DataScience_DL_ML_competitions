import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 10
h = 1.30933114
k = 0.5
def curve(x):
    yz1 = np.power(np.sqrt(np.power(n, 2) + np.power(h, 2)) -
                   np.sqrt(np.power(n, 2) + np.power(h, 2)) /
                   n * x, 1 + k) / \
          (2 * np.power(np.sqrt(np.square(h) + np.square(n)), k) * (1 + k))
    yz2 = np.power(np.sqrt(np.square(h) + np.square(n)), k) * \
          np.power(np.sqrt(np.square(h) + np.square(n)) -
                   np.sqrt(np.square(h) + np.square(n)) / n * x, 1 - k) / \
          (2 * (1 - k))
    yz3 = np.sqrt(np.square(h) + np.square(n)) / 2 * (1 / (1 - k) - 1 / (1 + k))

    y = yz1 - yz2 + yz3
    return y

x = np.arange(0, 10.1, 0.2)
y = curve(x)
z = h / n * x
plt.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure(figsize=(10, 8))
ax = Axes3D(fig)
ax.elev = 15
ax.azim = 140
ax.plot(x, y, z, marker='o', label='导弹')
x = [n for i in range(10)]
x = np.array(x)
y = np.linspace(0, y[-1] + 0.1, 10)
z = [h for j in range(10)]
print(x, y, z)
ax.plot(x, y, z, marker='*', color='r', label='敌机')
ax.set_xlabel('距离', fontdict={'size': 15})
ax.set_ylabel('距离', fontdict={'size': 15})
ax.set_zlabel('高度', fontdict={'size': 15})
ax.set_title('轨迹', fontdict={'size': 20})
plt.legend(fontsize='xx-large')
plt.savefig('轨迹.png')
plt.show()
















