import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig_1 = plt.figure()
fig_2 = plt.figure()
print(fig_2)
# fig.add_subplot(111, projection="3d")

ax = fig_1.gca(projection="3d")
ax_1 = fig_2.gca(projection="3d")
print(ax_1)
# 获取一个子图对象，可以对此进行操作

th = np.linspace(1, 100, 100)
x = 3*th
y = 6*th
z = 1*th

ax_1.plot(x, y, z, label="logx", color="r")
ax_1.set_ylim(-2, 2.0)
ax_1.set_xlim(-2.5, 2.5)
ax_1.set_xlabel("cos(th)")
ax_1.set_ylabel("loge(th*2)")
ax_1.set_zlabel("sin(th)")

ax.plot(y, x*5, z*8, label="logy")
ax.set_ylim(-3, 3.0)
ax.set_xlim(-3.0, 3.0)
ax.set_xlabel("loge(th*2)")
ax.set_ylabel("cos(th)")
# 对对象使用ylim时，要加上set_(这是对象式），不能使用函数式ax.ylim
# 是否引用ax都可以

ax.legend()
ax_1.legend()
plt.show()
print()
