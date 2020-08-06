import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig_2 = plt.figure()

print(fig_2)
# fig.add_subplot(111, projection="3d")

ax_1 = mplot3d.Axes3D(fig_2)
print(ax_1)
# 获取一个子图对象，可以对此进行操作

th = np.arange(1, 10, 1)
x = 3*th
y = 6*th
z = 1*th

ax_1.plot(x, y, z, label="logx", colormap='rainbow')

ax_1.set_xlabel("cos(th)")
ax_1.set_ylabel("loge(th*2)")
ax_1.set_zlabel("sin(th)")


# 对对象使用ylim时，要加上set_(这是对象式），不能使用函数式ax.ylim
# 是否引用ax都可以
plt.savefig('11.png',bbox_inches='tight')

ax_1.legend()
plt.show()
print()
