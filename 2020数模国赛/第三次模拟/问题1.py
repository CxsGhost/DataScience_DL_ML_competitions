import numpy as np
from numpy import sqrt, dot, cross
from numpy.linalg import norm, inv, pinv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 求三个球体交点
def intersection(P1, P2, P3, r1, r2, r3):
    temp1 = P2 - P1
    e_x = temp1 / norm(temp1)
    temp2 = P3 - P1
    i = dot(e_x, temp2)
    temp3 = temp2 - i * e_x
    e_y = temp3 / norm(temp3)
    e_z = cross(e_x, e_y)
    d = norm(P2 - P1)
    j = dot(e_y, temp2)
    x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    y = (r1 * r1 - r3 * r3 - 2 * i * x + i * i + j * j) / (2 * j)
    temp4 = r1 * r1 - x * x - y * y
    if temp4 < 0:
        raise Exception("次三球体无交点！")
    z = sqrt(temp4)
    p_12_a = P1 + x * e_x + y * e_y + z * e_z
    p_12_b = P1 + x * e_x + y * e_y - z * e_z
    if p_12_a[2] > 0:
        return p_12_a
    else:
        return p_12_b

# 雷达坐标，距离
p1 = np.array([6650, 3030, 0])
p2 = np.array([8020, 5430, 0])
p3 = np.array([8705, 2230, 0])

r1_ = 14177
r2_ = 14450
r3_ = 13541

plane_intersect = intersection(p1, p2, p3, r1_, r2_, r3_)
print('飞机位置为：{}'.format(plane_intersect))

# 构造计算所需的矩阵
mat = np.array([(plane_intersect - p1) / r1_,
                (plane_intersect - p2) / r2_,
                (plane_intersect - p3) / r3_])

try:
    mat_inv = inv(mat)
except:
    mat_inv = pinv(mat)

mat_inv_trans = np.transpose(mat_inv, axes=(1, 0))

mat_mid = np.zeros(shape=(3, 3))
x = []
y = []
z = []
for i in range(1, 51):
    for j in range(1, 51):
        mat_mid[0, 0] = i + j
        mat_mid[1, 1] = i + j
        mat_mid[2, 2] = j

        x.append(i)
        y.append(j)
        z.append(np.dot(np.dot(mat_inv, mat_mid), mat_inv_trans).trace())

# 绘图
x, y = np.meshgrid(range(1, 51), range(1, 51))
z = np.reshape(z, newshape=(len(x), len(y)))
plt.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure(figsize=(10, 8))
ax = fig.gca(projection='3d')
ax.elev = 10
ax.azim = -50
# ax = Axes3D(fig)
one = ax.plot_surface(y, x, z, cmap='rainbow')

# 缩小
small_p1 = p1.copy()
small_p1[0] *= 1.1
small_p2 = p2.copy()
small_p2[1] *= 0.9
small_p3 = p3.copy()
small_p3[0] -= 500
small_p3[1] -= 500
r1 = np.sqrt(np.square(plane_intersect[0] - small_p1[0]) +
             np.square(plane_intersect[1] - small_p1[1]) +
             np.square(plane_intersect[2]))
r2 = np.sqrt(np.square(plane_intersect[0] - small_p2[0]) +
             np.square(plane_intersect[1] - small_p2[1]) +
             np.square(plane_intersect[2]))
r3 = np.sqrt(np.square(plane_intersect[0] - small_p3[0]) +
             np.square(plane_intersect[1] - small_p3[1]) +
             np.square(plane_intersect[2]))

mat = np.array([(plane_intersect - small_p1) / r1_,
                (plane_intersect - small_p2) / r2_,
                (plane_intersect - small_p3) / r3_])

try:
    mat_inv = inv(mat)
except:
    mat_inv = pinv(mat)

mat_inv_trans = np.transpose(mat_inv, axes=(1, 0))

mat_mid = np.zeros(shape=(3, 3))
x = []
y = []
z = []
for i in range(1, 51):
    for j in range(1, 51):
        mat_mid[0, 0] = i + j
        mat_mid[1, 1] = i + j
        mat_mid[2, 2] = j

        x.append(i)
        y.append(j)
        z.append(np.dot(np.dot(mat_inv, mat_mid), mat_inv_trans).trace())

x, y = np.meshgrid(range(1, 51), range(1, 51))
z = np.reshape(z, newshape=(len(x), len(y)))
two = ax.plot_surface(y, x, z, cmap='winter')

# 极端
regular_p1 = np.array([6650, 3030, 0])
regular_p2 = np.array([7335, 0, 0])
regular_p3 = np.array([7335, 5430, 0])

r1 = np.sqrt(np.square(plane_intersect[0] - regular_p1[0]) +
             np.square(plane_intersect[1] - regular_p1[1]) +
             np.square(plane_intersect[2]))
r2 = np.sqrt(np.square(plane_intersect[0] - regular_p2[0]) +
             np.square(plane_intersect[1] - regular_p2[1]) +
             np.square(plane_intersect[2]))
r3 = np.sqrt(np.square(plane_intersect[0] - regular_p3[0]) +
             np.square(plane_intersect[1] - regular_p3[1]) +
             np.square(plane_intersect[2]))

mat = np.array([(plane_intersect - regular_p1) / r1_,
                (plane_intersect - regular_p2) / r2_,
                (plane_intersect - regular_p3) / r3_])

try:
    mat_inv = inv(mat)
except:
    mat_inv = pinv(mat)

mat_inv_trans = np.transpose(mat_inv, axes=(1, 0))

mat_mid = np.zeros(shape=(3, 3))
x = []
y = []
z = []
for i in range(1, 51):
    for j in range(1, 51):
        mat_mid[0, 0] = i + j
        mat_mid[1, 1] = i + j
        mat_mid[2, 2] = j

        x.append(i)
        y.append(j)
        z.append(np.dot(np.dot(mat_inv, mat_mid), mat_inv_trans).trace())

x, y = np.meshgrid(range(1, 51), range(1, 51))
z = np.reshape(z, newshape=(len(x), len(y)))
three = ax.plot_surface(y, x, z, cmap='hot')


# 放大
big_p1 = p1.copy()
big_p1[0] *= 0.9
big_p2 = p2.copy()
big_p2[1] *= 1.1
big_p3 = p3.copy()
big_p3[0] += 500
big_p3[1] += 500

r1 = np.sqrt(np.square(plane_intersect[0] - big_p1[0]) +
             np.square(plane_intersect[1] - big_p1[1]) +
             np.square(plane_intersect[2]))
r2 = np.sqrt(np.square(plane_intersect[0] - big_p2[0]) +
             np.square(plane_intersect[1] - big_p2[1]) +
             np.square(plane_intersect[2]))
r3 = np.sqrt(np.square(plane_intersect[0] - big_p3[0]) +
             np.square(plane_intersect[1] - big_p3[1]) +
             np.square(plane_intersect[2]))

mat = np.array([(plane_intersect - big_p1) / r1_,
                (plane_intersect - big_p2) / r2_,
                (plane_intersect - big_p3) / r3_])

try:
    mat_inv = inv(mat)
except:
    mat_inv = pinv(mat)

mat_inv_trans = np.transpose(mat_inv, axes=(1, 0))

mat_mid = np.zeros(shape=(3, 3))
x = []
y = []
z = []
for i in range(1, 51):
    for j in range(1, 51):
        mat_mid[0, 0] = i + j
        mat_mid[1, 1] = i + j
        mat_mid[2, 2] = j

        x.append(i)
        y.append(j)
        z.append(np.dot(np.dot(mat_inv, mat_mid), mat_inv_trans).trace())

x, y = np.meshgrid(range(1, 51), range(1, 51))
z = np.reshape(z, newshape=(len(x), len(y)))
four = ax.plot_surface(y, x, z, cmap='cool')

ax.set_title('误差影响分布', fontdict={'size': 20})
ax.set_xlabel('距离误差σ方', fontdict={'size': 15})
ax.set_ylabel('坐标误差σ方', fontdict={'size': 15})
ax.set_zlabel('方差的和', fontdict={'size': 15})
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_zticks([0, 10000, 20000, 30000, 40000, 50000, 60000])
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('方差和分布.png')
plt.show()





