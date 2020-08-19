import scipy as sp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']

# 读取附件2中的数据
question_1_data = pd.read_excel('data.xlsx', sheet_name='附件2')
question_1_data = question_1_data.values


# 建立各个层的模型，用于后续计算
class Layer:
    def __init__(self, p, c, k, thickness=None):
        self.p = p
        self.c = c
        self.k = k
        self.thickness = thickness


# 赋值密度，比热，热传导率，厚度
layer_1 = Layer(300, 1377, 0.082, thickness=0.6)
layer_2 = Layer(862, 2100, 0.37)
layer_3 = Layer(74.2, 1726, 0.045, thickness=3.6)
layer_4 = Layer(1.18, 1005, 0.028)


# 微分方程组求解函数，通用于问题1,2,3的计算
def calculate(environment_temp=None, time=None,
              layer2_thick=None, layer4_thick=None,
              ke=None, ks=None):

    # 环境温度，皮肤温度
    ue = environment_temp
    us = 37.0

    # 赋值第II层，第IV层厚度
    layer_2.thickness = layer2_thick
    layer_4.thickness = layer4_thick

    # 对各层做分割
    div_1 = np.int(layer_1.thickness * 10)
    div_2 = np.int(layer_2.thickness * 10)
    div_3 = np.int(layer_3.thickness * 10)
    div_4 = np.int(layer_4.thickness * 10)

    delta_x1 = layer_1.thickness / div_1 / 1e3
    delta_x2 = layer_2.thickness / div_2 / 1e3
    delta_x3 = layer_3.thickness / div_3 / 1e3
    delta_x4 = layer_4.thickness / div_4 / 1e3

    # 时间步长
    delta_time = 1

    # 总时长
    total_time = time * 60

    # 各层热传导方程中的常量
    lamb_1 = (delta_time * layer_1.k) / \
             (layer_1.c * layer_1.p * np.square(delta_x1))
    lamb_2 = (delta_time * layer_2.k) / \
             (layer_2.c * layer_2.p * np.square(delta_x2))
    lamb_3 = (delta_time * layer_3.k) / \
             (layer_3.c * layer_3.p * np.square(delta_x3))
    lamb_4 = (delta_time * layer_4.k) / \
             (layer_4.c * layer_4.p * np.square(delta_x4))

    # 热对流边界方程中的常量
    miu_e = (ke * delta_x1) / layer_1.k
    miu_s = (ks * delta_x4) / layer_4.k
    miu_e_ue = miu_e * ue
    miu_s_us = miu_s * us

    # 三个交界面方程中的常量
    v_1 = layer_1.k / delta_x1
    v_2 = layer_2.k / delta_x2
    v_3 = layer_3.k / delta_x3
    v_4 = layer_4.k / delta_x4

    # 构造三对角稀疏矩阵
    dim = div_1 + div_2 + div_3 + div_4 + 1
    matrix_a = np.zeros(shape=(dim, dim), dtype='float32')
    matrix_a[0, 0] = 1 + miu_e
    matrix_a[0, 1] = -1
    matrix_a[-1, -1] = 1 + miu_s
    matrix_a[-1, -2] = -1

    matrix_a[div_1, div_1] = v_1 + v_2
    matrix_a[div_1, div_1 - 1] = -v_1
    matrix_a[div_1, div_1 + 1] = -v_2

    pos_1 = div_1 + div_2
    matrix_a[pos_1, pos_1] = v_2 + v_3
    matrix_a[pos_1, pos_1 - 1] = -v_2
    matrix_a[pos_1, pos_1 + 1] = -v_3

    pos_2 = pos_1 + div_3
    matrix_a[pos_2, pos_2] = v_3 + v_4
    matrix_a[pos_2, pos_2 - 1] = -v_3
    matrix_a[pos_2, pos_2 + 1] = -v_4

    for i in range(1, div_1):
        matrix_a[i, i] = 1 + 2 * lamb_1
        matrix_a[i, i - 1] = -lamb_1
        matrix_a[i, i + 1] = -lamb_1
    for i in range(1 + div_1, pos_1):
        matrix_a[i, i] = 1 + 2 * lamb_2
        matrix_a[i, i - 1] = -lamb_2
        matrix_a[i, i + 1] = -lamb_2
    for i in range(1 + pos_1, pos_2):
        matrix_a[i, i] = 1 + 2 * lamb_3
        matrix_a[i, i - 1] = -lamb_3
        matrix_a[i, i + 1] = -lamb_3
    for i in range(1 + pos_2, dim - 1):
        matrix_a[i, i] = 1 + 2 * lamb_4
        matrix_a[i, i - 1] = -lamb_4
        matrix_a[i, i + 1] = -lamb_4

    # 用于存储结果
    result = np.array([[37.0 for _ in range(dim)]], dtype='float32')

    # 求解三对角线性方程组
    l, u = 1, 1
    matrix_ab = np.zeros(shape=(l + u + 1, dim), dtype='float32')
    matrix_b = np.zeros(shape=(dim, ), dtype='float32')
    matrix_b[0] = miu_e_ue
    matrix_b[-1] = miu_s_us
    for i in range(dim):
        for j in range(dim):
            if matrix_a[i, j] != 0:
                matrix_ab[u + i - j, j] = matrix_a[i, j]
    for t in range(1, total_time + 1):
        matrix_b[1: div_1] = result[t - 1][1: div_1]
        matrix_b[1 + div_1: pos_1] = result[t - 1][1 + div_1: pos_1]
        matrix_b[1 + pos_1: pos_2] = result[t - 1][1 + pos_1: pos_2]
        matrix_b[1 + pos_2: dim - 1] = result[t - 1][1 + pos_2: dim - 1]
        x = sp.linalg.solve_banded((l, u), matrix_ab, matrix_b)
        result = np.row_stack((result, x))
    return result


# 问题1求解函数
def problem_1():

    # 正确数据
    real_temp = question_1_data[:, -1]

    # 记录参数搜索过程
    list_ke = []
    list_ks = []
    list_error_mean = []
    list_error_square = []

    # 设置搜索范围，步长
    start = 1
    stop = 200
    step = 2

    # 大步长搜索确定大致范围
    print('正在大范围粗略搜索参数....')
    for ke in np.arange(start, stop, step):
        for ks in np.arange(start, stop, step):
            result = calculate(environment_temp=75.0, time=90,
                               layer2_thick=6, layer4_thick=5,
                               ke=ke, ks=ks)

            # 以平均绝对误差作为评价指标
            error_mean = mean_absolute_error(real_temp, result[:, -1])
            error_squ = mean_squared_error(real_temp, result[:, -1])
            list_ke.append(ke)
            list_ks.append(ks)
            list_error_mean.append(error_mean)
            list_error_square.append(error_squ)

    print('粗略搜索完成，正在绘图....')

    # 搜索结果，误差分布图
    fig_1 = plt.figure(figsize=(8, 6))
    ax_1 = Axes3D(fig_1)
    ke_g = np.arange(start, stop, step)
    ks_g = np.arange(start, stop, step)
    error_g = np.reshape(list_error_mean, newshape=(len(ke_g), len(ks_g)))
    ks_g, ke_g = np.meshgrid(ke_g, ks_g)
    ax_1.plot_surface(ks_g, ke_g, error_g, cmap='turbo')
    ax_1.set_title('粗略参数搜索')
    ax_1.set_ylabel('外界环境与第一层的热对流系数')
    ax_1.set_xlabel('空气与皮肤热对流系数')
    ax_1.set_zlabel('平均绝对误差值')
    plt.legend(fontsize='xx-large')
    plt.savefig('参数搜索与误差分布图.png')
    plt.show()

    fig_1 = plt.figure(figsize=(8, 6))
    ax_1 = Axes3D(fig_1)
    ke_g = np.arange(start, stop, step)
    ks_g = np.arange(start, stop, step)
    error_g = np.reshape(list_error_square, newshape=(len(ke_g), len(ks_g)))
    ks_g, ke_g = np.meshgrid(ke_g, ks_g)
    ax_1.plot_surface(ks_g, ke_g, error_g, cmap='turbo')
    ax_1.set_title('粗略参数搜索')
    ax_1.set_ylabel('外界环境与第一层的热对流系数')
    ax_1.set_xlabel('空气与皮肤热对流系数')
    ax_1.set_zlabel('均方误差值')
    plt.legend(fontsize='xx-large')
    plt.savefig('参数搜索与误差分布图.png')
    plt.show()


# 问题2求解函数
def problem_2():

    # 第一问解得的热对流系数
    ke = 156.5
    ks = 8.5

    # 根据附件中所给范围，设定搜索范围及步长
    start = 0.6
    stop = 25
    step = 0.2

    # 设置搜索优化的指标
    best_thick = 0
    best_temp = 44.0
    five_s = 300
    best_skin_temp = 47.0

    print('正在搜索第II层最优厚度....')
    for thick in np.arange(start, stop, step):
        result = calculate(environment_temp=65.0, time=60,
                           layer2_thick=thick, layer4_thick=5.5,
                           ke=ke, ks=ks)

        # 第55分钟处皮肤处温度
        five_min_temp = result[-five_s, -1]
        skin_temp = result[-1, -1]

        print('\n第II层厚度为：{0:.2f}mm'.format(thick))
        print('此厚度下55分钟时皮肤外侧温度为：{}度'.format(five_min_temp))
        print('此厚度下60分钟时皮肤外侧温度为：{}度'.format(skin_temp))

        # 搜索到符合条件的参数，保存并结束搜索
        if five_min_temp < best_temp and skin_temp <= best_skin_temp:
            best_temp = five_min_temp
            best_skin_temp = skin_temp
            best_thick = thick
            # 输出结果
            print('该厚度符合限制条件，搜索完成！')
            print('第II层最优厚度为：{}mm'.format(best_thick))
            break
        else:
            print('该厚度不符合限制条件！\n')


problem_2()