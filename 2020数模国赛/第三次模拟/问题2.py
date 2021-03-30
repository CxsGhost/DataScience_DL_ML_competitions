import numpy as np
import pandas as pd
from scipy import optimize
from tensorflow import keras
from tensorflow.compat.v1 import disable_eager_execution
disable_eager_execution()


jia_data = None
yi_data = None
bing_data = None
# 读取数据
try:
    jia_data = pd.read_excel('雷达数据.xlsx', sheet_name='第一组飞机甲', header=0)
    yi_data = pd.read_excel('雷达数据.xlsx', sheet_name='第二组飞机乙', header=0)
    bing_data = pd.read_excel('雷达数据.xlsx', sheet_name='第三组飞机丙', header=0)
except:
    print('数据读取失败！请将依赖数据：雷达数据.xlsx 与程序放于同一目录下！')
jia_data = jia_data.values / 10000
yi_data = yi_data.values / 10000
bing_data = bing_data.values / 10000


jia_y = jia_data[:, -1]
yi_y = (yi_data[:, -1] - np.average(yi_data[:, -1])) / yi_data.shape[0] + np.average(yi_data[:, -1])
bing_y = (bing_data[:, -1] - np.average(bing_data[:, -1])) / bing_data.shape[0] + np.average(bing_data[:, -1])

jia_x = jia_data[:, :3]
yi_x = yi_data[:, :3]
bing_x = bing_data[:, :3]


# 构建图模型，求解梯度，loss
def structure_graph(data_, y_):
    location = keras.backend.placeholder(shape=(3,), dtype='float64',
                                         name='location')
    loss = keras.backend.variable(value=0.0, dtype='float64',
                                  name='square_loss')
    for i in range(len(y_)):
        loss = loss + keras.backend.square(keras.backend.square(location[0] - data_[i][0]) +
                                           keras.backend.square(location[1] - data_[i][1]) +
                                           keras.backend.square(location[2]) -
                                           np.square(y_[i]))
    grads = keras.backend.gradients(loss, location)[0]
    iterate_func = keras.backend.function(inputs=[location], outputs=[loss, grads])
    return iterate_func


func_jia = structure_graph(jia_x, jia_y)
func_yi = structure_graph(yi_x, yi_y)
func_bing = structure_graph(bing_x, bing_y)


class Evaluator:
    def __init__(self, func):
        self.loss_value = None
        self.grads_value = None
        self.iterate_func = func

    def get_loss(self, x_):
        outs = self.iterate_func(inputs=[x_])
        loss_value = outs[0]
        grads_value = outs[1]
        self.loss_value = loss_value
        self.grads_value = grads_value

        return loss_value

    def get_grads(self, x_):
        assert self.grads_value is not None
        grads_value = np.copy(self.grads_value)
        self.grads_value = None
        self.loss_value = None

        return grads_value


evaluator_jia = Evaluator(func_jia)
evaluator_yi = Evaluator(func_yi)
evaluator_bing = Evaluator(func_bing)


def intersection(P1, P2, P3, r1, r2, r3):
    temp1 = P2 - P1
    e_x = temp1 / np.linalg.norm(temp1)
    temp2 = P3 - P1
    i = np.dot(e_x, temp2)
    temp3 = temp2 - i * e_x
    e_y = temp3 / np.linalg.norm(temp3)
    e_z = np.cross(e_x, e_y)
    d = np.linalg.norm(P2 - P1)
    j = np.dot(e_y, temp2)
    x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    y = (r1 * r1 - r3 * r3 - 2 * i * x + i * i + j * j) / (2 * j)
    temp4 = r1 * r1 - x * x - y * y
    if temp4 < 0:
        raise Exception("三球体无交点！")
    z = np.sqrt(temp4)
    p_12_a = P1 + x * e_x + y * e_y + z * e_z
    p_12_b = P1 + x * e_x + y * e_y - z * e_z
    if p_12_a[2] > 0:
        return p_12_a
    else:
        return p_12_b


x_jia = intersection(jia_x[4], jia_x[15], jia_x[28], jia_y[4], jia_y[15], jia_y[28]) / 10000
x_yi = intersection(yi_x[20], yi_x[24], yi_x[31], yi_y[20], yi_y[24], yi_y[31]) / 10000
x_bing = intersection(bing_x[2], bing_x[7], bing_x[11], bing_y[2], bing_y[7], bing_y[11]) / 10000


def solve(evaluator_, x):

    # 设置迭代次数
    iteration = 20

    min_val = None
    for i in range(iteration):
         x, min_val, info = optimize.fmin_l_bfgs_b(evaluator_.get_loss,
                                                   x,
                                                   fprime=evaluator_.get_grads,
                                                   )

    print('最终结果为：{}'.format(iteration, x))
    print('最终损失（loss）：{}'.format(min_val))
    evaluator_.get_loss(x)
    print('损失函数在该坐标点的梯度为：{}'.format(evaluator_.grads_value))


# 最终求解
print('\n甲组数据：')
solve(evaluator_jia, x_jia)

print('\n乙组数据：')
solve(evaluator_yi, x_yi)

print('\n丙组数据：')
solve(evaluator_bing, x_bing)
