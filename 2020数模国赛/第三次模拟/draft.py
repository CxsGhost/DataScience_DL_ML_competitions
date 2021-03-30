
import numpy as np
import pandas as pd
from scipy import optimize
# import keras
from tensorflow import keras
import tensorflow as tf

# 读取数据
jia_data = pd.read_excel('雷达数据.xlsx', sheet_name='第一组飞机甲', header=0)
yi_data = pd.read_excel('雷达数据.xlsx', sheet_name='第二组飞机乙', header=0)
bing_data = pd.read_excel('雷达数据.xlsx', sheet_name='第三组飞机丙', header=0)

jia_data = jia_data.values / 10000
yi_data = yi_data.values / 10000
bing_data = bing_data.values / 10000


jia_y = jia_data[:, -1]
yi_y = (yi_data[:, -1] - np.average(yi_data[:, -1])) / 62 + np.average(yi_data[:, -1])
bing_y = (bing_data[:, -1] - np.average(bing_data[:, -1])) / 62 + np.average(bing_data[:, -1])

jia_x = jia_data[:, :3]
yi_x = yi_data[:, :3]
bing_x = bing_data[:, :3]


data_ = bing_x
y_ = bing_y
# 构建图模型，求解梯度，loss
# def structure_graph(data_, y_):
location = keras.backend.placeholder(shape=(3, ), dtype='float64',
                                     name='location')
loss = keras.backend.variable(value=0.0, dtype='float64',
                              name='square_loss')
for i in range(len(y_)):
    loss = loss + keras.backend.square(keras.backend.square(location[0] - data_[i][0]) +
                                       keras.backend.square(location[1] - data_[i][1]) +
                                       keras.backend.square(location[2]) -
                                       np.square(y_[i]))
print()
grads = tf.gradients(loss, location)[0]
iterate_func = keras.backend.function(inputs=[location], outputs=[loss, grads])



# # func_jia = structure_graph(jia_x, jia_y)
# # func_yi = structure_graph(yi_x, yi_y)
# func_bing = structure_graph(bing_x, bing_y)