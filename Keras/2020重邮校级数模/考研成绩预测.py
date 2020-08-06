import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt

# 读取数据，设置第一行为列标签，第一行为行标签
file_dir = 'E:/py/科学计算与机器学习/Keras/2020重邮校级数模/B题夹/B题附件2.xlsx'
all_data = pd.read_excel(file_dir, index_col=0, header=0)

# 提取数据矩阵
all_data = all_data.values

# 概率论成绩有缺失，先由此划分训练集验证集，和待预测集，用线性模型补缺

# 两个列表分别存储数据索引
train_validation_samples = []
pre_samples_index = []

# 存放预测数据
pre_samples = []

# 值不缺失的数据索引归入训练、验证集，其余归入待预测数据
for i in range(all_data.shape[0]):
    if np.isnan(all_data[i, 3]):
        pre_samples_index.append(i)
    else:
        train_validation_samples.append(i)

# 根据分好的索引，从原数据集划分
train_validation_samples = all_data[train_validation_samples]
pre_samples = all_data[pre_samples_index]
print('------用于训练和验证的数据共：{}条'.format(len(train_validation_samples)))
print('------待预测（补缺）数据：{}条'.format(len(pre_samples)))

# 分离x和y
train_validation_samples_x = train_validation_samples[:, : -2]
train_validation_samples_y = train_validation_samples[:, 3]
print('x维度：{}'.format(train_validation_samples_x.shape))

# 划分训练集和验证集,数据较少，但考虑是回归问题，于是无需过多验证数据
train_x, validation_x, train_y, validation_y = train_test_split(train_validation_samples_x,
                                                                train_validation_samples_y,
                                                                test_size=0.15)

# 数据并不会出现离群值，采用一般的线性归一化即可
scaler = MinMaxScaler(feature_range=(0.1, 1))
scaler.fit(train_validation_samples_x)
train_validation_samples_x = scaler.transform(train_validation_samples_x)

# 建立补缺模型
# network_patch = keras.models.Sequential()
# network_patch.add(keras.layers.Dense(32, activation=keras.activations.relu, input_shape=(3, )))
# network_patch.add(keras.layers.Dense(32, activation=keras.activations.relu))
# network_patch.add(keras.layers.Dense(32, activation=keras.activations.relu))
# network_patch.add(keras.layers.Dense(1))
network = keras.models.Sequential()
network.add(keras.layers.Dense(32, activation=keras.activations.relu, input_shape=(3, )))
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32, activation=keras.activations.relu))
network.add(keras.layers.Dense(1))

network.compile(optimizer=keras.optimizers.RMSprop(lr=5e-4),
                loss=keras.losses.mean_squared_error,
                metrics=[keras.metrics.mae])

print(network.summary())

history = network.fit(x=train_validation_samples_x,
                      y=train_validation_samples_y,
                      batch_size=128,
                      epochs=600)

error = history.history['mean_absolute_error']

epoch = range(1, len(error) + 1)

# 绘制绝对损失趋势图
ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epoch, error, color='b', marker='*', linestyle='--', label='Error')
ax_acc.set_title('Error')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('error')
ax_acc.legend()

plt.show()


# 补全成绩，更新数据
pre_samples = scaler.transform(pre_samples[:, : -2])
all_data[pre_samples_index, 3] = network.predict(x=pre_samples)[:, 0]
final_x = all_data[:, : -1]
final_y = all_data[:, 4]
scaler.fit(final_x)
final_x = scaler.transform(final_x)

# 建立最终模型
network = keras.models.Sequential()
network.add(keras.layers.Dense(32, activation=keras.activations.relu, input_shape=(4, )))
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32, activation=keras.activations.relu))
network.add(keras.layers.Dense(1))

network.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
                loss=keras.losses.mean_squared_error,
                metrics=[keras.metrics.mae])

history = network.fit(x=final_x, y=final_y,
                      batch_size=128,
                      epochs=2100)

error = history.history['mean_absolute_error']

epoch = range(1, len(error) + 1)

# 绘制绝对损失趋势图
ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epoch, error, color='b', marker='*', linestyle='--', label='Error')
ax_acc.set_title('Error')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('error')
ax_acc.legend()

plt.show()

# 评估模型效果
pre_y = network.predict(x=final_x)
print(np.round(final_y - pre_y[:, 0]))
print('------平均绝对误差：{}'.format(np.mean
                               (np.absolute
                                (np.round
                                 (final_y - pre_y[:, 0])))))


# 对比论文中方法
weights = np.array([[0.3068],
                    [0.2707],
                    [0.3113],
                    [-0.0049]])
paper_pre_y = np.dot(all_data[:, : -1], weights)
print('-------paper平均绝对误差：{}'.format(np.mean
                                     (np.absolute
                                      (np.round
                                       (paper_pre_y[:, 0] + 4.089908 - final_y)))))


# [  2.  -1.  -5.   6.  -1.   9.   2.   2.   1.   5.  -7.   1.   5.   3.
#    3.  -4.  -7.   2.   1.  -2.  -4.   5.  -1.   8.  -1.   1.  -5.   2.
#   -3.  -3.   4.  -2.   3. -20.   7.   2.   1.   1.  -1.   0.   3.   3.
#    5.   3.  -3.  14.  -1.  -5.   0.   4.  -9.  -2.   2.  -2.  -3.   5.
#    4.  -0.  -7.  -0.   9.   5.   2.   1.   3.   3.  -9.   4.   2. -15.
#   -8.   3.   1.  -9.   1.  -2.  -7.   5.  -1.   3.  -1.   8.   1.   2.
#    5.   6.   8.   7.   6.  10.   7.   8.   4.   3.  -1.   4.   3.   4.
#    0.  -4.   3.   5.   0.   6.   2.   4.  -5.  -9.   7.   3.  -7.  -3.
#    1.   4.   5.   2.   5.  -2.  -5.   0.   8.   1.   4.   5.   0.  -1.
#    5.  -8.   0.   1.  -6.   5.  -3.   5.   0.  -4.   1.   3.   4.   2.
#   -1.   3.   5.   4.  -0.   3.   4.  -0.  -2.  -1.   3.   1.   3.  -1.
#    4.   2.  -1.   3.   2.   1.   3.   5.  -0.   3.   3.  -1.  -2.  -2.
#    5.   1.   1.   3.  -2.  -7.  16.   6.   4.   0.   2.   2.   2.  -1.
#    1.   2.   4.   0.   1.  -1.   5.   8.   2.  13.   0.   2.   2.   1.
#    1.   4.   2.  -1.   1.  -3.   0.   1.   2.   1.  -2.  -0.   1.  -3.
#    4.  -1.   2.  -1.   1.   1.  -0.  -1.   4.   5.   2.   0.  -2.  -3.
#   -1.   6.  -6.  -5.   3.   2.  -2.   2.  -2.  -0.]
# ------平均绝对误差：3.341880341880342
# -------paper平均绝对误差：14.384615384615385


















