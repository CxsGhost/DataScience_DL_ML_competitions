import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
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
pre_samples = []

print(all_data[10, 3] == np.nan)
# 值不缺失的数据索引归入训练、验证集，其余归入待预测数据
for i in range(all_data.shape[0]):
    if np.isnan(all_data[i, 3]):
        pre_samples.append(i)
    else:
        train_validation_samples.append(i)

# 根据分好的索引，从原数据集划分
train_validation_samples = all_data[train_validation_samples]
pre_samples = all_data[pre_samples]
print('用于训练和验证的数据共：{}条'.format(len(train_validation_samples)))
print('待预测（补缺）数据：{}条'.format(len(pre_samples)))

# 分离x和y
train_validation_samples_x = train_validation_samples[:, : -2]
train_validation_samples_y = train_validation_samples[:, 3]
print('x维度：{}'.format(train_validation_samples_x.shape))

# 划分训练集和验证集,数据较少，但考虑是回归问题，于是无需过多验证数据
train_x, validation_x, train_y, validation_y = train_test_split(train_validation_samples_x,
                                                                train_validation_samples_y,
                                                                test_size=0.15)

# 数据并不会出现离群值，于是采用一般的线性归一化即可
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_x)
train_x = scaler.transform(train_x)
validation_x = scaler.transform(validation_x)


# 考虑到数据较少，且维度不高，参数不多，于是先考虑岭回归预测
# 下面使用交叉验证的岭回归寻找最佳正则化强度
# 数据集很小，于是可以尝试更多的超参数,k_fold交叉验证5折
alphas_ = np.arange(0.01, 1, 0.01)
model = RidgeCV(alphas=alphas_, cv=6)
model.fit(train_x, train_y)

print('岭回归训练集上的R2系数：{}'.format(model.score(train_x, train_y)))
print('岭回归测试集上的R2系数：{}'.format(model.score(validation_x, validation_y)))

print('模型参数：{}'.format(model.coef_))
print(validation_y - model.predict(validation_x))
print(train_y - model.predict(train_x))


# 下面用神经网络来试一下
network = keras.models.Sequential()
network.add(keras.layers.Dense(16, activation='relu', input_shape=(3, )))
network.add(keras.layers.Dense(16, activation='relu'))
network.add(keras.layers.Dense(16, activation='relu'))
# network.add(keras.layers.Dense(32, activation='relu'))
network.add(keras.layers.Dense(1))

network.compile(optimizer=keras.optimizers.adam(),
                loss=keras.losses.mean_squared_error,
                metrics=[keras.metrics.mae])

history = network.fit(x=train_x, y=train_y,
                      batch_size=32,
                      epochs=300,
                      validation_data=(validation_x, validation_y))

error = history.history['mean_absolute_error'][-30:]
val_error = history.history['val_mean_absolute_error'][-30:]
# loss = history.history['loss']
# val_loss = history.history['val_loss']

epoch = range(1, len(error) + 1)

# 绘制绝对损失趋势图
ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epoch, error, color='b', marker='*', linestyle='--', label='Error')
ax_acc.plot(epoch, val_error, color='r', marker='o', linestyle='-.', label='Val_Error')
ax_acc.set_title('Error')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('error')
ax_acc.legend()

plt.show()


























