import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler


# 读取数据
train_data = pd.read_excel('data.xlsx', sheet_name='Sheet1',
                           index_col=None, header=None)
pre_data = pd.read_excel('data.xlsx', sheet_name='Sheet2',
                         index_col=None, header=None)
train_data = train_data.values
pre_data = pre_data.values

# 分离训练数据x和y
train_data_y = train_data[:, 0]
train_data_x = train_data[:, 1:]

# 归一化处理x，所有指标归至（0，1）
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_x = scaler.fit_transform(train_data_x)
pre_data = np.nan_to_num(pre_data, nan=1e-6)
pre_data_x = scaler.fit_transform(pre_data)

# 模型
network = keras.models.Sequential()
network.add(keras.layers.Dense(120, input_shape=(train_data_x.shape[1], )))
network.add(keras.layers.ReLU())
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.Dense(120))
network.add(keras.layers.ReLU())
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.Dense(120))
network.add(keras.layers.ReLU())
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.Dense(120))
network.add(keras.layers.ReLU())
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.Dense(120))
network.add(keras.layers.ReLU())
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.Dense(1))

network.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),
                loss=keras.losses.mean_squared_error,
                metrics=[keras.metrics.mean_absolute_error])

history = network.fit(x=train_data_x, y=train_data_y,
                      epochs=200,
                      batch_size=64)

pre_data_y = network.predict(x=pre_data_x)
print(pre_data_y)
