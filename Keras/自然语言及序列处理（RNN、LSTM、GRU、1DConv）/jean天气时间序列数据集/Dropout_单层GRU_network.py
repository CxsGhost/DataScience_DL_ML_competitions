import os
import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt


# 读取数据到numpy
data_dir = 'E:/py/科学计算与机器学习/Keras/自然语言及序列处理（RNN、LSTM、GRU、1DConv）' \
           '/jean天气时间序列数据集/jena_climate_2009_2016.csv'
data_file_name = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
raw_data = pd.read_csv(data_file_name, index_col=1)

# 删除第一列时间轴，转化为float——numpy
raw_data = raw_data.drop(columns=raw_data.columns[0])
float_data = raw_data.values.astype(np.float)

# 进行标准化处理(前20000个为训练数据）
mean_ = float_data[: 200000].mean(axis=0)
float_data -= mean_
std_ = float_data[: 200000].std(axis=0)
float_data /= std_


# 样本生成器
def generator(data, lookback, delay,
              min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - 1 - delay
    i = min_index + lookback
    while True:
        if shuffle:
            # 随机选128个数字，组成128个随机样本，而不是按顺序循环生成
            rows = np.random.randint(min_index + lookback,
                                     max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, i + batch_size)
            i += batch_size

        samples = np.zeros((batch_size,
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((batch_size, ))
        for j, row in enumerate(rows):
            indices = range(row - lookback, row, 6)
            samples[j] = data[indices]
            targets[j] = data[row + delay][1]
        yield samples, targets


# 设置生成数据集的各种参数
lookback_ = 1440
step_ = 6
delay_ = 144
batch_size_ = 128

train_generator = generator(float_data,
                            delay=delay_,
                            lookback=lookback_,
                            batch_size=batch_size_,
                            min_index=0,
                            max_index=200000,
                            shuffle=True,
                            step=step_)
val_generator = generator(float_data,
                          delay=delay_,
                          lookback=lookback_,
                          batch_size=batch_size_,
                          min_index=200001,
                          max_index=300000,
                          step=step_)
test_generator = generator(float_data,
                           delay=delay_,
                           lookback=lookback_,
                           batch_size=batch_size_,
                           min_index=300001,
                           max_index=None,
                           step=step_)

# 生成器fit会一直循环，需要在一定step停下
val_steps = (300000 - 200001 - lookback_) // batch_size_
test_steps = (len(float_data) - 300001 - lookback_) // batch_size_


network = models.Sequential()
network.add(layers.GRU(32,
                       dropout=0.2,  # 输入单元的dropout
                       recurrent_dropout=0.05,  # 循环单元的dropout
                       input_shape=(240, float_data.shape[-1])))
network.add(layers.Dense(1))

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.mae)

print(network.summary())

history = network.fit_generator(train_generator,
                                steps_per_epoch=500,
                                epochs=40,
                                validation_data=val_generator,
                                validation_steps=val_steps)

# 绘制损失曲线
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, color='r', marker='o', linestyle='--', label='tra_Loss')
plt.plot(epochs, val_loss, label='Val_Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('LOSS_VALUE')
plt.legend()
plt.show()






































