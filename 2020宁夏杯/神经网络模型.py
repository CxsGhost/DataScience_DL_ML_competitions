import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('data.xlsx')
data = data.values

# 数据划分
train_data = data[: 650]

validation_data = data[650: 733]
train_x = train_data[:, 0: 4]
train_y = train_data[:, -1] - 1
val_x = validation_data[:, 0: 4]
val_y = validation_data[:, -1] - 1


# 预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.concatenate([train_x, val_x], axis=0))
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
train_y = keras.utils.to_categorical(train_y, num_classes=6)
val_y = keras.utils.to_categorical(val_y, num_classes=6)


network = keras.models.Sequential()
network.add(keras.layers.Dense(32, use_bias= False, input_shape=(4, )))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32, use_bias=False))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(32, use_bias=False))
network.add(keras.layers.BatchNormalization())
network.add(keras.layers.ReLU())
# network.add(keras.layers.Dense(32))
# network.add(keras.layers.BatchNormalization())
# network.add(keras.layers.ReLU())
# network.add(keras.layers.Dense(32))
# network.add(keras.layers.BatchNormalization())
# network.add(keras.layers.ReLU())
# network.add(keras.layers.Dense(32))
# network.add(keras.layers.BatchNormalization())
# network.add(keras.layers.ReLU())
# network.add(keras.layers.Dense(32))
# network.add(keras.layers.BatchNormalization())
# network.add(keras.layers.ReLU())
network.add(keras.layers.Dense(6, activation=keras.activations.softmax))

network.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),
                loss=keras.losses.categorical_crossentropy,
                metrics=[keras.metrics.categorical_accuracy])

history = network.fit(x=train_x, y=train_y,
                      epochs=100,
                      batch_size=64,
                      validation_data=(val_x, val_y))

pre_train = network.predict(train_x)
pre_train = np.argmax(pre_train, axis=1)
print(pre_train)
print(network.evaluate(train_x, train_y))

history = history.history

loss = history['loss']
val_loss = history['val_loss']

accuracy = history['categorical_accuracy']
val_accuracy = history['val_categorical_accuracy']

epoch = range(1, len(accuracy) + 1)

# 绘制loss趋势图
ax_loss = plt.figure(figsize=(9, 9)).gca()
ax_loss.plot(epoch, loss, color='b', marker='>', linestyle='-', label='Loss')
ax_loss.plot(epoch, val_loss, color='r', marker='o', linestyle=':', label='Vla_Loss')
ax_loss.set_title('Loss')
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.legend()

# 绘制准确率趋势图
ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epoch, accuracy, color='b', marker='*', linestyle='--', label='Accuracy')
ax_acc.plot(epoch, val_accuracy, color='r', marker='o', linestyle='-.', label='Val_Accuracy')
ax_acc.set_title('Accuracy')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('accuracy')
ax_acc.legend()

plt.show()






