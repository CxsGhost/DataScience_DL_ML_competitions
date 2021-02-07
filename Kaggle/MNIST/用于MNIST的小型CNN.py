from keras.datasets import mnist
from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications import VGG16


# 获取内置的mnist数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, test_labels.shape)  # 输出(60000, 28, 28) (10000,)

# 输入数据要转化为（样本数量，宽，高，通道）本次通道为1。并且依然要进行归一化
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255

# 把labels进行one-hot化
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = np.concatenate([train_images, test_images], axis=0)
train_labels = np.concatenate([train_labels, test_labels], axis=0)


# 架构卷积神经网络
network = models.Sequential()
# 这次直接输入原型图片28,28,通道为1，卷积核的步幅默认为（1, 1)
network.add(layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
network.add(layers.ReLU())
network.add(layers.MaxPool2D(pool_size=(2, 2)))
network.add(layers.Conv2D(64, (3, 3)))
network.add(layers.ReLU())
network.add(layers.Dropout(0.2))
network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dropout(0.2))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))


# 查看当前网络的架构
print(network.summary())

# 编译模型并训练
network.compile(optimizer=optimizers.Adam(),
                loss=losses.categorical_crossentropy,
                metrics=[metrics.categorical_accuracy])

history = network.fit(x=train_images, y=train_labels,
                      batch_size=256,
                      epochs=15,
                      verbose=1)


train_images = np.reshape(train_images, newshape=(train_images.shape[0], 28 * 28))

network_1 = models.Sequential()
network_1.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network_1.add(layers.BatchNormalization())
network_1.add(layers.Dropout(0.2))
network_1.add(layers.Dense(512, activation='relu'))
network_1.add(layers.BatchNormalization())
network_1.add(layers.Dropout(rate=0.2))
network_1.add(layers.Dense(512, activation='relu'))
network_1.add(layers.BatchNormalization())
network_1.add(layers.Dense(10, activation='softmax'))

network_1.compile(optimizer='adam',
                  loss=losses.categorical_crossentropy)

network_1.fit(train_images, train_labels,
              batch_size=256,
              epochs=20)



# 在测试集检查准确率
test_loss, test_accuracy = network.evaluate(x=test_images, y=test_labels, batch_size=256)
print("loss：{} acc：{}".format(test_loss, test_accuracy))

test_data = pd.read_csv("test.csv", header=0, index_col=None)
test_data = test_data.values
test_data = np.reshape(test_data, newshape=(28000, 28, 28, 1))

pre_labels = network.predict(test_data, batch_size=64) * 0.7
test_data = np.reshape(test_data, newshape=(test_data.shape[0], 28 * 28))
pre_labels += network_1.predict(test_data, batch_size=64) * 0.3
pre_labels = np.argmax(pre_labels, axis=1)

image_id = np.arange(1, len(pre_labels) + 1, 1)
pre_data = pd.DataFrame(data={'ImageID': image_id, 'Label': pre_labels})
pre_data.to_csv("Submit.csv", index=False)



















































