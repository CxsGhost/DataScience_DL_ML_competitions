from keras.datasets import mnist
from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.utils import to_categorical

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

# 架构卷积神经网络
network = models.Sequential()
# 这次直接输入原型图片28,28,通道为1，卷积核的步幅默认为（1, 1)
network.add(layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPool2D(pool_size=(2, 2)))
network.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
network.add(layers.MaxPool2D(pool_size=(2, 2)))
network.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))  # 最后的卷积层输出形状为(None, 3, 3, 64)
# 最后一层卷积层，输出的是3D张量，要把它展平再分类
network.add(layers.Flatten())  # 展开层的输出形状为(None, 576)
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

# 查看当前网络的架构
print(network.summary())

# 编译模型并训练
network.compile(optimizer=optimizers.rmsprop(lr=0.002),
                loss=losses.categorical_crossentropy,
                metrics=[metrics.categorical_accuracy])

network.fit(x=train_images, y=train_labels,
            batch_size=64,
            epochs=5,
            verbose=1)

# 在测试集检查准确率
test_loss, test_accuracy = network.evaluate(x=test_images, y=test_labels)
print("loss：{} acc：{}".format(test_loss, test_accuracy))




















































