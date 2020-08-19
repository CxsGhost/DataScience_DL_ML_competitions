"""
这是一个图像二分类问题，损失函数二元交叉熵，最后一层sigmoid
相比之前MNIST的要适当增大
该网络的特征图深度在不断增大（卷积层的卷积核数量由小到大，覆盖图的比例越来越大）
而特征图的尺寸在减小。这几乎是所审卷积神经网络的模式
"""
from keras import layers
from keras import models
from keras import metrics
from keras import optimizers
from keras import losses

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# 架构模型
network = models.Sequential()

# 输入尺寸150, 150是随意的选择
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Flatten())
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

# 查看模型概览
print(network.summary())

# 编译模型
network.compile(optimizer=optimizers.rmsprop(lr=1e-4),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])


# 对数据的处理:
# 读取并处理成RGB像素网格，这样输入就有三个深度
# 将像素网格转化成浮点型张量
# 进行归一化
# Keras有成熟的工具，可以创建python生成器将硬盘图像转换为张量，并且顺带可以进行归一化
train_data_generator = ImageDataGenerator(rescale=1 / 255.0)
test_data_generator = ImageDataGenerator(rescale=1 / 255.0)

train_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/train'
validation_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/validation'

train_generator = train_data_generator.flow_from_directory(train_dir,
                                                           target_size=(150, 150),
                                                           batch_size=20,
                                                           class_mode='binary')  # 二进制分类标签
validation_generator = test_data_generator.flow_from_directory(validation_dir,
                                                               target_size=(150, 150),
                                                               batch_size=20,
                                                               class_mode='binary')

# 来查看一个生成器的输出，生成器会不断循环图片，要人为break
for data_batch, labels_batch in train_generator:
    print('data batch shape:{}'.format(data_batch.shape))
    print('labels batch shape:{}'.format(labels_batch.shape))
    break
    # 输出：
    # data batch shape:(20, 150, 150, 3)
    # labels batch shape:(20,)

# 可以直接使用generator来拟合模型
history = network.fit_generator(train_generator,
                                steps_per_epoch=100,  # 数据是不断生成的，我们要说明生成多少次完成一个epoch
                                epochs=30,
                                validation_data=validation_generator,
                                validation_steps=50)  # 同理要说明生成多少次走完一遍

# 在训练完成后保存模型
network.save('cats_and_dogs_small_1.h5')

# 绘制训练过程中损失曲线和精度曲线
accuracy = history.history['binary_accuracy']
val_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epochs, accuracy, color='r', marker='o', linestyle='--', label='accuracy')
ax_acc.plot(epochs, val_accuracy, color='b', marker='>', linestyle='-', label='val_accuracy')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('acc')
ax_acc.set_title('Accuracy')
ax_acc.legend()

ax_loss = plt.figure(figsize=(9, 9)).gca()
ax_loss.plot(epochs, loss, color='r', marker='o', linestyle='--', label='loss')
ax_loss.plot(epochs, val_loss, color='b', marker='>', linestyle='-', label='val_loss')
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.set_title('Loss')
ax_loss.legend()

plt.show()


# 经过绘图观察后发现，因为仅2000个样本，所以很早就开始过拟合






