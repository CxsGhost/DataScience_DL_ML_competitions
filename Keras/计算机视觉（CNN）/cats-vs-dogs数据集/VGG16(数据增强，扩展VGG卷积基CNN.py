from keras.applications import VGG16
from keras import layers
from keras import models
from keras import metrics
from keras import optimizers
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

conv_base = VGG16(weights='imagenet',  # 指定模型初始化的权重检查点
                  include_top=False,  # 指定是否取用分类器部分（全连接层）
                  input_shape=(150, 150, 3))  # 形状参数非必须，不给则可以处理任意形状

network = models.Sequential()
network.add(conv_base)
network.add(layers.Flatten())
network.add(layers.Dense(256, activation='relu'))
# 数据增强了则可以不用dropout
network.add(layers.Dense(1, activation='sigmoid'))

print(network.summary())

# 在编译和训练分类器之前，要先冻结卷积基，否则Dense层一开始随机的权重会在网络中传播，导致卷积基权重被破坏
# 这样设置后总共剩下两个Dense层的4个权重需要训练（主权重矩阵和偏置矩阵各2个）
conv_base.trainable = False

# 接下来处理数据进行数据增强
train_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/train'
validation_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/validation'

train_data_generator = ImageDataGenerator(rescale=1.0 / 255,
                                          rotation_range=40,
                                          height_shift_range=0.2,
                                          width_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest')  # 填充方式默认就是nearest

# 注意，不能对验证数据增强
validation_data_generator = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_data_generator.flow_from_directory(train_dir,
                                                           target_size=(150, 150),
                                                           batch_size=20,
                                                           class_mode='binary')
validation_generator = validation_data_generator.flow_from_directory(validation_dir,
                                                                     target_size=(150, 150),
                                                                     batch_size=20,
                                                                     class_mode='binary')

network.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

history = network.fit_generator(train_generator,
                                steps_per_epoch=100,
                                epochs=30,
                                validation_data=validation_generator,
                                validation_steps=50,
                                verbose=1)

# 再次绘制曲线可知，过拟合明显减弱，验证集正确率达到96%
accuracy = history.history['binary_accuracy']
val_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epochs, accuracy, color='r', marker='o', linestyle='--', label='Accuracy')
ax_acc.plot(epochs, val_accuracy, color='b', marker='>', linestyle='-', label='Val_accuracy')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('acc')
ax_acc.set_title('Accuracy')
ax_acc.legend()

ax_loss = plt.figure(figsize=(9, 9)).gca()
ax_loss.plot(epochs, loss, color='r', marker='o', linestyle='--', label='Loss')
ax_loss.plot(epochs, val_loss, color='b', marker='>', linestyle='-', label='Val_loss')
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.set_title('Loss')
ax_loss.legend()

plt.show()


# 下面将对最后一部分的几个卷积层进行微调（联合之前训练过的分类器）
# 先将所有层解冻
conv_base.trainable = True

# 然后选出要微调的层，其他冻结
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 重新编译模型，用很小的学习率来拟合，防止变化过大
network.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

history = network.fit_generator(train_generator,
                                steps_per_epoch=100,
                                epochs=100,
                                validation_data=validation_generator,
                                validation_steps=50)


# 再次绘制曲线对比，但发现曲线噪声很大，于是用指数加权移动平均（EWMA）来平滑曲线
def smooth_curve(points, factor=0.8):
    beta = 0
    for point in points:
        if points.index(point) == 0:
            beta = point
        else:
            point = (1 - factor) * point + factor * beta
            beta = point


accuracy = history.history['binary_accuracy']
val_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

smooth_curve(accuracy)
smooth_curve(val_accuracy)
smooth_curve(loss)
smooth_curve(val_loss)

epochs = range(1, len(accuracy) + 1)

ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epochs, accuracy, color='r', marker='o', linestyle='--', label='smoothed_Accuracy')
ax_acc.plot(epochs, val_accuracy, color='b', marker='>', linestyle='-', label='smoothed_Val_accuracy')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('acc')
ax_acc.set_title('Accuracy')
ax_acc.legend()

ax_loss = plt.figure(figsize=(9, 9)).gca()
ax_loss.plot(epochs, loss, color='r', marker='o', linestyle='--', label='smoothed_Loss')
ax_loss.plot(epochs, val_loss, color='b', marker='>', linestyle='-', label='smoothed_Val_loss')
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.set_title('Loss')
ax_loss.legend()

plt.show()





