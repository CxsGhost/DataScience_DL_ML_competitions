from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/train'
validation_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/validation'

# 架构新模型
network = models.Sequential()
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Flatten())
network.add(layers.Dropout(0.5))  # 这里的权重比较多，且是第一个全连接层，使用Dropout
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

# 利用数据增强生成器训练集卷积神经网络
train_data_generator = ImageDataGenerator(rescale=1.0 / 255,
                                          rotation_range=40,
                                          height_shift_range=0.2,
                                          width_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest')  # 填充方式默认就是nearest

# 注意，不能对验证数据增强
test_data_generator = ImageDataGenerator(rescale=1.0 / 255)

# 从train_dir以及validation_dir读取数据
train_generator = train_data_generator.flow_from_directory(train_dir,
                                                           target_size=(150, 150),  # 会自动分成RGB三通道
                                                           batch_size=32,
                                                           class_mode='binary')

validation_generator = test_data_generator.flow_from_directory(validation_dir,
                                                               target_size=(150, 150),
                                                               batch_size=32,
                                                               class_mode='binary')

# 开始使用generator训练并保存模型
history = network.fit_generator(train_generator,
                                steps_per_epoch=100,
                                epochs=100,
                                validation_data=validation_generator,
                                validation_steps=50)

network.save('cats_and_dogs_small_2.h5')


# 绘制损失曲线和准确度曲线
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







































