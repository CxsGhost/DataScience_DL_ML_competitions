from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
import numpy as np

# 按照以下代码在keras中设置按需分配
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

print(conv_base.summary())  # 最后输出为（4,4,512）

train_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/train'
validation_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/validation'
test_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/test'

data_generator = ImageDataGenerator(rescale=1.0 / 255)
batch_size = 20


def extract_feature(directory, sample_count):
    feature = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count, ))
    generator = data_generator.flow_from_directory(directory,
                                                   target_size=(150, 150),
                                                   batch_size=batch_size,
                                                   class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:

        # 生成器生成的数据进入VGG16进行特征提取（走一遍卷积基）
        feature_batch = conv_base.predict(inputs_batch)

        # 把卷积基输出的特征图存入生成好的feature零矩阵，同时把标签也存入
        feature[i * batch_size: (i + 1) * batch_size] = feature_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch

        # 生成器会一直循环生成，所以要根据生成进度break
        i += 1
        if i * batch_size >= sample_count:
            break
    return feature, labels


train_feature, train_labels = extract_feature(train_dir, 2000)
validation_feature, validation_labels = extract_feature(validation_dir, 1000)
test_feature, test_labels = extract_feature(test_dir, 1000)

# 目前各数据集特征的形状是（samples，4,4,512），要输入到全连接分类器中，需要转为（samples，8192）
train_feature = np.reshape(train_feature, (2000, 4 * 4 * 512))
validation_feature = np.reshape(validation_feature, (1000, 4 * 4 * 512))
test_feature = np.reshape(test_feature, (1000, 4 * 4 * 512))


network = models.Sequential()
# input_shape可以理解为是input_length和input_dim的组合
network.add(layers.Dense(256, activation='relu', input_shape=(8192, )))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

history = network.fit(train_feature, train_labels,
                      batch_size=20,
                      epochs=30,
                      validation_data=(validation_feature, validation_labels))

# 绘制曲线图
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

# 不使用数据增强，即便dropout比率很大，但依然很快过拟合，并且比较严重，但是验证集和训练集正确率却都是提高的















