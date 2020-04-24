from keras.datasets import mnist
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras import optimizers  # 各类优化器

# 获取内置的mnist数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 进行预处理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 转化为one-hot表示法
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 函数式架构神经网络，十分类似于之前手动实现的神经网络
input_tensor = layers.Input(shape=(28 * 28, ))
affine_hidden = layers.Dense(100, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(affine_hidden)

model = models.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])

model.fit(x=train_images, y=train_labels, batch_size=128, epochs=5)
