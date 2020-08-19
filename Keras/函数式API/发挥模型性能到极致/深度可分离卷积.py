"""
该模型是在小型数据集上构建一个轻量化的模型，用于图像多分类任务
"""
from keras import models
from keras import layers

height = 64
width = 64
channels = 3
num_class = 10

network = models.Sequential()
network.add(layers.SeparableConv2D(32, (3, 3), activation='relu',
                                   input_shape=(height, width, channels)))
network.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))

network.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
network.add(layers.SeparableConv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))

network.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
network.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
network.add(layers.GlobalMaxPooling2D())

network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(num_class, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])




























