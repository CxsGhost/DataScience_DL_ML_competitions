from keras import layers
from keras import activations

# 恒等残差连接
x = 1
y = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(x)
y = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(y)
y = layers.Conv2D(128, 3, activation=activations.relu)(y)

y = layers.add([x, y])

# 线性残差连接
x_ = 1
y_ = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(x_)
y_ = layers.Conv2D(128, 3, activation=activations.relu, padding='same')(y_)
y_ = layers.MaxPool2D(2, strides=2)(y_)

residual = layers.add([x_, y_])
















