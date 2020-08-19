import keras

# BN通常用在卷积层或者密集连接层之后
network = keras.models.Sequential()
network.add(keras.layers.Conv2D(128, (3, 3), strides=2, activation=keras.activations.relu))

# axis默认为-1，即对最后一个轴进行BN，但对于通道在前的conv2d，需要设置为1
network.add(keras.layers.BatchNormalization(axis=-1))

network.add(keras.layers.Dense(32, activation=keras.activations.relu))
network.add(keras.layers.BatchNormalization())




























