import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
import tensorflow
tensorflow.compat.v1.disable_eager_execution()

img_shape = (28, 28, 1)
batch_size = 16

# 潜在空间维度
latent_dim = 2

# VAE编码网络
input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, (3, 3),
                  padding='same', activation=keras.activations.relu)(input_img)
x = layers.Conv2D(64, (3, 3),
                  padding='same', activation=keras.activations.relu, strides=(2, 2))(x)
x = layers.Conv2D(64, (3, 3),
                  padding='same', activation=keras.activations.relu)(x)
x = layers.Conv2D(64, (3, 3),
                  padding='same', activation=keras.activations.relu)(x)

# 保存编码前最后的形状，用于后来输入解码器
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation=keras.activations.relu)(x)

# 输入图像最终被编码为如下两个参数(多元高斯分布）
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)


# 下面使用输出的两个统计参数选取一个潜在空间点z来进行解码
def sampling(args):
    z_mean_, z_log_var_ = args

    # 一个批量中有多少样本点（z_mean[0]，就抽取多少个(在标准正态分布抽取
    epsilon = K.random_normal(shape=(K.shape(z_mean_)[0], latent_dim),
                              mean=0.0, stddev=1.0)
    return z_mean_ + K.exp(z_log_var_ * 0.5) * epsilon


# 在keras中，任何对象都该是一个层，所以上面函数不是内置层的一部分，则应该包装到lambda层中（或自定义层）
z = layers.Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])

# VAE解码网络（输入编码出来的潜在空间点）
decoder_input = layers.Input(K.int_shape(z)[1:])

# np.prod函数用来计算所有元素的乘积，对于有多个维度的数组可以指定轴
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation=keras.activations.relu)(decoder_input)

# 重新整理成编码前图片的尺寸(即转化成特征图）
x = layers.Reshape(shape_before_flattening[1:])(x)


# 使用以下两个层，将z解码为与原始图像相同尺寸的特征图
# 此处是一个逆卷积层，也就是进行上采样，放大图片，具体推导百度
x = layers.Conv2DTranspose(32, (3, 3),
                           padding='same', activation=keras.activations.relu, strides=(2, 2))(x)
x = layers.Conv2D(1, (3, 3),
                  padding='same', activation=keras.activations.sigmoid)(x)


# 解码器模型实例化，注意！！该模型的输入是decoder——input，
decoder = models.Model(inputs=decoder_input, outputs=x)

# 将随机抽取点得到的结果应用于模型，得到解码后的模型
z_decoded = decoder(z)


# VAE是双重损失，不适合于loss（output， target）的形式。
# 因此，设置损失函数的方法为：写一个自定义层，在内部使用内置的add_loss层方法创建想要的损失函数
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x_, z_decoded_):
        x_ = K.flatten(x_)
        z_decoded_ = K.flatten(z_decoded_)
        xent_loss = keras.metrics.binary_crossentropy(x_, z_decoded_)
        kl_loss = -5e-4 * K.mean(1 + z_log_var -
                                 K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # 通过编写call方法来实现自定义层
    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        z_decoded_ = inputs[1]
        loss = self.vae_loss(x_, z_decoded_)
        self.add_loss(loss, inputs=inputs)

        # 我们不使用这个输出，但层必须要有返回值
        return x_


y = CustomVariationalLayer()([input_img, z_decoded])


from keras.datasets import mnist

# 实例化模型
vae = models.Model(inputs=input_img, outputs=y)

# 损失函数包含在自定义层中，编译时无需指定外部损失，这意味着训练过程中不需要传入目标数据
vae.compile(optimizer=keras.optimizers.RMSprop(),
            loss=None)
print(vae.summary())

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.astype(np.float) / 255
train_x = train_x.reshape(train_x.shape + (1, ))  # 由于通道数必须有，所以加上1个轴
test_x = test_x.astype(np.float) / 255
test_x = test_x.reshape(test_x.shape + (1, ))

callback_list = [keras.callbacks.TensorBoard(log_dir='my_log',
                                             histogram_freq=1)]

vae.fit(x=train_x, y=None,
        shuffle=True,
        epochs=4,
        batch_size=batch_size,
        validation_data=(test_x, None))


# 一旦训练完成，我们就可以用decoder来将潜在空间任意向量转化为图像
import matplotlib.pyplot as plt
from scipy.stats import norm

# 我们将显示15*15的数字网格（生成255个数字）
n = 15

digit_size = 28
figure = np.zeros(shape=(digit_size * n, digit_size * n))

# 使用scipy的ppf函数对线性分割的坐标进行变换，以生成潜在变量z的值
# 说白了就是等差数列，在（0，1）*（0,1）间生成255个点
# 然后ppf对其进行变换（整个函数的作用是找到正态分布中分布函数为x时对应的x轴的点（是求积分的反向操作）
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([xi, yi])

        # 将z多次重复，以构建一个完整的批量
        z_sample = np.tile(z_sample, batch_size).reshape((batch_size, 2))

        # 批量解码为数字图像
        z_decoder = decoder.predict(z_sample, batch_size=batch_size)

        # 去掉通道数
        digit = z_decoder[0].reshape((digit_size, digit_size))

        # 将图片放入网格
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
























