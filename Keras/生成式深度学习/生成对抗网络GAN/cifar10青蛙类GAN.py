from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

# 生成器网络
generator_input = keras.Input(shape=(latent_dim, ),
                              name='generator_input')

# 将输入转换为16，16宽高，128个通道的特征图
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape(target_shape=(16, 16, 128))(x)

x = layers.Conv2D(256, (5, 5), padding='same')(x)
x = layers.LeakyReLU()(x)

# 上采样为32*32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', name='reverse_conv')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation=keras.activations.tanh, padding='same')(x)

# 实例化模型，他将（latent,)的向量转化为（32,32,3）的图像
generator = keras.models.Model(inputs=generator_input, outputs=x, name='generator_model')
print(generator.summary())


# 判别器网络
discriminator_input = keras.Input(shape=(height, width, channels),
                                  name='discriminator_input')
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# 该dropout层是很重要的技巧，一定要有
x = layers.Dropout(rate=0.4)(x)

x = layers.Dense(1, activation=keras.activations.sigmoid)(x)

discriminator = keras.models.Model(inputs=discriminator_input, outputs=x, name='discriminator_model')
print(discriminator.summary())


discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4,
                                                   clipvalue=1.0,  # 梯度裁剪（限制梯度范围
                                                   decay=1e-8)  # 学习率衰减，稳定过程

# 判别器需要被单独训练，故在此编译一下
discriminator.compile(optimizer=discriminator_optimizer,
                      loss=keras.losses.binary_crossentropy)


# 冻结判别器(该处设置仅仅在接下来的GAN模型中起作用，因为上面已经编译过，故上面的依然可以训练更新）
discriminator.trainable = False

# 设置对抗网络，将生成器和判别器连在一起
gan_input = keras.Input(shape=(latent_dim, ),
                        name='GAN_input')
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(inputs=gan_input, outputs=gan_output)
print(gan.summary())

# 设置优化器，编译GAN模型（此处的编译是为了训练生成器）
gan_optimizer = keras.optimizers.RMSprop(lr=4e-4,
                                         clipvalue=1.0,
                                         decay=1e-8)
gan.compile(optimizer=gan_optimizer,
            loss=keras.losses.binary_crossentropy)


# 实现GAN的训练
import os
from tensorflow.keras.preprocessing import image

# 加载数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('训练数据维度：{}'.format(x_train.shape))
print('标签维度：{}'.format(y_train.shape))

# 筛选出青蛙类，编号为6
x_train = x_train[y_train.flatten() == 6]
print(x_train.shape)
x_train = np.reshape(x_train, newshape=(x_train.shape[0], ) + (height, width, channels))

# 永远不要忘记标准化！！！
x_train = x_train.astype(np.float) / 255.0


iteration = 10000
batch_size = 20
save_dir = 'your_dir'

start = 0
for step in range(iteration):

    # 批量生成随机点
    random_latent_vectors = np.random.normal(loc=0.0, scale=1.0,
                                             size=(batch_size, latent_dim))

    # 通过生成器生成虚拟图像
    generated_images = generator.predict(x=random_latent_vectors)

    # 每生成一批虚拟图像，就要抽取一批真实图像，将二者混合形成新的训练数据
    stop = start + batch_size
    real_image = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_image], axis=0)

    # 同时也要更新标签
    labels = np.concatenate([np.zeros(shape=(batch_size, 1)), np.ones(shape=(batch_size, 1))])

    # 向标签中添加随机噪声，这是很重要的技巧
    labels += 0.05 * np.random.random(labels.shape)

    # 先训练判别器
    d_loss = discriminator.train_on_batch(x=combined_images, y=labels)

    # 在生成一批新的随机向量，训练GAN（其实是生成器）
    random_latent_vectors = np.random.normal(loc=0.0, scale=1.0,
                                             size=(batch_size, latent_dim))

    misleading_targets = np.ones(shape=(batch_size, 1))

    # 训练生成器
    a_loss = gan.train_on_batch(x=random_latent_vectors, y=misleading_targets)

    #
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    # 每100补保存并绘图
    if step % 100 == 0:
        gan.save_weights('gan.h5')

        print('discriminator loss:{}'.format(d_loss))
        print('generator loss:{}'.format(a_loss))

        # 从前面生成器生成的图像批量中，保存一张生成图像(别忘了逆归一化操作）
        img = image.array_to_img(generated_images[0] * 255.0, scale=False)
        img.save(os.path.join(save_dir, 'generated_frog_{}_.png'.format(step)))

        # 也保存一张真实图像用于对比观察
        img = image.array_to_img(real_image[0] * 255.0, scale=False)
        img.save(os.path.join(save_dir, 'real_frog_{}_.png'.format(step)))





