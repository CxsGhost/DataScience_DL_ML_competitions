from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models


# keras中设置按需分配(因为将建立两个模型
import tensorflow as tf

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# 可视化drop和数据增强的CNN
network = load_model('cats_and_dogs_small_2.h5')
print(network.summary())

# 中间激活，展示的是卷积层和池化层输出的特征图（层的输出称为该层的激活）
# 我们要从三个维度进行可视化（宽 高 通道）


# 在测试集取一张图片，并进行预处理
img_path = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/' \
             'dogs-vs-cats_small/test/cats/cat.1700.jpg'
img = image.load_img(img_path,
                     target_size=(150, 150))  # 返回的是一个图片对象实例
img_tensor = image.img_to_array(img)
print(img_tensor.shape)  # (150, 150, 3)
img_tensor = np.expand_dims(img_tensor, axis=0)  # 插入一个新轴
img_tensor /= 255.0
# 看一下这张图的形状
print(img_tensor.shape)  # (1, 150, 150, 3)


# 展示一下原图
plt.imshow(img_tensor[0])
plt.show()


# 使用Model类，以图像批量为输入，输出层的激活，Model允许模型有多个输出，不同于Sequential类
# 提取network前8层的输出(只有这8层是卷积层）
layer_outputs = [layer.output for layer in network.layers[:8]]
# 可以理解为Model操控network。该模型1个输入，8个输出
activation_model = models.Model(inputs=network.input, outputs=layer_outputs)

# 输出图像得到激活，返回的是一个列表，列表中的元素是每层的激活
activations = activation_model.predict(img_tensor)

# 取出并查看第一层的激活
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# 输出第1层2个卷积核的响应图（第1层激活的2个通道）
plt.matshow(first_layer_activation[0, :, :, 4])
plt.matshow(first_layer_activation[0, :, :, 7])
plt.show()


# 将每个中间激活的所有通道可视化（当时弄的时候写错一个池化层参数，结果导致只有前5层特征图是正方形，于是就只展示前5层的吧

# 获取每层的名字
layer_names = [layer.name for layer in network.layers]

# 只展示前五层
number = 0

for layer_name, layer_activation in zip(layer_names, activations):

    number += 1
    if number > 5:
        break

    # 每行最多绘制16个响应图
    images_per_row = 16

    # 特征图的通道个数
    n_channel = layer_activation.shape[-1]

    # 计算需要几行
    n_rows = n_channel // images_per_row

    # 特征图的形状为（1，size，size， n_feature）
    channel_size = layer_activation.shape[1]

    # 在这个矩阵中将通道平铺
    display_grid = np.zeros((channel_size * n_rows, channel_size * images_per_row))

    for row in range(n_rows):
        for col in range(images_per_row):
            channel_image = layer_activation[0, :, :, row * col]  # 坐标位置相乘，便是第几个通道的图片

            # 进行一些处理，看起来更美化
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            # 剪切过大值和过小值，限制在0~255
            channel_image = np.clip(channel_image, 0, 255).astype(np.int)
            display_grid[channel_size * row: channel_size * (row + 1),
                         channel_size * col: channel_size * (col + 1)] = channel_image
    scale = 1.0 / channel_size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)  # 去掉网格线
    # aspect设置auto是会更改图像宽高比来适应figure宽高比，简单来说会导致图片被拉长或压扁
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()





