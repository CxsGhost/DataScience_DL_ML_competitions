import tensorflow as tf

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
"""
可以在输入空间中进行梯度上升来实现：
从空白输入图像开始，将梯度上升用于卷积神经网络输入的值，目的是让某个过滤器响应最大化。
得到的输入图像是选定过滤器具有最大响应的图像
"""
# 构建一个损失函数，目的是让某个过滤器的值最大化。例如对于VGG16，某个过滤器的损失如下
from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

network = VGG16(weights='imagenet',
                include_top=False)
print(network.summary())

# # layer_name = 'block3_conv1'
# # filter_index = 0
# #
# # layer_output = network.get_layer(layer_name).output
# # loss = K.mean(layer_output[:, :, :, filter_index])  # 直接把输出的值平均定义为损失，并且梯度上升，最大化
# #
# # # 为了实现梯度下降，我们需要得到损失对于模型输入的梯度。使用内置的gradient
# # grads = K.gradients(loss, network.input)[0]  # 返回的是一个张量列表（本例中也只有一个），直接把张量从列表取出就好
# #
# # # 为了让梯度下降顺利进行，将梯度除以L2范数来标准化，同时确保输入图像的更新大小始终位于相同的范围
# # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # 任何时候都要加一个数保底
# #
# # # 现在需要一种方法，给定输入图像，能够计算损失和梯度
# # # iterate是一个函数，输入一个张量，返回两个张量，分别是loss和grad
# # iterate = K.function([network.input], [loss, grads])
# # loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
# #
# # # 通过随机梯度下降让损失 最大化
# # input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.0  # 从一张随机灰度图像开始
# # step = 1.0  # 学习率
# # for i in range(40):
# #     loss_value, grads_value = iterate([input_img_data])
# #
# #     input_img_data += grads_value * step


# 得到的图像不位于0,255，需要处理才能显示为图像
def deprocess_img(x):
    # 进行标准化处理，均值为0
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 将上述代码片段放到一个函数中，输入层的名字和过滤器索引，返回一个张量图，是能够使该过滤器最大激活的图案样式
def generator_pattern(layer_name, filter_index, size_=150):
    layer_output = network.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])  # 直接把输出的值平均定义为损失，并且梯度上升，最大化

    # 为了实现梯度上升，我们需要得到损失对于模型输入的梯度。使用内置的gradient
    grads = K.gradients(loss, network.input)[0]  # 返回的是一个张量列表（本例中也只有一个），直接把张量从列表取出就好

    # 为了让梯度上升顺利进行，将梯度除以L2范数来标准化，同时确保输入图像的更新大小始终位于相同的范围（这是RMSprop优化法）
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # 任何时候都要加一个数保底

    # 现在需要一种方法，给定输入图像，能够计算损失和梯度
    # iterate是一个函数，输入一个张量，返回两个张量，分别是loss和grad
    iterate = K.function([network.input], [loss, grads])

    # 通过梯度上升让损失 最大化
    input_img_data = np.random.random((1, size_, size_, 3)) * 20 + 128.0  # 从一张随机灰度图像开始
    step = 1.0  # 学习率
    for _ in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_img(img)


image = generator_pattern('block3_conv1', 0)
plt.imshow(image)
plt.show()


# 生成某一层中所有过滤器响应模式的网格
size = 64
margin = 5

# 结果方块，留出边框的位置（0,0,0是黑色）
width = 8 * size + 7 * margin
length = 8 * size + 7 * margin
channel = 3
results = np.zeros((width, length, channel)).astype(np.int)

for i in range(8):
    for j in range(8):
        filter_img = generator_pattern('block1_conv1', j + (i * 8), size_=size)

        row_start = i * (size + margin)
        col_start = j * (size + margin)
        col_end = col_start + size
        row_end = row_start + size
        results[row_start: row_end, col_start: col_end, :] = filter_img

plt.figure(figsize=(20, 20))
results = np.clip(results, 0, 255)
plt.imshow(results)
plt.show()









































