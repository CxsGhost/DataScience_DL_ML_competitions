# keras中设置按需分配
import tensorflow as tf

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
# 加载预训练的inception
from keras.applications import inception_v3
from keras import backend as K

# 我们不需要训练，这个命令能够禁用所有与训练相关的操作
K.set_learning_phase(0)

# 加载模型，只需要预训练的卷积部分
network = inception_v3.InceptionV3(weights='imagenet',
                                   include_top=False)

print(network.summary())

# 随意的选择了4层，并硬编码权重
layer_contributions = {'mixed1': 0.9,
                       'mixed10': 3.0,
                       'mixed5': 2.0,
                       'mixed9': 1.5}


# 定义一个包含损失的张量，损失就是上面层中激活L2范数的加权

# 创建一个字典，将层的名称映射为层的实例
layer_dict = {layer.name: layer for layer in network.layers}

# 在定义损失时将层的贡献添加到这个标量变量中
loss = K.variable(0.0)
for layer_name in layer_contributions.keys():
    coeff = layer_contributions[layer_name]

    # 获取层的输出
    activation = layer_dict[layer_name].output

    # 将本层输出所有的张量进行乘积并加和（cast转换类型，prod是乘积）
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    # 为了避免边界伪影，剪裁掉边界，然后平方加和并标准化，然后乘对应的权重加到
    loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling


# 设置梯度上升过程

# 用于保存生成的图像，即梦境图像
dream = network.input

# 计算梯度（返回一个张量列表，因为只有一个，直接取出来即可
grads = K.gradients(loss, dream)[0]

# 将梯度标准化，重要技巧！！！
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# 给定一张输出图像，设置一个keras函数获取损失值和梯度值
fetch_loss_and_grads = K.function(inputs=[dream], outputs=[loss, grads])


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_value = outs[1]
    return loss_value, grad_value


# 梯度上升迭代函数
def gradient_ascent(x, iteration_, step_, max_loss_=None):
    for i in range(iteration_):
        loss_value, grad_value = eval_loss_and_grads(x)

        # 因为是梯度上升，要设置一个迭代的阈值
        if max_loss_ is not None and loss_value > max_loss_:
            break

        print('....Loss value at {} : {}'.format(i, loss_value))
        x += step_ * grad_value
    return x


# 图像处理辅助函数
import scipy
import imageio
import numpy as np
from keras.preprocessing import image


def resize_img(img_, size):
    img_ = np.copy(img_)
    factors = (1,
               float(size[0]) / img_.shape[1],
               float(size[1]) / img_.shape[2],
               1)
    return scipy.ndimage.zoom(img_, factors, order=1)


def save_img(img_, file_name):
    pil_img = deprocess_img(np.copy(img_))
    imageio.imsave(file_name, pil_img)


# 通用函数，打开图像，改变图像大小，以及将图像格式转化为inception v3可处理的张量
def preprocess_image(image_path):
    img_ = image.load_img(image_path)
    img_ = image.img_to_array(img_)

    # 将图像转化为批量形式，以便输入inception v3
    img_ = np.expand_dims(img_, axis=0)

    img_ = inception_v3.preprocess_input(img_)
    return img_


# 通用函数，将一个张量转化为有效图像(从批量中提取出，并且转化为channel_last
def deprocess_img(x):
    if K.image_data_format() == 'channel_first':
        x = np.reshape(x, newshape=(3, x.shape[2], x.shape[3]))
        x = np.transpose(x, (1, 2, 0))
    else:
        x = np.reshape(x, newshape=(x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255

    # 裁剪图像数组的范围至0到255之内
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 多个连续尺度运行梯度上升（详见238）

# 学习率
step = 0.01

# 运行梯度上升尺度个数
num_octave = 3

# 每个尺度之间的大小比例
octave_scale = 1.4

# 最大迭代次数
iteration = 20

# 设置一个阈值，以免图像处理过度
max_loss = 10

base_image_path = 'E:/py/科学计算与机器学习/Keras/生成式深度学习（GAN）/DeepDream/iron_man.jpg'

img = preprocess_image(base_image_path)

# 预处理完成后，取出图像真正的尺寸
original_shape = img.shape[1: 3]

# 准备一个形状元组组成的列表，它定义了运行梯度上升的不同尺度(由大到小
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

# 将列表形状反转，变为升序
successive_shapes.reverse()

# 将原始图像缩到最小尺寸
original_img = np.copy(img)
shrunk_original_img = resize_img(original_img, size=successive_shapes[0])

for shape in successive_shapes:
    print('Process image shape: {}'.format(shape))
    img = resize_img(img, size=shape)

    # 先对当前尺度进行变换（梯度上升
    img = gradient_ascent(img, iteration_=iteration, step_=step, max_loss_=max_loss)

    # 然后将原始小图转化为相同的尺度，他会像素化(这是一个放大的过程
    upscale_shrunk_original_img = resize_img(shrunk_original_img, size=shape)

    # 在这尺寸上计算原始图像的高质量版本（这是一个缩小的过程
    same_size_original = resize_img(original_img, size=shape)

    # 二者的差别就是在放大过程中损失的细节
    lost_detail = same_size_original - upscale_shrunk_original_img

    # 补充差别
    img += lost_detail

    # 更新缩小图，用于下个尺度
    shrunk_original_img = resize_img(original_img, size=shape)

    save_img(img, file_name='dream_at_scale_{}.png'.format(shape))

save_img(img, file_name='final_dream.png')








