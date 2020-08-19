# 定义初始变量
from keras.preprocessing.image import load_img, img_to_array

target_img_path = 'E:/py/科学计算与机器学习/Keras/生成式深度学习（GAN）/神经风格迁移/target_img.jpg'
style_img_path = 'E:/py/科学计算与机器学习/Keras/生成式深度学习（GAN）/神经风格迁移/style_img.jpg'

width, height = load_img(target_img_path).size

# 等比例调整
img_height = 400
img_width = int(width * img_height / height)


# 一些辅助函数，对VGG19的图像进行加载，预处理，后处理
import numpy as np
from keras.applications import vgg19


def preprocess_image(image_path):

    # 将所有图像统一尺寸
    img_ = load_img(image_path, target_size=(img_height, img_width))
    img_ = img_to_array(img_)
    img_ = np.expand_dims(img_, axis=0)
    img_ = vgg19.preprocess_input(img_)
    return img_


def deprocess_image(x):

    # vgg19的预处理是减去imageNet的平均像素值，使其中心为0，这里是逆操作
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 将图像由BGR转换为RGB，这也是预处理逆操作做的一部分
    x = x[:, :, :: -1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 构建VGG19网络
from keras import backend as K

# 风格参考图像和目标内容图像是固定的，用constant定义、
target_image = K.constant(preprocess_image(target_img_path))
style_reference_image = K.constant(preprocess_image(style_img_path))

# 网络接收三张图的张量，为生成图像申请一个占位符，大小与target_image相同
combination_image = K.placeholder(shape=(1, img_height, img_width, 3))

# 创建输入张量（记住此处顺序，很重要，后期要用）
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

network = vgg19.VGG19(weights='imagenet',
                      include_top=False,
                      input_tensor=input_tensor)
print('------Model loaded-------')


# 定义content_loss，保证generate-image与target-image在顶层卷积有尽量相似的激活结果
def content_loss(base, combination):
    return K.sum(K.square(base - combination))


# 风格损失，使用一个辅助函数来计算输入矩阵的格拉姆矩阵(表示一层输出特征图特征之间的相互关系（统计规律）（也就是纹理））
def gram_matrix(x):

    # permute_dim是重新排列张量的轴（Keras通道在后，所以要移到前面），batch_flatten是批量展平张量（展成2维矩阵）
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    size = img_height * img_width

    # 二者损失依然是用L2范数表示，除的式子应该是防止loss过大
    return K.sum(K.square(s - c) / (4.0 * (channels ** 2) * (size ** 2)))


# 还有第三个损失，总变差损失，对组合图像进行操作，防止图像过度像素化（使其具有更好的连续性），可以理解为正则化loss
def total_variation_loss(x):
    a = K.square(x[:, : img_height - 1, : img_width - 1, :] -
                 x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, : img_height - 1, : img_width - 1, :] -
                 x[:, : img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 定义最终损失
outputs_dict = {layer.name: layer.output for layer in network.layers}

# 用于内容损失的层
content_layer = 'block5_conv2'

# 用于风格损失的层
style_layers = ['block{}_conv1'.format(i) for i in range(1, 5)]

# 各个损失的权重
total_variation_weight = 1e-4
style_weight = 1.0
content_weight = 0.025

loss = K.variable(0.0)

# 取出该层的激活
layer_feature = outputs_dict[content_layer]

# 根据之前输入批量的顺序，取出target的激活和combination的激活，计算并添加内容损失
target_image_feature = layer_feature[0, :, :, :]
combination_image_feature = layer_feature[2, :, :, :]
loss = loss + content_weight * content_loss(target_image_feature, combination_image_feature)

# 计算并添加风格损失
for layer in style_layers:
    layer_feature = outputs_dict[layer]
    style_feature = layer_feature[1, :, :, :]
    combination_image_feature = layer_feature[2, :, :, :]
    loss = loss + style_weight * style_loss(style_feature, combination_image_feature)

# 计算并添加总变差损失。合成图之前用placeholder申请的，并且也是上一次更新后的结果，故不应该用本次激活
loss = loss + total_variation_weight * total_variation_loss(combination_image)


# Lbfgs内置于scipy中，但是只能用于展平向量，但分别计算loss和grad效率是很低的
# 于是创建一个类，可同时计算loss和grad，在第一次调用时会返回loss，缓存梯度值用于下一次调用

grads = K.gradients(loss, combination_image)[0]

fetch_loss_and_grads = K.function(inputs=[combination_image], outputs=[loss, grads])


# 以下类将上面的函数包装起来，可以让你用两个单独方法调用来获取grads和loss，这是我们要使用的scipy优化器所求的
class Evaluator:

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x_):
        assert self.loss_value is None
        x_ = x_.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x_])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value

        # 这里先保存grads的值
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x_):

        # 这里读取grads的值，并返回，这样就避免了二次计算
        assert self.grads_values is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values


evaluator = Evaluator()


# 使用scipy中的L-BFGS算法来运行梯度上升，每一次迭代都保存当前图像
from scipy.optimize import fmin_l_bfgs_b
from imageio import imsave
import time

result_prefix = 'my_result'
iteration = 20

# 读取图片并展平，因为scipy的优化器只能处理展平矩阵
x = preprocess_image(target_img_path)
x = x.flatten()
for i in range(iteration):
    print('Start of iteration {}'.format(i))
    start_time = time.time()

    # 定义类的原因就在这，loss和gradient必须单独作为方法传入
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value is {}'.format(min_val))
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    file_name = result_prefix + '_at_iteration_{}.png'.format(i)
    imsave(file_name, img)
    print('Image saved as file_name: {}'.format(file_name))
    end_time = time.time()
    print('Iteration {} completed in {}'.format(i, - start_time + end_time))



































