import tensorflow as tf

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 这次保留分类器
network = VGG16(weights='imagenet',
                include_top=True)
print(network.summary())
print(network.output)

img_path = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/CNN的可视化/creative_commons_elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

# 使用VGG16的预处理操作
x = preprocess_input(x)

# 进行预测，并对输出进行编码，使得可读懂
predict = network.predict(x)
# 查看预测类的编号
print(np.argmax(predict))
predict = decode_predictions(predict, top=3)
print(predict)


# 展示哪些部分最像非洲象，使用Grad-CAM

# 取负责非洲象的输出节点的输出(keras中一切默认批量输入和批量输出，所以第一维是输出数量，这里实际就是1）
african_elephant = network.output[:, 386]

# 取出最后卷积块中的最后卷积层
last_conv_layer = network.get_layer('block5_conv3')  # block5_conv3 (Conv2D)        (None, 14, 14, 512)

# 计算非洲象输出节点对于最后卷积层的梯度（none，14,14,512）
grads = K.gradients(african_elephant, last_conv_layer.output)[0]


pooled_grads = K.mean(grads, axis=(0, 1, 2))  # 形状为（512，）的向量，每个元素是特征图不同通道的梯度平均大小

# 访问刚刚定义的量，对于给定的输入，特征图的梯度和输出特征图
iterate = K.function([network.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads[i]

# 合并输出特征图的通道，每个像素点取各通道的平均值
class_heat_map = np.mean(conv_layer_output_value, axis=-1)

# 为了便于可视化，还要标准化

# 去除小于0的值
class_heat_map = np.maximum(class_heat_map, 0)
class_heat_map /= np.max(class_heat_map)
plt.matshow(class_heat_map)
plt.show()


# 最后来生成一张图，将原始图像叠加到热力图
img = cv2.imread('creative_commons_elephant.jpg')

# 扩大热力图与原图相同
class_heat_map = cv2.resize(class_heat_map, (img.shape[1], img.shape[0]))

# 之前将热力图标准化到0,1,。现在变回RGB
class_heat_map = np.uint8(255 * class_heat_map)

# 将热力图应用于原始图像，0.4是热力图强度因子
class_heat_map = cv2.applyColorMap(class_heat_map, cv2.COLORMAP_JET)
superimposed_img = class_heat_map * 0.4 + img

# 保存图像
cv2.imwrite('elephant_heat_map.jpg',
            superimposed_img)


















