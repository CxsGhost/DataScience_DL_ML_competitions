# 针对CV领域的防止过拟合方法：数据增强
# 主要思想是，利用多种能够生成可信图像的随机变换来增加样本。略有区别于mnist中对图像人为进行的旋转
# Keras中可以对generator读取的图像进行多次随机变换来实现

from keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# 设置数据增强,这里只是部分参数，详细参考keras文档
data_generator = image.ImageDataGenerator(rotation_range=40,  # 随机旋转图像的角度值
                                          width_shift_range=0.2,  # 水平方向的平移范围（相对于原来的比例
                                          height_shift_range=0.2,  # 垂直方向平移的范围
                                          shear_range=0.2,  # 随机错切变换的角度
                                          zoom_range=0.2,  # 图像随机缩放的范围
                                          horizontal_flip=True,  # 随机将一半的图像水平翻转，对于水平对称的图像是有意义的
                                          fill_mode='nearest')  # 对变换后图像创建的新像素的填充方法 （这里是最近邻）

# 随机显示几个增强后的训练图像
train_cats_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small/train/cats'

# 把训练数据中每个文件路径存入列表
file_names = [os.path.join(train_cats_dir, file_name) for file_name in os.listdir(train_cats_dir)]

image_path = file_names[3]  # 随机选一个文件

img = image.load_img(image_path, target_size=(150, 150))

# 转化为（150， 150， 3）的数组，数组输入generator才能进行变化
x = image.img_to_array(img)

# 形状为（1， 150， 150， 3）,因为下面的flow方法要求输入4维numpy（一批数据）
x = x.reshape((1, ) + x.shape)

# 生成随机变换后的图像批量
# 循环是无限的，需要在某个时刻终止
i = 0
for batch in data_generator.flow(x, batch_size=1):
    plt.figure(i)
    image_plot = plt.imshow(image.array_to_img(batch[0]))  # batch[0]是因为batch返回的是批量图像，我们需要提取一个
    i += 1
    if i % 4 == 0:
        break

plt.show()











