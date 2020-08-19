"""
训练集包含25000张，猫狗各一半，我们不需要这么大的数据集
前12500是猫，后面是狗
我们的训练集：猫狗各1000
验证集：猫狗各500
测试集：猫狗各500
"""
import os
import shutil

# 指定原数据集目录和创建的数据集目录
original_dataset_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats/train/train'

base_dir = 'E:/py/科学计算与机器学习/Keras/计算机视觉（CNN）/cats-vs-dogs数据集/dogs-vs-cats_small'
os.mkdir(base_dir)


# 划分对应的训练测试验证集的目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 在三个集下分别划分猫狗数据的目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)


# 将猫的图像归入各个数据集
file_names = ['cat.{}.jpg'.format(i) for i in range(1000)]
for file_name in file_names:
    src = os.path.join(original_dataset_dir, file_name)
    dst = os.path.join(train_cats_dir, file_name)
    shutil.copy(src=src, dst=dst)

file_name = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for file_name in file_names:
    src = os.path.join(original_dataset_dir, file_name)
    dst = os.path.join(validation_cats_dir, file_name)
    shutil.copy(src, dst)

file_names = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for file_name in file_names:
    src = os.path.join(original_dataset_dir, file_name)
    dst = os.path.join(test_cats_dir, file_name)
    shutil.copy(src, dst)

# 将狗的图像归入各个数据集
file_names = ['dog.{}.jpg'.format(i) for i in range(1000)]
for file_name in file_names:
    src = os.path.join(original_dataset_dir, file_name)
    dst = os.path.join(train_dogs_dir, file_name)
    shutil.copy(src, dst)

file_names = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for file_name in file_names:
    src = os.path.join(original_dataset_dir, file_name)
    dst = os.path.join(validation_dogs_dir, file_name)
    shutil.copy(src, dst)

file_names = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for file_name in file_names:
    src = os.path.join(original_dataset_dir, file_name)
    dst = os.path.join(test_dogs_dir, file_name)
    shutil.copy(src, dst)






























