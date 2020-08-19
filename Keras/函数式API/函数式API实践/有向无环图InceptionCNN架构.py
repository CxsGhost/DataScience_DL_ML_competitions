"""
Inception特色是多个分支中，有进行1x1卷积的分之，也称为逐点卷积
逐点卷积相当于输入特征图经过一个Dense层，把通道间的信息融合，而不影响空间特征
这有利于网络分别学习空间特征和通道特征
最后分支合并时，要注意具有相同的尺寸
"""


from keras import layers
from keras import activations
from keras import Input

x = Input(shape=(299, 299, 3),
          dtype='float32',
          name='pic')

branch_a = layers.Conv2D(128, 1, strides=2, activation=activations.relu)(x)

branch_b = layers.Conv2D(128, 1, strides=1, activation=activations.relu)(x)
branch_b = layers.Conv2D(128, 3, strides=2, activation=activations.relu)(branch_b)

# 这个分支用到了平均池化
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation=activations.relu)(branch_c)

branch_d = layers.Conv2D(128, 1, activation=activations.relu)(x)
branch_d = layers.Conv2D(128, 3, activation=activations.relu)(branch_d)
branch_d = layers.Conv2D(128, 3, strides=2, activation=activations.relu)(branch_d)

# 将它们拼接(这些只是伪代码
output = layers.concatenate([branch_a,
                             branch_b,
                             branch_c,
                             branch_d],
                            axis=-1)




























































































