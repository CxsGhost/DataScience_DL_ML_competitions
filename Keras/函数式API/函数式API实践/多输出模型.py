from keras import models
from keras import Input
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import activations
from keras.utils import to_categorical

vocabulary_size = 50000

# 收入水平分类的种类
num_income_groups = 10

posts_inputs = Input(shape=(500, ),
                     dtype='int32',
                     name='posts')

embedding_posts = layers.Embedding(vocabulary_size, 256)(posts_inputs)
x = layers.Conv1D(128, 5, activation=activations.relu)(embedding_posts)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation=activations.relu)(x)
x = layers.Conv1D(256, 5, activation=activations.relu)(x)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation=activations.relu)(x)
x = layers.Conv1D(256, 5, activation=activations.relu)(x)

# 一个全局最大池化，得到一个最终序列
x = layers.GlobalMaxPool1D()(x)

x = layers.Dense(128, activation=activations.relu)(x)

"该模型有三个不同类型的输出e"

# age作为回归，不需要激活函数
age_prediction = layers.Dense(1, name='age')(x)

# 收入水平预测，多分类
income_prediction = layers.Dense(num_income_groups,
                                 activation=activations.softmax,
                                 name='income')(x)
# 性别预测，二分类
gender_prediction = layers.Dense(1, activation=activations.sigmoid, name='gender')(x)

network = models.Model(inputs=posts_inputs,
                       outputs=[age_prediction,
                                income_prediction,
                                gender_prediction])

print(network.summary())

"""模型有三个输出，各不相同的损失函数，于是要合并三个损失有利于梯度下降
但每个损失函数标量范围不一样，可能导致只优化一个大的损失
于是我们可以手动给损失加权"""

# 给予多个损失函数(或监控指标)时，可以用按顺序的列表，也可以用字典和输出层的名字一一对应（建议用字典)
# mae通常在3-5，而交叉熵可能低至1以下，于是加权时要衡量利弊
network.compile(optimizer=optimizers.RMSprop(),
                loss={'age': losses.mae,
                      'gender': losses.binary_crossentropy,
                      'income': losses.categorical_crossentropy},
                loss_weights={'age': 0.25,
                              'gender': 1.0,
                              'income': 10.0},
                metrics={'age': metrics.mae,
                         'gender': metrics.binary_accuracy,
                         'income': metrics.categorical_accuracy})

# 生成虚拟数据
import numpy as np

posts = np.random.randint(low=1, high=vocabulary_size, size=(1000, 500))

age_targets = np.random.randint(low=20, high=40, size=(1000, ))
gender_targets = (np.random.normal(loc=0, scale=1, size=(1000, )) > 0).astype(np.int)
income_targets = to_categorical(np.random.randint(low=0, high=10, size=(1000, )),
                                num_classes=10,
                                dtype='int32')

# 给予y时，也要对应输出层的名字用字典（顺序列表也可以）
network.fit(x=posts,
            y={'age': age_targets,
               'gender': gender_targets,
               'income': income_targets},
            batch_size=128,
            epochs=2,
            verbose=1)










































