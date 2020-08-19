"""
比如可以有多个输入分支，但是他们共享相同的权重
进行相同的计算，相同的表示
比如两个句子的相似度，如果设置两个输入分支，分别走入两个不同的LSTM就是不合理的
因为句子相似是对称的，设置两个处理显然不合理，极有可能过拟合
所以两个输入分支应该走过同一LSTM，然后进行相似度判断
"""

from keras import models
from keras import Input
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np


left_input = Input(shape=(None, 128),
                   name='left')
right_input = Input(shape=(None, 128),
                    name='right')

lstm = layers.LSTM(32)

left_output = lstm(left_input)
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output], axis=-1)
prediction = layers.Dense(1, activation=activations.sigmoid)(merged)

network = models.Model(inputs=[left_input, right_input], outputs=prediction)

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

print(network.summary())


left = np.random.randint(low=1, high=100, size=(100, 10, 128))
right = np.random.randint(low=1, high=100, size=(100, 10, 128))
labels = (np.random.normal(loc=0, scale=1, size=(100, )) > 0).astype(np.int)


network.fit(x={'left': left,
               'right': right},
            y=labels,
            epochs=2,
            batch_size=16)





































