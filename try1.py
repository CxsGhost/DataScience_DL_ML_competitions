# import os
#
# _dir = 'C:/Users/16591/Desktop/寒假线上教学/大学体育'
#
# for i in range(1, 15):
#     dir_ = os.path.join(_dir, '{}'.format(i))
#     os.mkdir(dir_)

from keras import Input
from keras import models
from keras import layers
from keras import losses

_input = Input(shape=(None, ),
               dtype='int32',
               name='ceshi')
s1 = layers.Dense(1, activation='sigmoid')(_input)

i = [[1, 2, 3, 4], [1, 2, 3]]

m = models.Model(inputs=_input, outputs=s1)
m.compile(optimizer='rmsprop',
          loss=losses.binary_crossentropy)

print(m.summary())
out = m.layers




