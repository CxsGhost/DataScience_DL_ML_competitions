"""
有时候我们不满足于仅仅共享层
我们还可以多输入共享模型
"""
from keras import Input
from keras import applications
from keras import layers

# 这里不使用ImageNet训练好的权重，只调用架构
x_inception = applications.Xception(weights=None,
                                    include_top=False)

left_input = Input(shape=(250, 250, 3),
                   dtype='float32',
                   name='left')
right_input = Input(shape=(250, 250, 3),
                    dtype='float32',
                    name='right')

# 把不同分支输入放进同一模型
left_output = x_inception(left_input)
right_output = x_inception(right_input)

output = layers.concatenate([left_input, right_input])







































