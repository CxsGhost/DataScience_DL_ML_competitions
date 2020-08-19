from keras import models
from keras import layers
from keras import Input
from keras import activations
from keras import losses
from keras import optimizers
from keras import metrics
from keras import regularizers
from keras.utils import to_categorical

# 确定两个输入文本序列词典容量
text_vocabulary_size = 10000
question_vocabulary_size = 10000

# 输出文本词的词典大小
answer_vocabulary_size = 500

# 创建第一个输入, 是一个长度可变的整数序列，可以对输入命名
text_input = Input(shape=(None, ),
                   dtype='int32',
                   name='text')

# 创建词嵌入层
embedded_text = layers.Embedding(text_vocabulary_size,
                                 64)(text_input)

# 输入LSTM层
encoded_text = layers.LSTM(32,
                           dropout=0.1,
                           return_sequences=False)(embedded_text)

# 第二个输入同样
question_input = Input(shape=(None,),
                       dtype='int32',
                       name='question')
embedded_question = layers.Embedding(question_vocabulary_size,
                                     32)(question_input)
encoded_question = layers.LSTM(16,
                               dropout=0.1,
                               return_sequences=False)(embedded_question)

# 合并两个输入
concatenated = layers.Concatenate(axis=-1)([encoded_text, encoded_question])

answer = layers.Dense(answer_vocabulary_size,
                      activation=activations.softmax,
                      kernel_regularizer=regularizers.l2(l=0.01))(concatenated)

network = models.Model(inputs=[text_input, question_input],
                       outputs=answer)

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.categorical_crossentropy,
                metrics=[metrics.categorical_accuracy])


import numpy as np

# 设置样本容量和数据长度
num_samples = 1000
max_length = 100

# 随机生成伪数据
text = np.random.randint(low=1, high=text_vocabulary_size,
                         size=(num_samples, max_length))
question = np.random.randint(low=1, high=question_vocabulary_size,
                             size=(num_samples, max_length))
answers = np.random.randint(low=1, high=answer_vocabulary_size,
                            size=(num_samples, ))

# 转换成one_hot编码
answers = to_categorical(answers, num_classes=answer_vocabulary_size)

# 第一种输入方式，按顺序的列表
network.fit(x=[text, question], y=answers,
            epochs=5,
            batch_size=128)

# 第二种输入方式，根据input类的名称写成对应字典
network.fit(x={'text': text, 'question': question}, y=answers,
            epochs=5,
            batch_size=128)






























































