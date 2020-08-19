from keras.datasets import imdb
from keras import models
from keras import layers
from keras import Input
from keras import activations
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.preprocessing import sequence
from keras import callbacks
import numpy as np


num_words = 10000
max_len = 100

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

print(max(train_data))


# 进行数据预处理，转化为10000长度的one-hot表示
def vectorizer_sequences(sequences, dimension=10000):
    result = np.zeros(shape=(sequences.shape[0], dimension))
# enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据索引和数据
    for i, sequences in enumerate(sequences):
        result[i, sequences] = 1
    return result


train_data = vectorizer_sequences(train_data)
test_data = vectorizer_sequences(test_data)
# train_data = sequence.pad_sequences(train_data, maxlen=max_len, value=0)
# test_data = sequence.pad_sequences(test_data, maxlen=max_len, value=0)


train_text = Input(shape=(train_data.shape[-1], ),
                   dtype='float32',
                   name='train_text')
dense1 = layers.Dense(128, activation=activations.relu,
                      kernel_regularizer=regularizers.l2(l=0.01))(train_text)
dense2 = layers.Dense(64, activation=activations.relu)(dense1)
dense3 = layers.Dense(32, activation=activations.relu)(dense2)
output = layers.Dense(1, activation=activations.sigmoid)(dense3)

network = models.Model(inputs=train_text, outputs=output)

print(network.summary())

# 接下来我们使用回调函数监控训练
callback_list = [callbacks.EarlyStopping(monitor='binary_accuracy',  # 监控指标
                                         patience=2),  # 多于2轮不改善则提前终止
                 callbacks.ModelCheckpoint(
                    filepath='E:/py/科学计算与机器学习/Keras/函数式API/回调函数和Tensorboard/my_model.h5',  # 模型保存的路径
                    monitor='val_loss',  # 如果val_loss有改善，则保存模型参数
                    save_best_only=True)]  # 仅仅保存最优模型

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

network.fit(x=train_data, y=train_labels,
            batch_size=128,
            epochs=15,
            callbacks=callback_list,
            validation_split=0.2,
            verbose=1)














































































