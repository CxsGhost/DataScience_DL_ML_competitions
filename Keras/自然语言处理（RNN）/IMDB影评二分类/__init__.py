"""
完成词嵌入后，输入dense将向量展平了
于是依旧忽略了句子本身的结构，例如：
this movie is a bomb(电影很垃圾)，this movie is the bomb（电影很棒）
但得到了相对还可以的正确率
毕竟只取了每个sample的前20词
"""
from keras.datasets import imdb
from keras import preprocessing
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as plt

# 限制10000个词，文本长度20
max_feature = 10000
max_len = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)
print(x_train.shape, y_test.shape)

# 将整数列表转化为（samples，max—len）的二维张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
print(x_train.shape)


network = models.Sequential()

# 指定最大输入长度，以便于后面将嵌入展平
network.add(layers.Embedding(max_feature, 8, input_length=max_len))
network.add(layers.Flatten())
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

print(network.summary())

history = network.fit(x=x_train, y=y_train,
                      epochs=15,
                      batch_size=32,
                      validation_split=0.2)  # 不用手动分验证集了，直接指定比例就可

accuracy = history.history['binary_accuracy']
val_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(1, len(accuracy) + 1)

# 绘制loss趋势图
ax_loss = plt.figure(figsize=(9, 9)).gca()
ax_loss.plot(epoch, loss, color='b', marker='>', linestyle='-', label='Loss')
ax_loss.plot(epoch, val_loss, color='r', marker='o', linestyle=':', label='Vla_Loss')
ax_loss.set_title('Loss')
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.legend()

# 绘制准确率趋势图
ax_acc = plt.figure(figsize=(9, 9)).gca()
ax_acc.plot(epoch, accuracy, color='b', marker='*', linestyle='--', label='Accuracy')
ax_acc.plot(epoch, val_accuracy, color='r', marker='o', linestyle='-.', label='Val_Accuracy')
ax_acc.set_title('Accuracy')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('accuracy')
ax_acc.legend()

plt.show()










































