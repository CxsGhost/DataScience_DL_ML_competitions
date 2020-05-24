from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from keras import activations
import matplotlib.pyplot as plt

# 设置数据预处理时的各种参数
max_feature = 10000
max_len = 500
batch_size = 32

# 加载数据, 内置数据已经转化为单词索引
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_feature)
print(input_train.shape, y_test.shape)

# 把序列张量化
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
print(input_test.shape)

# 建立含单层RNN的模型
network = models.Sequential()
network.add(layers.Embedding(max_feature, 32))
# 需不需要返回整个序列，只需要最后一个即可(前面所有的信息集中到了最后一个序列中
network.add(layers.SimpleRNN(32, activation=activations.tanh, return_sequences=False))
network.add(layers.Dense(1, activation=activations.sigmoid))

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

print(network.summary())

history = network.fit(x=input_train, y=y_train,
                      batch_size=128,
                      epochs=10,
                      validation_split=0.2)

# 绘制训练曲线
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

































