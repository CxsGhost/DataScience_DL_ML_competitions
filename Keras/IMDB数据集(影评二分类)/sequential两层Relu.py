"""
整个数据集与sklearn中CountVectorizer十分相似，拥有一个词汇表字典
"""
from keras.datasets import imdb
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from keras import regularizers  # 正则化模块
import matplotlib.pyplot as plt
import numpy as np


# num_words是保留数据中前10000个最常出现的词，这样向量数据不会太大，便于处理
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(test_data.shape, train_labels.shape)
print(train_labels)  # <class 'numpy.ndarray'>

# labels是0或1，负面与正面。input数据每条长度不一样，由单词索引组成
print(train_data[0])
print(train_labels[0])

# 下面的代码可以把数字向量转化为单词向量，因为只取了10000词，所以会有缺失，用？标注
word_index = imdb.get_word_index()
reverse_word = dict((value, key) for key, value in word_index.items())
decode_review = ' '.join([reverse_word.get(i, '?') for i in train_data[0]])
print(decode_review)


# 进行数据预处理，转化为10000长度的one-hot表示
def vectorizer_sequences(sequences, dimension=10000):
    result = np.zeros(shape=(sequences.shape[0], dimension))
# enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据索引和数据
    for i, sequences in enumerate(sequences):
        result[i, sequences] = 1
    return result


train_x = vectorizer_sequences(train_data)
test_x = vectorizer_sequences(test_data)
train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

# 输入是向量，标签是0,1，relu的全连接激活层表现很好
model = models.Sequential()
model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.02), activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.02), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 二元分类问题，且输出是概率，那二元交叉熵损失函数是很好的选择。各个参数如果不使用默认，可以自定义
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])  # 选择评估器的时候一定要注意，指明是二分类，普通accuracy会当做回归来评测！

# 留出一部分验证数据集,训练时会用到监控损失和准确率
val_x = train_x[: 10000]
train_x = train_x[10000:]

val_y = train_labels[: 10000]
train_labels = train_labels[10000:]

# 开始训练，并保留训练过程的数据,每epoch都会计算一次损失和正确率
history = model.fit(x=train_x, y=train_labels,
                    epochs=5,
                    batch_size=512,
                    validation_data=(val_x, val_y))

# history对象中有一个成员history字典，保存了训练过程中所有的数据
history = history.history
print(history.keys())

loss_values = history['loss']
val_loss_values = history['val_loss']
accuracy = history['binary_accuracy']
val_accuracy = history['val_binary_accuracy']
epochs = range(1, len(loss_values) + 1)

# 绘制loss变化趋势图
ax_1 = plt.figure(figsize=(9, 9)).gca()
ax_1.plot(epochs, loss_values, color='b', marker='o', linestyle='-', label='Loss_Value')
ax_1.plot(epochs, val_loss_values, color='r', marker='>', linestyle='--', label='Val_Loss_Value')
ax_1.set_title('Train and Validation Loss')
ax_1.set_xlabel('epoch')
ax_1.set_ylabel('loss')
ax_1.legend()

# 绘制正确率趋势图
ax_2 = plt.figure(figsize=(9, 9)).gca()
ax_2.plot(epochs, accuracy, color='b', marker='*', linestyle='-.', label='Accuracy')
ax_2.plot(epochs, val_accuracy, color='r', marker='<', linestyle=':', label='Val_accuracy')
ax_2.set_title('Accuracy')
ax_2.set_xlabel('epoch')
ax_2.set_ylabel('accuracy')
ax_2.legend()

plt.show()

# 使用测试集,可以用模型的predict方法进行实际应用的预测
test_loss, test_accuracy = model.evaluate(test_x, test_labels)
predict = model.predict(test_x)
print(predict)
print("测试集结果：loss:{},acc:{}".format(test_loss, test_accuracy))




