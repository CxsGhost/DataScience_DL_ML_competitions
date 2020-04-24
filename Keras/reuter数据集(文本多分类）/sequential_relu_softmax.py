"""
共有46个不同的主题，主题样本数量不一致，但至少有10个
"""
from keras.datasets import reuters
from keras import models
from keras import layers
from keras import metrics
from keras import losses
from keras import optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(train_data.shape, test_labels.shape)

# 可以把文本向量转为文本查看
word_index = reuters.get_word_index()
reverse_word_index = dict((value, key) for key, value in word_index.items())
decode_content = ' '.join(reverse_word_index.get(i, '?') for i in train_data[0])
print(decode_content)


# 预处理，转化为one-hot向量表示(输入和标签）
def vectorizer_sequence(sequences, dimension=10000):
    result = np.zeros(shape=(sequences.shape[0], dimension))
    for j, sequence in enumerate(sequences):
        result[j, sequence] = 1
    return result

# Keras内置也可以实现
# def labels_one_hot(sequences, dimension=46):
#     result = np.zeros(shape=(sequences.shape[0], dimension))
#     for j, label in enumerate(sequences):
#         result[j, label] = 1
#     return result


train_data = vectorizer_sequence(train_data)
test_data = vectorizer_sequence(test_data)

#
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 建立模型,并编译
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer=optimizers.rmsprop(lr=0.001),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

# 留出验证集
val_quantity = int(train_data.shape[0] * 0.15)

val_data = train_data[:val_quantity]
train_data = train_data[val_quantity:]

val_labels = train_labels[:val_quantity]
train_labels = train_labels[val_quantity:]

# 训练模型，监控训练过程
history = model.fit(x=train_data, y=train_labels,
                    batch_size=256,
                    epochs=9,
                    validation_data=(val_data, val_labels))

# 解析模型的情况
history = history.history

loss = history['loss']
val_loss = history['val_loss']

accuracy = history['categorical_accuracy']
val_accuracy = history['val_categorical_accuracy']

epoch = range(1, len(loss) + 1)

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

test_loss, test_acc = model.evaluate(test_data, test_labels)
print("测试集loss：{}  准确率：{}".format(test_loss, test_acc))




















































