import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

imdb_dir = 'E:/py/科学计算与机器学习/Keras/自然语言及序列处理（RNN、LSTM、GRU、1DConv）/IMDB影评二分类/IMDB原始数据集/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

texts = []
labels = []

for class_file_name in ['neg', 'pos']:
    data_file_name = os.path.join(train_dir, class_file_name)
    for each_data in os.listdir(data_file_name):
        each_data_file = open(os.path.join(data_file_name, each_data),
                              encoding='utf-8')
        data = each_data_file.read()
        each_data_file.close()
        texts.append(data)
        if class_file_name == 'neg':
            labels.append(0)
        else:
            labels.append(1)


max_len = 100
max_words = 10000
train_samples = 200
validation_samples = 10000
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
all_data = tokenizer.texts_to_sequences(texts)
all_data = pad_sequences(all_data, maxlen=max_len)

labels = np.asarray(labels)

indices = np.arange(len(labels))
np.random.shuffle(indices)

train_x = all_data[indices[: train_samples]]
train_y = labels[indices[: train_samples]]

val_x = all_data[indices[train_samples: train_samples + validation_samples]]
val_y = labels[indices[train_samples: train_samples + validation_samples]]


network = models.Sequential()
network.add(layers.Embedding(max_words, embedding_dim, input_length=max_len))
network.add(layers.Flatten())
network.add(layers.Dense(64, activation=activations.relu, kernel_regularizer=regularizers.l2()))  # l2要带括号
network.add(layers.Dense(1, activation=activations.sigmoid))
print(network.summary())

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

history = network.fit(x=train_x, y=train_y,
                      batch_size=16,
                      epochs=10,
                      validation_data=(val_x, val_y))

# 绘制曲线
accuracy = history.history['binary_accuracy']
val_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

# 绘制loss变化趋势图
ax_1 = plt.figure(figsize=(9, 9)).gca()
ax_1.plot(epochs, loss, color='b', marker='o', linestyle='-', label='Loss_Value')
ax_1.plot(epochs, val_loss, color='r', marker='>', linestyle='--', label='Val_Loss_Value')
ax_1.set_title('Train and Validation Loss')
ax_1.set_xlabel('epoch')
ax_1.set_ylabel('loss')
ax_1.legend()

# 绘制正确率趋势图
ax_2 = plt.figure(figsize=(len(accuracy), len(accuracy))).gca()
ax_2.plot(epochs, accuracy, color='b', marker='*', linestyle='-.', label='Accuracy')
ax_2.plot(epochs, val_accuracy, color='r', marker='<', linestyle=':', label='Val_accuracy')
ax_2.set_title('Accuracy')
ax_2.set_xlabel('epoch')
ax_2.set_ylabel('accuracy')
ax_2.legend()

plt.show()






















