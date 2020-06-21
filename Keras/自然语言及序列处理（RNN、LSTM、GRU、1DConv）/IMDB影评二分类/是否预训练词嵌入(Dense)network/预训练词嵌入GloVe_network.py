"""
词典的0索引保留，不代表任何词
如果指定oov_token，则超出（num_words - 1）的词会替换成这个词，不设置则直接忽略超过词
设置的这个词，如果不在词典中，则被添加至词典并且取代最大词频的词成为1，其他词索引依次向后推一位
如果在词典中，那么其他词依然会向后推一位，然后超出词被替换成该词索引，无论索引多大（真傻逼）
然后根据变换后的词表取前num_words个，进行编码
当然，对测试集编码要用同一个tokenizer，对于测试集中的新词，会直接忽略（详见源码）
不得不说这个机制真的sb
"""
import os

imdb_dir = 'E:/py/科学计算与机器学习/Keras/自然语言及序列处理（RNN、LSTM、GRU、1DConv）/IMDB影评二分类/IMDB原始数据集/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

# 读取数据和标签，把标签转化为one-hot
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for file_name in os.listdir(dir_name):
        if file_name[-4:] == '.txt':
            file_ = open(os.path.join(dir_name, file_name), encoding='utf-8')
            texts.append(file_.read())
            file_.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

"""
预训练的词嵌入对数据较少的问题表现很好（对于大量数据，针对任务的词嵌入会更好）
于是我们限制训练数据200个
"""
from keras.preprocessing.text import Tokenizer  # tokenizer获得的词典的索引，是根据词频来排序的，索引1的词频最大
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# 每条评论限制长度100个词
max_len = 100
# 训练数据200
training_samples = 200
# 验证数据10000
validation_samples = 10000
# 限制词典中前10000个最常出现的单词
max_words = 10000

tokenizer = Tokenizer(num_words=max_words, oov_token=None)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found {} unique tokens'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=max_len)

# asarray不会像array一样copy，而是原地转换
labels = np.asarray(labels)
print('shape of data tensor:{}'.format(data.shape))
print('shape of labels tensor:{}'.format(labels.shape))

# 将数据划分为训练集验证集，一开始是排好顺序的（读取时先读的neg后pos）
indices = np.arange(data.shape[0])
# 打乱数据的索引（相当于打乱数据
np.random.shuffle(indices)
# 用打乱的索引重新赋值data和labels
data = data[indices]
labels = labels[indices]

# 划分训练集和验证集
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# 下面进行glove词嵌入
glove_dir = 'E:/py/科学计算与机器学习/Keras/自然语言及序列处理（RNN、LSTM、GRU、1DConv）/glove.6B'

embeddings_index = {}

# 文件中每一行是一个单词对应着其向量，单词在0索引位置
file_ = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')

for line in file_:
    values = line.split()

    # 取出单词
    word = values[0]

    # 把向量数字字符串列表转化为一个向量
    coefs = np.asarray(values[1:], dtype='float32')

    # 单词及对应向量存入字典
    embeddings_index[word] = coefs

file_.close()

print('Found {} word vectors.'.format(len(embeddings_index)))

# 接下来构建一个可加载到embedding层中的嵌入矩阵，必须是形状为（max_words，embeddings_index）
# 读取的是100d，每个词向量维度100
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():

    # 限制最常出现前10000个词（实际在向量化时候，会保留1000 - 1），索引大于等于999的要舍弃
    if i < max_words:

        # 词很多，不一定保证这个词嵌入字典中一定有我们想要的所有单词，所以没有的单词，直接跳过，默认为0向量即可
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


from keras import layers
from keras import models
from keras import losses
from keras import metrics
from keras import optimizers
from keras import activations
import matplotlib.pyplot as plt

network = models.Sequential()
network.add(layers.Embedding(max_words, embedding_dim, input_length=max_len))
network.add(layers.Flatten())
network.add(layers.Dense(128, activation=activations.relu))
network.add(layers.Dense(1, activation=activations.sigmoid))
print(network.summary())

# 在模型中加载GloVe词嵌入
# Embedding层只有一个二维浮点数权重矩阵，每个i元素是与索引i对应的嵌入向量

# 用准备好的词嵌入矩阵替代未训练的矩阵（权重）
network.layers[0].set_weights([embedding_matrix])

# 并且冻结这一层不需要再训练
network.layers[0].trainable = False

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])
history = network.fit(x=x_train, y=y_train,
                      batch_size=32,
                      epochs=10,
                      validation_data=(x_val, y_val))

# 保存模型的权重
network.save_weights('pre_trainable_glove_model.h5')

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

# 因为仅有200个训练样本，所以很少的epoch就有严重的过拟合，训练数据acc达到100%，而测试集50%








