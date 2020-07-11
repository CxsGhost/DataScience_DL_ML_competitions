# 准备数据
import os
import tensorflow.keras as keras
import numpy as np
import random
import sys

path = keras.utils.get_file('nietzsche.txt',
                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

# 读取数据并小写化
text = open(path, encoding='utf-8').read().lower()
print('第一条数据：{}'.format(text[0]))
print('corpus length:{}'.format(len(text)))

# 将字符序列向量化

# 提取60个字符组成序列
max_len = 60
# 每三个字符采取一个新序列
step = 3
# 保存所提取的序列
sentences = []
# 保存目标
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])

print('Number of sequences:{}'.format(len(sentences)))

# 统计语料中出现的字符,转换成字典
chars = sorted(list(set(text)))
print('Unique character:{}'.format(len(chars)))
chars_indices = {key: values for values, key in enumerate(chars)}

# 把数据转化为one-hot编码
print('Vectorization...')
x = np.zeros(shape=(len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros(shape=(len(next_chars), len(chars)), dtype=np.bool)
for j in range(len(sentences)):
    for char_index, char in enumerate(sentences[j]):
        x[j, char_index, chars_indices[char]] = 1
    y[j, chars_indices[next_chars[j]]] = 1

# 构建网络
input_text = keras.Input(shape=(max_len, len(chars)),
                         dtype='float32',
                         name='text_sequences')
lstm = keras.layers.LSTM(units=128, return_sequences=False)(input_text)
classifier = keras.layers.Dense(len(chars), activation=keras.activations.softmax)(lstm)
network = keras.models.Model(inputs=input_text, outputs=classifier)

# 查看模型概览
print(network.summary())
keras.utils.plot_model(network, show_shapes=True, show_layer_names=True, to_file='model.png')

network.compile(optimizer=keras.optimizers.RMSprop(lr=1e-2),
                loss=keras.losses.categorical_crossentropy)


# 训练语言模型并从中采样
# 1.给定目前已生成的文本（这是一个持续生成的模型，输出要做下一次输出的输入），预测下一个字符的分布概率
# 2.根据某个温度对概率分布重新加权
# 3.根据加权后的概率分布对下一个字符进行随机采样
# 4.将新字符添加到文本末尾

# 该函数可以重新加权概率分布，并抽取字符
def sampling(prediction_, temperature_=1.0):
    prediction_ = np.asarray(prediction_).astype(np.float64)
    prediction_ = np.log(prediction_) / temperature_
    exp_prediction = np.exp(prediction_)
    prediction_ = exp_prediction / np.sum(exp_prediction)

    # 从多项式分布里抽取结果，参数：试验一次，概率序列，返回size个与prediction长度相同的序列，表示在试验次数内每个结果出现的次数
    probabilistic = np.random.multinomial(1, prediction_, size=1)
    return np.argmax(probabilistic)


log_dir = 'tensorboard_log_nicai'
# os.mkdir(log_dir)
callback_list = [keras.callbacks.TensorBoard(log_dir=log_dir,
                                             histogram_freq=1)]

# 循环反复训练文本，每轮过后都用不同的温度值来生成文本。我们可以看到，随着模型收敛文本的变化，以及温度的影响
for epoch in range(1, 60):
    print('\nthe epoch:{}------------------------------------------------------\n'.format(epoch))
    network.fit(x=x, y=y,
                batch_size=128,
                epochs=1,
                callbacks=callback_list)

    # 从原文中随机选取一个开始字符位置取一段文本(文本种子），进行预测
    start_index = random.randint(0, len(text) - max_len - 1)
    seed_text = text[start_index: start_index + max_len]
    print('\n--- Generating with seed:"{}"\n'.format(seed_text))

    # 以不同的温度生成文本
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('\n--- temperature:{}\n'.format(temperature))
        sys.stdout.write(seed_text)

        # 预测400个字符
        for k in range(400):

            # 将种子文本向量化
            sampled = np.zeros(shape=(1, max_len, len(chars)))
            for char_index, char in enumerate(seed_text):
                sampled[0, char_index, chars_indices[char]] = 1.0

            # 预测下一个字符索引
            prediction = network.predict(x=sampled, verbose=0)[0]

            # 将下一个字符接入种子文本，并更新种子文本用于下一次预测
            next_index = sampling(prediction, temperature)
            next_char = chars[int(next_index)]

            seed_text += next_char
            seed_text = seed_text[1:]

            # 输出本次预测的字符
            sys.stdout.write(next_char)
        print('\n---------------------------------------------------------\n')















































