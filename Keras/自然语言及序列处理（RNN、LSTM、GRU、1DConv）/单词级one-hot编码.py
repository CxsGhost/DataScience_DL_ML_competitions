from keras.preprocessing.text import Tokenizer

samples = ['the cat sat on the mat', 'the dog ate my homework']

# 单词索引向量长度限制1000（词表中前1000个最常见的单词）
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

# 将字符串转换为整数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)

# 也可以直接得到one-hot二进制表示。这个分词器还支持除one-hot外其他向量化模式
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print(one_hot_results)
print(one_hot_results.shape)

# 词表索引字典
word_index = tokenizer.word_index

print('Found {} unique tokens.'.format(len(word_index)))

















