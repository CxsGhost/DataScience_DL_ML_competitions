"""
  在未对content做任何处理之前，直接CountVectorizer，词汇表中有49000个词左右
  通过散点图发现，favorites和retweets是成正比的，并且数据整体十分集中，沿着 y=x
  然后分别对，favorites和retweets画散点图，两图相似度十分高
  都在最后10000多个数据中明显的升高，底部有明显的空白弧形区域。形成一个方块，这部分数据约占25%-30%
  要找到有利的词，不妨在这些数据中寻找
  但是做tf-idf时，依然是要用全部数据
  我想的是分别按照favorites和retweet找词，然后取并集
  先对数据从小到大进行排序，这样可以取到之前所有的离群值

"""
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


row = pd.read_csv('trumptweets.csv', encoding='utf-8')
row = row[['content',
           'retweets',
           'favorites']]
row = row.values


def visualize():
    x_ = row[:, 1]
    y_ = row[:, 2]
    z_ = np.arange(len(row))
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.scatter(x_, y_, s=5)
    plt.xlabel('retweets')
    plt.ylabel('favorites')
    plt.subplot(2, 2, 2)
    plt.scatter(z_, x_, c='r', s=5, label='retweets')
    plt.ylabel('retweets')
    plt.subplot(2, 2, 3)
    plt.scatter(z_, y_, c='k', s=5, label='favorites')
    plt.ylabel('favorites')
    plt.legend()
    plt.savefig('favorites与retweets关系图 ')
    plt.show()


"""
  在未处理文本的词典中发现了很多字母数字混杂的乱码      
  在Excel搜索后发现很多推文的末尾有http或pic.twitter开头的地址
  这是乱码来源，用正则把网址全部洗掉
  还有一些数字也被单独的分成词了，于是把数字也洗掉
  顺便洗掉标点符号和川普的名字，还有停用词
  然后用porter提取词干
"""


def text_preprocessing():
    punc = punctuation + u'？。，“”‘’；：！（）'  # 在少部分文本中发现了中文引号，不排除出现其他符号的可能
    stop_words = set(stopwords.words('english'))
    name = {'Donald', 'Trump', 'J', 'donald', 'trump'}
    stop_words.update(name)
    porter = PorterStemmer()
    for c in range(len(row)):
        row[c][0] = re.sub('http.*', '', row[c][0], re.S)
        row[c][0] = re.sub(r'pic\.twitter.*', '', row[c][0], re.S)
        row[c][0] = re.sub(r'\d+', ' ', row[c][0], re.S)
        for p in punc:
            row[c][0] = row[c][0].replace(p, ' ')
        word_tokens = word_tokenize(row[c][0])
        w_s = []
        for w in word_tokens:
            w = porter.stem(w)
            if w not in stop_words:
                w_s.append(w)
        row[c][0] = " ".join(w_s)


"""
  清洗之后，词汇表减少至27000左右
  在词汇表中看不到乱码或数字了，效果还是不错的
  除了有一条推全文都是阿拉伯语
  
"""


def tf_idf(k):
    """
    本来是打算所有函数只走一遍，然后把tfidf返回的数组和favorites，retweets合并
    合并以后再分别对两个排序找最好的词
    但是无奈，电脑竟然内存掉链子，合并的时候MemoryError
    试了好多种方法就是不行
    于是只能从tfidf这里就排序，让清洗后的文本直接变得有序
    弊端就是tfidf跑了两次，但这种方式是可以成功运行的
    """
    content = pd.DataFrame(row)
    content = content.sort_values(k, ascending=False).values
    count = CountVectorizer()
    bag = count.fit_transform(content[:, 0])
    dic = count.vocabulary_
    vcab = dict(zip(dic.values(), dic.keys()))  # 反转词汇表键值
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True,
                                   sublinear_tf=False)
    tfidf = transformer.fit_transform(bag).toarray()
    return tfidf, vcab


def find_word(arr, proportion, vcb):
    end = int(proportion * len(arr))
    arr = arr[:end]
    values = arr.argmax(axis=1)  # 因为数据量足够大，不妨就提取tf-idf最大的词
    target = [vcb[value] for value in values]
    return set(target)


visualize()
text_preprocessing()
best_favorites = None
best_retweets = None
for i in range(1, 3):
    tfidf_arr, vocabulary = tf_idf(1)
    if i == 1:
        best_retweets = find_word(tfidf_arr, 0.25, vocabulary)
    else:
        best_favorites = find_word(tfidf_arr, 0.3, vocabulary)
best_favorites.intersection(best_retweets)  # 取交集
print('对favorites和retweets有贡献的是:\n{}\n共有{}个！'.format(best_favorites, len(best_favorites)))

"""
运行结果统计
对favorites和retweets有贡献的共有5666个
分别看则各有近5000个，看来其中是有很多重复的
这也照应了成正比的散点图

"""
