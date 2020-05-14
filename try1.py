
#
# def f(n):
#     return lambda x, y = [n]: (y.append(x), sum(y))[1]
# print(f(4)(2))
# print(f(4)(1))
#
#
#
# a = [2, 3]
# print(a.index(2))
from keras.preprocessing.text import Tokenizer

a = ['is is is cat cat', 'is is dog']
b = ['is ok ']
t = Tokenizer()
t.fit_on_texts(a)
a = t.texts_to_sequences(a)
b = t.texts_to_sequences(b)
print(b)
print(a)