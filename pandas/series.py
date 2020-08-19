import pandas as pd
import numpy as np


ser1 = pd.Series([1, 2, 3])
print(ser1)
ser2 = pd.Series([[1, 2], [3, 4]], index=['w', 'l'])
print(ser2)
dic1 = {'w': np.nan, 'l': 11, 'x': 13}
ser3 = pd.Series(dic1)
print(ser3.index)

# series对象是一维的，所以实际上是表示一个列表
# 我们可以为其中的元素指定索引，类似于字典
# 也可以直接把字典转化为series
'-----------------------------------------------------'

