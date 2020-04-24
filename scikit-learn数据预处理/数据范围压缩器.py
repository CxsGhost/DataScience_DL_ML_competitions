from sklearn.preprocessing import MinMaxScaler, RobustScaler

import numpy as np


s = MinMaxScaler(feature_range=(0, 1))  # 指定缩放范围，默认（0， 1）

arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [3, 9, 1]])
arr2 = np.array([[0, -1, -3],
                [4, 5, -6],
                [3, 9, 1]])

data = s.fit_transform(arr)
print(data)

# 可以拆开使用

d1 = s.fit(arr2)
print(d1)
d2 = s.transform(arr)
# 此时，转换arr不再按照其本身的数据性质，而是按照arr2的数据性质
print(d2)
'-------------------------------------------------------------- '

# 为了避免离群值和异常值的影响
# 我们利用四分位来稳定的缩放数据
# 25%， 50%， 75%
r = RobustScaler()
data1 = r.fit_transform(arr)
print(data1)
'------------------------------------------------'
# 以上方法均是针对列，也就是每一种数据的归一化
# 接下来介绍，如何以行为单位对数据进行归一化
# 在多特征数据中，一行可以看作是一个向量

# 在聚类问题中，可能会进行l2归一化
from sklearn.preprocessing import Normalizer

n = Normalizer()
data2 = n.fit_transform(data)
print(data2)
# 实质是每一个向量除以模长（L2范数）
# 再说白点，就是求一个向量各个轴的夹角余弦
# 这样向量就缩放到0,1之内了
'---------------------------------------------------------'
