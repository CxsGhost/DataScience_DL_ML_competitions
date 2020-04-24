# 优化正则回归模型的超参数的另一种方法是使用贝叶斯技术。
#
# 在贝叶斯统计中，主要思想是在拟合数据之前对模型参数的概率分布做出某些假设。这些初始分布假设被称为模型参数的先验。
#
# 在贝叶斯岭回归模型中，有两个要优化的超参数：α和λ。α超参数与常规岭回归具有相同的精确目的。即，它作为惩罚项的比例因子。
#
# λ超参数充当模型权重的精度。基本上，λ值越小，各个权重值之间的差异就越大。

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from re import findall

csv = pd.read_csv('housing.csv').values
data = []
for i in range(len(csv)):
    data.append(list(map(float, findall(r'\d+\.?\d*', csv[i][0]))))
data = np.array(data)
y = data[:, -1]
data = data[:, :-1]

b = BayesianRidge()
b.fit(data, y)
print(b)
print(b.predict([data[0]]))
print(b.coef_)
print(b.intercept_)
print(b.score(data, y))
print(b.alpha_)  # 这两个日后研究算法的时候，再深究
print(b.lambda_)
