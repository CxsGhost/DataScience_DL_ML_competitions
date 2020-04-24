# 脊正则化使用L2范数惩罚项
# 而另一种称为LASSO的正则化方法将L1范数用于权重惩罚项
# LASSO正则化倾向于使用参数值较少的线性模型
# 这意味着它可能会将某些权重系数归零。

from sklearn.linear_model import Lasso, LassoCV
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from re import findall

csv = pd.read_csv('housing.csv').values
data = []
for i in range(len(csv)):
    data.append(list(map(float, findall(r'\d+\.?\d*', csv[i][0]))))
data = np.array(data)
y = data[:, -1]
data = data[:, :-1]

# lasso回归倾向于得出更少参数的模型
la = Lasso(alpha=1)
la.fit(data, y)
print(la.predict([data[0]]))
print(la.coef_)
print(la.intercept_)
print(la.score(data, y))

# 交叉验证版
alphas_ = [0.1, 0.2, 0.3]
lac = LassoCV(alphas=alphas_)
lac.fit(data, y)
print(lac.predict([data[0]]))
print(lac.coef_)
print(lac.intercept_)
print(lac.score(data, y))






