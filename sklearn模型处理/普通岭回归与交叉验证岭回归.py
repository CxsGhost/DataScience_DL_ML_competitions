# 对于正则化，目标不仅是使残差平方和最小化
# 而且要使用尽可能小的系数。
# 系数越小，就越不容易受到数据中随机噪声的影响。
# 最常用的正则化形式是岭正则化。


from sklearn.linear_model import Ridge, RidgeCV
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

r = Ridge(alpha=0.5)
# 普通岭回归， alpha默认是1，这时权重向量长度影响十分大，可以自行调节
r.fit(data, y)
print(r.predict([data[0]]))
print(r.coef_)
print(r.intercept_)
print(r.score(data, y))
'--------------------------------------------'

# 交叉验证岭回归
alphas_ = [0.1, 0.2, 0.3]  # 提供不同的alpha值，将会选出最好的
rc = RidgeCV(alphas=alphas_)
# 这里还并未介绍交叉验证的原理，后面的章节会有
rc.fit(data, y)
print(rc.predict([data[0]]))
print(rc.coef_)
print(rc.intercept_)
print(rc.score(data, y))


