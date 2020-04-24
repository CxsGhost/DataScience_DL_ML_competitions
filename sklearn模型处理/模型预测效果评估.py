import numpy as np
import pandas as pd
from re import findall
from sklearn.linear_model import RidgeCV
from sklearn import metrics

csv = pd.read_csv('housing.csv').values
data = []
for i in range(len(csv)):
    data.append(list(map(float, findall(r'\d+\.?\d*', csv[i][0]))))
data = np.array(data)
y = data[:, -1]
data = data[:, :-1]


alphas_ = [0.1, 0.3, 0.7]
r = RidgeCV(alphas=alphas_)
r.fit(data, y)
pre = r.predict(data)
'--------------------------------'
# 以下是评估方法(回归类）

print(metrics.r2_score(y, pre))  # r2系数来评估
print(metrics.mean_absolute_error(y, pre))  # 平均绝对损失函数
print(metrics.mean_squared_error(y, pre))  # 均方损失函数

# （分类模型）
print(metrics.accuracy_score(y, pre))  # 评估分类的准确率
