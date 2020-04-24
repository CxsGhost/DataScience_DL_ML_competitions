from sklearn import linear_model
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from re import findall


csv = pd.read_csv('housing.csv').values
data = []
for i in range(len(csv)):
    data.append(list(map(float, findall(r'\d+\.?\d*', csv[i][0]))))
data = np.array(data)
y = data[:, -1]
data = data[:, :-1]

pca = PCA(n_components=10)
data = pca.fit_transform(data)
reg = linear_model.LinearRegression()  # 线性模型下最小二乘回归模型
reg.fit(data, y)  # 输入数据和y
pre = reg.predict([data[0]])  # 预测功能, 我们用第一个数据来试一下
print(pre)
print(reg.coef_)  # 查看预测的参数
print(reg.intercept_)  # 查看预测的截距b
print(reg.score(data, y))
# 查看r2系数，是决定系数，而不是相关系数，范围-无穷， 1。 越接近1，拟合得越好

