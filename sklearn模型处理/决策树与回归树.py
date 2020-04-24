# 由于大型决策树倾向于过度拟合数据
# 因此手动设置树的最大深度可能会有所帮助

from sklearn import tree
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

tre = tree.DecisionTreeRegressor(max_depth=8)  # 尝试不同的深度，防止过拟合
# 这里是回归树，决策树不妨见cifar10
tre.fit(data, y)
print(tre.predict([data[0]]))
print(tre.score(data, y))
