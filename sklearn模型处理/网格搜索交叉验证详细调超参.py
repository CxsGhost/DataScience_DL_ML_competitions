# 如果我们的应用程序要求我们绝对获得模型的最佳超参数，并且数据集足够小
# 则可以应用详尽的网格搜索来调整超参数。
# 对于网格搜索交叉验证，我们为每个超参数指定可能的值
# 然后搜索将遍历每个超参数的可能组合，并以最佳组合返回模型。


from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
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

reg = BayesianRidge()
params = {
  'alpha_1': [0.1, 0.2, 0.3],
  'alpha_2': [0.1, 0.2, 0.3]
}
reg_cv = GridSearchCV(reg, params, cv=5, iid=False)
reg_cv.fit(data, y)
print(reg_cv.best_params_)














