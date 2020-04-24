# 交叉验证一般是应用于数据量相对较小的模型
# 这样能最大限度的利用数据
# 而对于大型数据集，我们可以直接分成训练集和数据集
# 大于大型数据集使用交叉验证，将会十分消耗时间

# 下面进行k-fold交叉验证
import numpy as np
import pandas as pd
from re import findall
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
"""
特别说明，交叉验证k折数据的选取是非常讲究的
为了避免数据选择不均衡造成的评估误差
无论是 训练数据 验证数据 测试数据
任何一折数据都会与与原数据的各类比例十分相似
比如原数据有60% A类，40% B类
则每一折也将差不多是这个样子

"""
csv = pd.read_csv('housing.csv').values
data = []
for i in range(len(csv)):
    data.append(list(map(float, findall(r'\d+\.?\d*', csv[i][0]))))
data = np.array(data)
y = data[:, -1]
data = data[:, :-1]
'------------------------------------'
lr = LinearRegression()  # 建立线性回归模型
cvs = cross_val_score(lr, data, y, cv=4)  # 指定CV来指定折数
# 对于回归模型，将使用R2系数来评估模型，或均方误差，或平均绝对误差
print(cvs)  # 返回k轮评估的得分
'---------------------------------'

log = LogisticRegression(solver='lbfgs',
                         multi_class='multinomial',
                         max_iter=50)
cvs2 = cross_val_score(log, data, y, cv=4)
# 对于逻辑回归分类，一般以分类准确率来评估得分
print(cvs2)
# 这里只是为了演示而建立回归分类，实际上波士顿数据集根本不可能用分类解决











