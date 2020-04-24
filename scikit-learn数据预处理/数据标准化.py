import numpy as np
import pandas as pd
from re import findall
from sklearn.preprocessing import scale  # 数据标准化


house = pd.read_csv('housing.csv').values
house_np = []
for d in range(len(house)):
    house_np.append(list(map(float, findall(r'\d+\.?\d*', house[d][0]))))
house = np.array(house_np)

col = scale(house)
print(col)

mean = col.mean(axis=0)  # round是四舍五入，参数值指定三位小数点
std = col.std(axis=0)  # 标准差
var = col.var(axis=0).round(decimals=4)
print(mean)
print(std)
print(var)
