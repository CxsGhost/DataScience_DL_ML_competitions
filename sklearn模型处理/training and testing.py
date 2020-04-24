from sklearn.model_selection import train_test_split
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

cut = train_test_split(data, y,
                       test_size=0.375)  # 默认比例是0.25，可以自行设置
# 在切分数据之前，会自动对数据进行打乱，防止系统性排序的影响

print(type(cut))  # 将会返回一个列表，四个元素
print(len(cut))

print(cut[0])  # train_data
print(cut[1])  # test_data
print(cut[2])  # train_y
print(cut[3])  # test_y

