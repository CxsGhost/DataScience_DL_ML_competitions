import pandas as pd
import numpy as np


df = pd.DataFrame({'name': ['john doe', 'al smith', 'jack lee'],
                   'year': [2000, 2004, 2012],
                   'rbi': [80, 100, np.nan]})

co = pd.get_dummies(df)  # 将特征值转换为数值指标，1代表是，0代表否
print(co)

print(co.columns)
# 简单来说就是，因为numpy中只能用数据来说话，所以dataframe中的字符串或其他特征必须转化为数字表示
'-----------------------------------------'
matrix = co.values  # 转换为数字指标后，可以转换为numpy矩阵
print(matrix)

