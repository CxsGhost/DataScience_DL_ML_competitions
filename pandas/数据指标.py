import pandas as pd
import numpy as np


df = pd.DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6],
                   'c3': [7, 8, 7]}, index=['r1', 'r2', 'r3'])

des1 = df.describe()
print(des1)
# 将会返回：行数
# 平均数
# 中位数
# 标准差
# 最小值，最大值
# 四分之一位置的数
# 四分之三那位置的数
'---------------------------'
des2 = df.describe(percentiles=[0.1, 0.8])
print(des2)
# 指定参数来得到想要的百分位置的数
'---------------------------------'
des3 = df['c3'].value_counts()
print(des3)
# 统计本列中特征值得种类及个数
des4 = df['c3'].value_counts(normalize=True)
print(des4)
# 统计种类，以及按照个数计算每个值的比例，总比例为1
des5 = df['c3'].value_counts(ascending=True)
print(des5)
# 按照值出现的次数，从升序排序，默认是降序
'------------------------------'
des6 = df['c3'].unique()
print(des6)
# unique将会告诉我们出现了哪些值，不会进行计算次数或频率等操作


# 以上操作对字符串类型特征或是数字型都可




