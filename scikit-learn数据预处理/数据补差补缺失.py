# 在现实生活中，我们经常不得不处理包含缺失值的数据。有时，如果数据集缺少太多值，我们就是不使用它。
# 但是，如果仅缺少几个值，则可以执行数据插补，以用其他一些值替换丢失的数据。
#
# 有很多不同的数据插补方法。在scikit-learn中，SimpleImputer转换器执行四种不同的数据插补方法。
#
# 四种方法是：
# 使用平均值
# 使用中值
# 使用最频繁的值
# 用常量填充缺失值
from sklearn.impute import SimpleImputer
import numpy as np

arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [np.nan, 9, 1]])

i = SimpleImputer()
data1 = i.fit_transform(arr)  # 默认使用平均值
print(data1)
'_-----------------------------'

i2 = SimpleImputer(strategy='median')  # 使用中位数(本列
data2 = i2.fit_transform(arr)
print(data2)
'---------------------------'

i3 = SimpleImputer(strategy='most_frequent')  # 使用出现次数最多的值（本列
data3 = i3.fit_transform(arr)
print(data3)
'-----------------------------'

i4 = SimpleImputer(strategy='constant', fill_value=22222)  # 使用指定值（本列
data4 = i4.fit_transform(arr)
print(data4)
'-----------------------------------'
# 以上是最基本的方法
# 还有现金的方法比如，knn根据其他特征的值，来计算缺失值的最优解
