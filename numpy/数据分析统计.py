import numpy as np

arr1 = np.random.randint(10, size=(3, 4))
print(arr1)

'-------------------'
print(arr1.max())
print(arr1.min())

print(arr1.min(axis=0))  # 每一列最大值
print(arr1.max(axis=1))  # 每一行最大值

print(arr1.argmax(axis=0))  # 返回的是索引
print(arr1.argmin(axis=-1))  # -1是最低维度的每一维
print(arr1.argmax())  # 不加参数返回展平后的数组的最大数的索引，就一个
'___________________________________________'
print(arr1.mean(axis=0))  # 平均数，不加参数则是平坦数组的平均数
print(arr1.var())  # 方差
print(np.median(arr1))  # 中位数
'_________________________________________________'
