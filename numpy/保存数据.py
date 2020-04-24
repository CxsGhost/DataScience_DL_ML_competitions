import numpy as np

arr3 = np.random.normal(loc=2, scale=3, size=(3, 3))
print(arr3)
'----------------'
np.save('normal', arr3)  # 第一个参数是文件名及路径
'------------------------------------------'
arr4 = np.load('normal.npy')  # 加载目标文件
print(arr4)
'-------------------------------------'