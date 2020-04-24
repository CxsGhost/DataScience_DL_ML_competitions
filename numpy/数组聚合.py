import numpy as np

arr2 = np.random.randint(low=0, high=11, size=(3, 3))
print(arr2)
'---------------------'
print(np.sum(arr2))  # 所有元素求和
print(np.sum(arr2, axis=0))  # 每一列进行求和
'____________________'
print(np.cumsum(arr2))  # 平坦数组进行累积求和
print(np.cumsum(arr2, axis=0))  # 每一列进行累积求和

# 示例：
arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(repr(np.cumsum(arr)))
print(repr(np.cumsum(arr, axis=0)))
print(repr(np.cumsum(arr, axis=1)))
# 输出:
"""array([ 0, 72, 75, 76, 79, 19, 16, 14, 18])
array([[  0,  72,   3],
       [  1,  75, -57],
       [ -2,  73, -53]])
array([[  0,  72,  75],
       [  1,   4, -56],
       [ -3,  -5,  -1]])"""
'---------------------------------------------------------'
print(np.concatenate([arr, arr2], axis=0))  # 纵向拼接数组
print(np.concatenate([arr, arr2], axis=1))  # 横向拼接数组
