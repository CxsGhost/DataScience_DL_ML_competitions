import numpy as np
from sklearn.neighbors import NearestNeighbors

data = np.array([
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3. , 1.4, 0.2],
  [4.7, 3.2, 1.3, 0.2],
  [4.6, 3.1, 1.5, 0.2],
  [5. , 3.6, 1.4, 0.2],
  [5.4, 3.9, 1.7, 0.4],
  [4.6, 3.4, 1.4, 0.3],
  [5. , 3.4, 1.5, 0.2],
  [4.4, 2.9, 1.4, 0.2],
  [4.9, 3.1, 1.5, 0.1]])

nbrs = NearestNeighbors(n_neighbors=2)  # 指定返回最近的两个数据，默认五个
nbrs.fit(data)
targets = np.array([
  [5. , 3.5, 1.6, 0.3],
  [4.8, 3.2, 1.5, 0.1]])  # 观测数据

distance, data_ = nbrs.kneighbors(targets, return_distance=True)  # 指定返回他们的距离
print(distance)
print(data_)
"""
[[0.17320508 0.24494897]
 [0.14142136 0.24494897]]
[[7 0]
 [9 2]]
 """
# 第一行对应着第一个观测数据...


