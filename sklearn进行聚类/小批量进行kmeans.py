# 当使用非常大的数据集时，常规K均值聚类可能会非常慢。
# 为了减少计算时间，我们可以执行小批量 K均值聚类
# 这只是将常规K均值聚类一次应用于随机采样的数据子集（小批量）。
# 在使用小批量聚类时需要权衡，因为结果可能不如常规K均值聚类好。
# 但是，实际上质量上的差异可以忽略不计，因此在处理大型数据集时通常选择小批量聚类。


import numpy as np
from sklearn.cluster import MiniBatchKMeans


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

data2 = np.random.rand(100, 4)

test = np.random.randint(low=0, high=5, size=(10, 4)) / 4

data = np.concatenate([data, data2], axis=0)

minikmeans = MiniBatchKMeans(n_clusters=2, batch_size=10)  # 指定小批量的大小
minikmeans.fit(data)

print(minikmeans.labels_)
print(len(minikmeans.labels_))  # 110，分批进行聚类，加快迭代收敛速度
print(minikmeans.cluster_centers_)
print(minikmeans.predict(test))
"""
[1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
[[0.47673878 0.52597462 0.48286977 0.60397956]
 [4.96470588 3.44117647 1.47058824 0.24117647]]
[0 0 0 0 0 0 0 0 0 0]
"""
