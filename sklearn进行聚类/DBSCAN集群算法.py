"""
上一章中的均值漂移聚类算法通常表现良好，可以选择合理数量的聚类。
但是，由于计算时间的原因，它的可伸缩性不是很高，并且仍然假设群集具有类似“斑点”的形状（尽管此假设不如K-means那样强）。

另一个也会自动选择群集数量的群集算法是DBSCAN。
DBSCAN通过在数据集中查找密集区域来对数据进行聚类。
数据集中包含大量紧密观察到的数据的区域被视为高密度区域，
而数据稀疏的区域被视为低密度区域。

DBSCAN算法将高密度区域视为数据集中的聚类，而将低密度区域视为聚类之间的区域
（因此，将低密度区域中的观察结果视为噪声，而不放在聚类中）。

高密度区域由核心样本定义，核心样本只是对许多邻居的数据观测。
每个聚类由几个核心样本以及与核心样本相邻的所有观测值组成。

与均值漂移算法不同，DBSCAN算法既具有高度可伸缩性，
又不对数据集中聚类的基本形状进行任何假设。
"""

# “邻居”和“核心样本”的确切定义取决于我们在集群中的需求。
# 我们指定两个被视为邻居的数据观测值之间的最大距离ε。
# 较小的距离会导致群集更小且更紧密。
# 我们还指定了数据观测附近的最小点数，以便将该观测视为核心样本
# （该邻域由数据观测及其所有邻居组成）。

# 使用关键字参数eps（代表ε的值）
# min_samples（代表核心样本的邻域的最小大小）初始化对象。

import numpy as np
from sklearn.cluster import DBSCAN

data = np.random.normal(loc=0, scale=10, size=(300, 4))
data_ = np.random.randint(low=0, high=100, size=(400, 4))
data_e = np.concatenate([data, data_], axis=0)

dbscan = DBSCAN(eps=15, min_samples=5)

pre = np.random.rand(3, 4)

dbscan.fit(data_e)
print(dbscan.labels_)
print(dbscan.core_sample_indices_)
print(len(dbscan.core_sample_indices_))



