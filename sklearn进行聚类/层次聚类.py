"""
K-均值聚类算法做出的主要假设是数据集由球形（即圆形）聚类组成。
在此假设下，K均值算法将创建围绕质心呈圆形的数据观测值簇。
但是，现实生活中的数据通常不包含球形聚类，
这意味着由于其假设，K均值聚类可能最终产生不准确的聚类。

K均值聚类的一种替代方法是层次聚类。
分层聚类允许我们聚类任何类型的数据，因为它不对数据或聚类做出任何假设。

有两种方法可以进行层次结构聚类：自下而上（划分）和自上而下（聚集）。
除法方法最初将所有数据视为一个群集，然后将其重复拆分为较小的群集，直到达到所需的群集数量。
聚集方法最初将每个数据观察视为其自己的群集，然后反复合并两个最相似的群集，
直到达到所需的群集数量。
"""
# 在实践中，由于使用了更好的算法，因此更常用凝聚方法。
# 因此，本章将重点介绍使用聚集聚类。
import numpy as np
from sklearn.cluster import AgglomerativeClustering

data = np.random.normal(loc=0, scale=10, size=(30, 4))
data_ = np.random.randint(low=0, high=100, size=(40, 4))
data_e = np.concatenate([data, data_], axis=0)

ag=AgglomerativeClustering(n_clusters=3)
ag.fit(data_e)

print(ag.labels_)

# 由于聚集聚类不使用质心，因此对象中没有cluster_centers_属性AgglomerativeClustering。
# 也没有predict用于对新数据进行聚类预测的功能（因为K均值聚类将其最终质心用于新数据预测）。




