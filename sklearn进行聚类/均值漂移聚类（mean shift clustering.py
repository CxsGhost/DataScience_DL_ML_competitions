"""
如果我们对数据集的实际簇数应该有多少不太了解，则可以使用一些算法为我们自动选择簇数。

一种这样的算法是均值漂移聚类算法
像K-means聚类算法一样，均值漂移算法也基于发现聚类质心。
由于我们没有提供簇的数量，因此该算法将在数据中寻找可能成为簇候选对象的“斑点”。

使用这些“斑点”，算法可找到许多候选质心。
然后，它删除基本上与其他对象重复的候选对象，以形成最后的质心集。
最后一组质心确定聚类的数量以及数据集聚类分配（将数据观测值分配给最近的质心）。
"""

import numpy as np
from sklearn.cluster import MeanShift

data = np.random.normal(loc=0, scale=10, size=(30, 4))
data_ = np.random.randint(low=0, high=100, size=(40, 4))
data_e = np.concatenate([data, data_], axis=0)

meanshitf = MeanShift()

meanshitf.fit(data_e)
print(meanshitf.cluster_centers_)
print(meanshitf.labels_)

pre = np.random.rand(3, 4)
print(meanshitf.predict(pre))
"""
[[ 6.78675119e-03  1.28261904e+00  1.54973206e+00 -1.56454357e+00]
 [ 7.36500000e+01  6.58500000e+01  3.01500000e+01  4.12500000e+01]
 [ 4.78461538e+01  1.90769231e+01  5.89230769e+01  6.10769231e+01]]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 1 2 2 1 1 2
 1 2 2 2 2 1 1 2 0 1 1 1 2 1 1 1 2 2 1 1 1 1 1 1 1 1 2 2 1 2 1 2 2]
[0 0 0]
"""
