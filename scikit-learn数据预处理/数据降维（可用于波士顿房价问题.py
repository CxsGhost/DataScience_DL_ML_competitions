# 有一些特征的影响是很小的
# 进行 PCA(主成分分析)

from sklearn.decomposition import PCA
import numpy as np

arr = np.array([[1.5,  3.,  9., -0.5,  1.],
                [2.2,  4.3,  3.5,  0.6,  2.7],
                [3.,  6.1,  1.1,  1.2,  4.2],
                [8., 16.,  7.7, -1.,  7.1]])

p = PCA(n_components=4)  # 假设n个特征，则默认提取n-1个主成分，可自行设定
data = p.fit_transform(arr).round(decimals=3)
print(data)

# 以上如果不限制round，则能提取出四个
# 限制了，则数据产生了变化，会发现最后一列全是0,
# 意味着实际上通过计算，只提取出了三个主成分

