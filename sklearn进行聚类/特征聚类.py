"""
在“ 数据预处理”部分，我们使用PCA对数据​​集执行特征维数缩减。
我们还可以使用聚类来执行特征维数缩减。
通过将公共要素合并到群集中，我们减少了总要素的数量，同时仍保留了数据集中的大多数原始信息。

比如某两个个特征在所有数据中极其相似，我们可以把两个合为一个

"""
import numpy as np
from sklearn.cluster import FeatureAgglomeration

data1 = np.random.rand(10, 4)
data2 = np.random.randint(low=0, high=2, size=(1, 10))
data3 = np.random.randint(low=10, high=12, size=(1, 10))

data2 = np.transpose(data2, axes=(1, 0))
data3 = np.transpose(data3, axes=(1, 0))

data1 = np.concatenate([data1, data2], axis=1)
data1 = np.concatenate([data1, data3], axis=1)

print(data1)

fa = FeatureAgglomeration(n_clusters=5)  # 指定要降低到的维数

new_data = fa.fit_transform(data1)
print(new_data)




