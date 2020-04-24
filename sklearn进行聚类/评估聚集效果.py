"""
当我们无权访问任何真实的群集分配（标签）时，
我们可以做的最好的评估群集就是只看一下它们，
看看它们对于数据集和域是否有意义。
但是，如果我们确实有权访问数据观察的真实聚类标签，则可以应用许多指标来评估聚类算法。

一种流行的评估指标是调整后的兰德指数。
常规Rand索引可衡量真实聚类分配（真实标签）和预测聚类分配（预测标签）之间的相似性。
调整后的兰德指数（ARI）是常规指数的经修正概率版本，
这意味着对分数进行了调整，以便随机聚类分配不会获得良好的分数。

ARI值的范围是-1至1（含）。负分数表示不良标签，随机标签的分数接近0，完美标签的分数为1。
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score

# 请注意，该adjusted_rand_score函数是对称的。
# 这意味着更改参数的顺序将不会影响得分。
# 此外，标签中的排列或更改标签名称（即0和和1与1和3）不会影响得分。

data = np.random.randint(low=0, high=10, size=(1, 10))
pre = np.random.randint(low=5, high=15, size=(1, 10))
ars = adjusted_rand_score(data[0], pre[0])  # 输入数据必须是1维
print(ars)


"""
另一个常见的聚类评估度量是调整后的互信息（AMI）
ARI和AMI指标非常相似。
他们对完美标签的得分均为1.0，对随机标签的得分接近0.0，对不良标签的得分为负。
"""
from sklearn.metrics import adjusted_mutual_info_score

# 何时使用哪种方法的一般经验法则：
# 当真实集群较大且大小近似相等时，使用ARI；
# 而当真实集群大小不平衡且存在小集群时，则使用AMI。

ami = adjusted_mutual_info_score(data[0], pre[0])
print(ami)





