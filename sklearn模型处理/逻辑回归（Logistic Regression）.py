# 逻辑回归模型尽管有其名称，但实际上是用于分类的线性模型。
# 之所以称为逻辑回归，是因为它对logits进行回归
# 然后使我们能够基于模型概率预测对数据进行分类。
#
# 默认设置LogisticRegression是二进制分类
# 即对标记为0或1的数据观测进行分类。

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


da = load_breast_cancer()
data = da.data
y = da.target

log = LogisticRegression(solver='lbfgs',
                         max_iter=10000)   # 逻辑回归默认是二分类问题
log.fit(data, y)
print(log.predict([data[0]]))
print(log.coef_)
print(log.intercept_)
print(log.score(data, y))

# 交叉验证版本
log2 = LogisticRegressionCV(solver='lbfgs',
                            max_iter=10000)  # solver参数可以指定不同的求解器，max_iter指定迭代次数
log2.fit(data, y)
print(log2.predict(data[0:2]))
print(log2.coef_)
print(log2.intercept_)
print(log2.score(data, y))
# 多种模型的对比发现
# 交叉验证的模型决定系数更高，但这并不代表模型一定更好

# 以上均为二分类模型，多分类见数据结构与算法1 cifar数据集
