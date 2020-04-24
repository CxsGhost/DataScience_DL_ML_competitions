# 可以用交叉验证的得分，来有效评估和调整决策树的深度
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
# 以cifar10数据集为例，来进行决策树分类


def load_data(_file):
    with open(_file, 'rb') as fi:
        dic = pickle.load(fi, encoding='bytes')
    return dic[b'data'], np.array([dic[b'labels']])


file_list = ['cifar10/data_batch_{}'.format(i) for i in range(2, 6)]
data_x, data_y = load_data('cifar10/data_batch_1')

for f in file_list:
    data_x = np.concatenate([data_x, load_data(f)[0]], axis=0)
    data_y = np.concatenate([data_y, load_data(f)[1]], axis=1)

# 交叉验证来评估不同深度的树，尝试寻找最佳深度的大体区间


def cv_decision_tree(if_clf, data, labels,
                     max_depth, cv):
    if if_clf:
        d_tree = DecisionTreeClassifier(max_depth=max_depth)
    else:
        d_tree = DecisionTreeRegressor(max_depth=max_depth)
    scores = cross_val_score(d_tree, data, labels, cv=cv)
    return scores


for depth in range(500, 3001, 250):
    score = cv_decision_tree(
        True, data_x, data_y[0], depth, cv=5)
    mean = score.mean()
    std_2 = 2 * score.std()
    print('深度：{}，95%的置信区间：{} +/- {}'.format(depth, mean, std_2))

"""
深度：500，95%的置信区间：0.26054 +/- 0.008151171694916023
深度：750，95%的置信区间：0.2599 +/- 0.01024499877989255
深度：1000，95%的置信区间：0.2612 +/- 0.00785162403582851
深度：1250，95%的置信区间：0.25926 +/- 0.010874263193430631
深度：1500，95%的置信区间：0.26028 +/- 0.011456631267523628
深度：1750，95%的置信区间：0.25952000000000003 +/- 0.01115519609867976
深度：2000，95%的置信区间：0.26056 +/- 0.007088695225498157
"""













