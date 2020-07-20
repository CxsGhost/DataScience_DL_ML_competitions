"""
XGBoost的基本数据结构是DMatrix，它表示一个数据矩阵。在DMatrix可以从NumPy的阵列构成。

下面的代码创建DMatrix带有和不带有标签的对象。
"""
import xgboost as xgb
import numpy as np

data = np.random.randint(low=1, high=10, size=(10, 3)) / 9
labels_1 = np.ones((1, 5))
labels_2 = np.zeros((1, 5))
labels = np.concatenate([labels_1, labels_2], axis=1)
np.random.shuffle(labels[0])

dmat1 = xgb.DMatrix(data)  # 可以创建没有labels的数据对象
dmat2 = xgb.DMatrix(data, label=labels[0])  # 和其他模型一样，labels必须是一维的

print(dmat2)

"""
该DMatrix对象可用于训练和使用Booster对象，
该对象表示梯度增强决策树。
trainXGBoost中的函数使我们可以训练Booster带有一组指定参数的。
"""
params = {
    'max_depth': 0,  # 设置0表示不限制最大深度
    'objective': 'binary:logistic'  # 设置目标函数为逻辑回归二分类
    # 其他参数使用默认
}
# 以下是多分类的参数设置方法，num_class是种类数
# params = {
#     'max_depth': 2,
#     'objective': 'multi:softmax',
#     'num_class': 3
# }

bst = xgb.train(params=params, dtrain=dmat2)  # booster对象

eval_data = np.random.randint(low=4, high=10, size=(10, 3)) / 9
eval_labels = labels
np.random.shuffle(eval_labels)

deval = xgb.DMatrix(eval_data, label=eval_labels[0])

print(bst.eval(deval))  # evaluation，评估返回的结果是分类误差

pre_data = np.random.randint(low=-2, high=14, size=(5, 3)) / 12
pre_data = xgb.DMatrix(pre_data)
pre_labels = bst.predict(pre_data)  # 因为这里是逻辑回归二分类，所以返回的是概率
print(pre_labels)







