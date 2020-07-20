"""
由于XGBoost中有许多参数，并且每个参数有多个可能的值，因此通常需要调整参数。
换句话说，我们想尝试不同的参数设置，看看哪一个可以给我们最好的结果。
"""

import numpy as np
import xgboost as xgb

data = np.random.randint(low=1, high=10, size=(100, 3)) / 9
labels_1 = np.ones((1, 50), dtype='int')
labels_2 = np.zeros((1, 50), dtype='int')
labels = np.concatenate([labels_1, labels_2], axis=1).flatten()
np.random.shuffle(labels)
dtrain = xgb.DMatrix(data, label=labels)

params = {  # 也就是对当前设置的参数进行交叉验证评估
    'max_depth': 3,
    'lambda': 1.5,
    'objective': 'binary:logistic'
}
cv_results = xgb.cv(params, dtrain,
                    nfold=4,  # 设置交叉验证折数，默认3
                    num_boost_round=12)  # 设置最大迭代次数，默认10
"""
cv的输出是一个panda DataFrame(有关详细信息，请参阅数据处理部分)。
它包含应用于给定数量的增强迭代的k次交叉验证的训练和测试结果(平均值和标准偏差)。
K-fold交叉验证的K值是用nfold关键字参数设置的(默认为3)。
"""
print(cv_results)








