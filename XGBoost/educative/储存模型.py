import numpy as np
import xgboost as xgb

data = np.random.randint(low=1, high=10, size=(100, 3)) / 9
labels_1 = np.ones((1, 50), dtype='int')
labels_2 = np.zeros((1, 50), dtype='int')
labels = np.concatenate([labels_1, labels_2], axis=1).flatten()
np.random.shuffle(labels)
dtrain = xgb.DMatrix(data, label=labels)

param = {
'max_depth': 0,
'objective': 'binary:logistic',
}
booster = xgb.train(params=param, dtrain=dtrain, num_boost_round=10)

pre_data = np.random.randint(low=1, high=10, size=(100, 3)) / 9
pre__ = xgb.DMatrix(pre_data)
pre_labels = booster.predict(pre__)
print(pre_labels)

booster.save_model('model.bin')  # 保存当期训练完的模型

# 以下是如何加载先前保存的模型
bst = xgb.Booster()
bst = bst.load_model('model.bin')

"""
也可以使用pickle或者joblib来高效保存，详见B站教学"""

