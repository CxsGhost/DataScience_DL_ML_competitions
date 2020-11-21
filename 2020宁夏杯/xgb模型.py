import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


# 读取数据
data = pd.read_excel('data.xlsx')
data = data.values

# 数据划分
train_data = data[: 650]
pre_data = data[733:]
train_pre_data = data[650: 733]
train_x = train_data[:, 0: 4]
train_y = train_data[:, -1] - 1
train_pre_x = train_pre_data[:, 0: 4]
train_pre_y = train_pre_data[:, -1] - 1
pre_x = pre_data[:, 0: 4]

# 预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.concatenate([train_x, pre_x], axis=0))
train_x = scaler.transform(train_x)
train_pre_x = scaler.transform(train_pre_x)
pre_x = scaler.transform(pre_x)

Data = xgb.DMatrix(data=train_x, label=train_y)
D_test_1 = xgb.DMatrix(data=train_pre_x)
D_test_2 = xgb.DMatrix(data=train_x)
params = {'booster': 'gbtree',
          'subsample': 0.8,
          'objective': 'multi:softmax',
          'num_class': 6,
          'silent': False,
          'eta': 0.1,
          'gamma': 0.2,
          'max_depth': 6,
          'eval_metric': 'merror'}
num_round = 100
result = xgb.train(params=params,
                   dtrain=Data,
                   num_boost_round=num_round)
pre_y_1 = result.predict(D_test_1)
pre_y_2 = result.predict(D_test_2)
e1 = pre_y_1 - train_pre_y
e2 = pre_y_2 - train_y
e1 = np.absolute(e1)
e2 = np.absolute(e2)

r = 0
for i in e1:
    if not i:
        r += 1
print(r / len(e1) * 100)
print(len(e1) - r)
r = 0
for i in e2:
    if not i:
        r += 1
print(r / len(e2) * 100)




