import pandas as pd
import numpy as np
import xgboost as xgb


# 读取数据
train_data = pd.read_excel('data.xlsx', sheet_name='Sheet1',
                           index_col=None, header=None)
pre_data = pd.read_excel('data.xlsx', sheet_name='Sheet2',
                         index_col=None, header=None)
train_data = train_data.values
pre_data = pre_data.values

# 分离训练数据x和y
train_data_y = train_data[:, 0]
train_data_x = train_data[:, 1:]

Data = xgb.DMatrix(data=train_data_x, label=train_data_y)
Data_x = xgb.DMatrix(data=train_data_x)
D_test = xgb.DMatrix(data=pre_data)
params = {'booster': 'gbtree',
          'subsample': 1,
          'objective': 'reg:squarederror',
          'silent': False,
          'eta': 0.1,
          'gamma': 0.1,
          'max_depth': 6,
          'eval_metric': 'merror'}
num_round = 100
result = xgb.train(params=params,
                   dtrain=Data,
                   num_boost_round=num_round)

pre_train_y = result.predict(Data_x)
print('拟合平均误差：{}'.format(np.sum(np.absolute(pre_train_y - train_data_y))
                         / len(train_data_y)))
pre_y = result.predict(D_test)
print(pre_y)






