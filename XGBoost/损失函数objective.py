"""
objective可以指定不同的损失函数，xgb损失函数是两部分组成的，loss和正则化
我们只选择loss部分即可，
xgb.train()默认是binary:logistic
xgb.XGBRegressor()默认是reg：liner
xgb.XGBClassifier()默认是binary：logistic

reg:liner 线性回归损失函数，均方误差
binary:logistic 逻辑回归损失函数，二分类
binary:hinge 支持向量机的损失函数，二分类
multi:softmax 多分类
"""