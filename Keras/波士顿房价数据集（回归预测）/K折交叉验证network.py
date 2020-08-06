from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras import models
from keras import metrics
from keras import losses
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape, test_targets.shape)

# 进行数据标准化
scale = StandardScaler()
scale.fit(train_data)
train_data = scale.transform(train_data)


# def build_model():
#     model_ = models.Sequential()
#     model_.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
#     model_.add(layers.Dense(64, activation='relu'))
#     # 因为是回归问题，故最后一层不设置激活函数,否则会限制输出范围
#     model_.add(layers.Dense(1))
#     model_.compile(loss=losses.mean_squared_error,
#                    optimizer=optimizers.rmsprop(lr=0.001),
#                    metrics=[metrics.mae])
#     return model_
#
#
# # 利用K折交叉验证来评估模型
# k_fold = 4
# num_val_samples = train_data.shape[0] // k_fold
# num_epochs = 200  # 指定训练多少epoch
# all_epochs_finished_scores = []  # 存有每k折每epoch完成后的mae，是k个列表，每个列表有num_epochs个元素
# every_epoch_scores = []  # 存有每k折所有epoch完成后的最终mae，是k个数
#
# for i in range(k_fold):
#     # 准备验证数据，第i个分区的数据
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#
#     # 准备训练数据，截取第k折之外的其他数据并合并
#     partial_train_data = np.concatenate([train_data[: i * num_val_samples],
#                                          train_data[(i + 1) * num_val_samples:]],
#                                         axis=0)
#     partial_train_targets = np.concatenate([train_targets[: i * num_val_samples],
#                                             train_targets[(i + 1) * num_val_samples:]],
#                                            axis=0)
#
#     model = build_model()
#     history = model.fit(x=partial_train_data, y=partial_train_targets,
#                         epochs=num_epochs,
#                         batch_size=1,
#                         validation_data=(val_data, val_targets),
#                         # verbose：日志显示
#                         # verbose = 0 不在标准输出流输出日志信息
#                         # verbose = 1 输出进度条记录
#                         # verbose = 2 每个epoch输出一行记录
#                         verbose=0)
#
#     print('processing k-fold: {}'.format(i))
#     # 获得每epoch后的mae
#     every_epoch_mae_list = history.history['val_mean_absolute_error']
#     every_epoch_scores.append(every_epoch_mae_list)
#
#     # evaluate总是返回loss和metrics,获得训练完成所有epoch后的mae
#     val_mse, val_mae = model.evaluate(x=val_data, y=val_targets,
#                                       # verbose：日志显示
#                                       # verbose = 0 不在标准输出流输出日志信息
#                                       # verbose = 1 输出进度条记录
#                                       verbose=0)
#     all_epochs_finished_scores.append(val_mae)
#
#
# # 查看每epoch后mae的变化，以寻找最佳epochs，要把k个列表数据平均一下再绘图
# # average_every_epoch_scores = np.mean([[x[i] for x in every_epoch_scores] for i in range(num_epochs)])
# every_epoch_scores = np.array(every_epoch_scores)
# average_every_epoch_scores = every_epoch_scores.mean(axis=0)
#
# plt.plot(range(1, num_epochs + 1), average_every_epoch_scores, color='b')
# plt.xlabel('epoch')
# plt.ylabel('average_mae')
# plt.show()
#
#
# # 初步绘图后发现，曲线波动比较大，所以下面用指数加权移动平均值（EWMA）来画图，得到更平滑的曲线
# def EWMA_score(data, beta=0.9):
#     smoothed_data = []
#     new_data = 0
#     for j in data:
#         if new_data:
#             new_data = (1 - beta) * j + beta * new_data
#             smoothed_data.append(new_data)
#         else:
#             new_data = j
#     return smoothed_data
#
#
# average_every_epoch_scores = EWMA_score(average_every_epoch_scores[10:])
# plt.plot(range(1, len(average_every_epoch_scores) + 1), average_every_epoch_scores, color='r')
# plt.xlabel('epoch')
# plt.ylabel('EWMA_mae')
# plt.show()

network = models.Sequential()
network.add(layers.Dense(32, activation='relu', input_shape=(len(train_data[0]), )))

network.add(layers.Dense(32, activation='relu'))

network.add(layers.Dense(32, activation='relu'))

network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(1))

network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.mean_squared_error,
                metrics=[metrics.mean_absolute_error])

network.fit(x=train_data, y=train_targets,
            batch_size=32,
            epochs=30,
            verbose=1)

print(network.evaluate(x=train_data, y=train_targets))












