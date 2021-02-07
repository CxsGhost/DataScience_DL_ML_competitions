import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers
from sklearn.preprocessing import MinMaxScaler


train_data = pd.read_csv("./titanic/train.csv", index_col=False, header=0)
train_data = train_data.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin'])

# 填充上船地点，nan用平均ASCII码值
embarked_count = train_data['Embarked'].value_counts()
emb_fill = sum(map(lambda ch: embarked_count[ch] * ord(ch) / sum(embarked_count),
                   embarked_count.index))
embarked_dic = {key: ord(key) for key in embarked_count.index}
train_data['Embarked'] = train_data['Embarked'].apply(lambda ch: embarked_dic.get(ch, emb_fill))

# 确认其他数据无nan
print(train_data['Pclass'].isnull().sum(),
      train_data['Sex'].isnull().sum(),
      train_data['SibSp'].isnull().sum(),
      train_data['Parch'].isnull().sum(),
      train_data['Fare'].isnull().sum())

# 填充年龄
female_ave = train_data[train_data['Sex'] == 'female']['Age'].mean()
male_ave = train_data[train_data['Sex'] == 'male']['Age'].mean()


def fill_sex(sex_, value_):
    if pd.isnull(value_):
        if sex_ == 'male':
            return male_ave
        return female_ave
    else:
        return value_


train_data = train_data.values
for i in range(len(train_data)):
    train_data[i][3] = fill_sex(train_data[i][2], train_data[i][3])
train_data[:, 2] = np.array(list(map(lambda ch: 1 if ch == 'male' else -1, train_data[:, 2])))


train_x = train_data[:, 1:]
train_label = train_data[:, 0]
scaler = MinMaxScaler(feature_range=(0, 2))
train_x = scaler.fit_transform(train_x)
train_label = train_label.astype(np.float)

network = models.Sequential()
network.add(layers.Dense(units=128, activation=activations.relu, input_shape=(7, )))
network.add(layers.Dense(units=128))
network.add(layers.BatchNormalization())
network.add(layers.ReLU())
network.add(layers.Dense(units=128))
network.add(layers.BatchNormalization())
network.add(layers.ReLU())
network.add(layers.Dense(units=128, activation=activations.relu))
network.add(layers.Dropout(rate=0.5))
network.add(layers.Dense(units=1, activation=activations.sigmoid))

network.compile(optimizer=optimizers.Adam(),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

print(network.summary())

history = network.fit(x=train_x, y=train_label,
                      epochs=100,
                      batch_size=256,
                      verbose=1)

# import matplotlib.pyplot as plt
# history = history.history
# print(history.keys())
#
# loss_values = history['loss']
# val_loss_values = history['val_loss']
# accuracy = history['binary_accuracy']
# val_accuracy = history['val_binary_accuracy']
# epochs = range(1, len(loss_values) + 1)
#
# # 绘制loss变化趋势图
# ax_1 = plt.figure(figsize=(9, 9)).gca()
# ax_1.plot(epochs, loss_values, color='b', marker='o', linestyle='-', label='Loss_Value')
# ax_1.plot(epochs, val_loss_values, color='r', marker='>', linestyle='--', label='Val_Loss_Value')
# ax_1.set_title('Train and Validation Loss')
# ax_1.set_xlabel('epoch')
# ax_1.set_ylabel('loss')
# ax_1.legend()
#
# # 绘制正确率趋势图
# ax_2 = plt.figure(figsize=(9, 9)).gca()
# ax_2.plot(epochs, accuracy, color='b', marker='*', linestyle='-.', label='Accuracy')
# ax_2.plot(epochs, val_accuracy, color='r', marker='<', linestyle=':', label='Val_accuracy')
# ax_2.set_title('Accuracy')
# ax_2.set_xlabel('epoch')
# ax_2.set_ylabel('accuracy')
# ax_2.legend()
#
# plt.show()


def preprocessing_data():
    train_data = pd.read_csv("./titanic/test.csv", index_col=False, header=0)
    train_data = train_data.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin'])

    # 填充上船地点，nan用平均ASCII码值
    embarked_count = train_data['Embarked'].value_counts()
    emb_fill = sum(map(lambda ch: embarked_count[ch] * ord(ch) / sum(embarked_count),
                       embarked_count.index))
    embarked_dic = {key: ord(key) for key in embarked_count.index}
    train_data['Embarked'] = train_data['Embarked'].apply(lambda ch: embarked_dic.get(ch, emb_fill))

    # 确认其他数据无nan
    print(train_data['Pclass'].isnull().sum(),
          train_data['Sex'].isnull().sum(),
          train_data['SibSp'].isnull().sum(),
          train_data['Parch'].isnull().sum(),
          train_data['Fare'].isnull().sum())

    # 填充年龄
    female_ave = train_data[train_data['Sex'] == 'female']['Age'].mean()
    male_ave = train_data[train_data['Sex'] == 'male']['Age'].mean()

    def fill_sex(sex_, value_):
        if pd.isnull(value_):
            if sex_ == 'male':
                return male_ave
            return female_ave
        else:
            return value_

    train_data = train_data.values
    for i in range(len(train_data)):
        train_data[i][2] = fill_sex(train_data[i][1], train_data[i][2])
    train_data[:, 1] = np.array(list(map(lambda ch: 1 if ch == 'male' else -1, train_data[:, 1])))
    return train_data


test_x = preprocessing_data()
print(test_x)
test_x = scaler.transform(test_x)
pre_y = network.predict(x=test_x)

pre_y = (pre_y >= 0.5).astype(np.int)
pre_y = np.array(pre_y).flatten()
submit_data = pd.DataFrame({'PassengerId': range(892, 892 + len(pre_y)), 'Survived': pre_y})
submit_data.to_csv('submit.csv', index=False)
