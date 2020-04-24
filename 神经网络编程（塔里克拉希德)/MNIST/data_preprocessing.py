import pandas as pd
import numpy as np


def read_segmentation_train_x_labels():
    # 读取数据到dataframe
    train_data = pd.read_csv("MNIST_all/mnist_train.csv", header=None)

    # 把label和input先分离
    train_data_targets = train_data.iloc[:, 0].values
    train_data_inputs = train_data.values

    # 删除第一列的labels，只留下输入
    train_data_inputs = np.delete(train_data_inputs, obj=0, axis=1)

    return train_data_inputs, train_data_targets


def preprocessing(data_inputs, targets):
    # 进行归一化，缩放input的数据范围，来适用于sigmoid
    data_inputs = data_inputs / 255.0 + 0.01

    # 需要把数据按行罗列，根据数据数量生成对应0矩阵，把target转化为标记矩阵（这里并不是标准one-hot）
    data_targets = np.zeros(shape=(len(targets), 10))
    for t in range(len(data_targets)):
        data_targets[t][targets[t]] = 0.99

    return data_inputs, data_targets


def get_mature_data():
    inputs, targets = read_segmentation_train_x_labels()
    inputs, targets = preprocessing(inputs, targets)

    return inputs, targets


def read_segmentation_test_x_labels():
    # 读取并处理测试数据集
    test_data = pd.read_csv("MNIST_all/mnist_test.csv", header=None)
    test_data_targets = test_data.iloc[:, 0].values
    test_data_inputs = np.delete(test_data.values, obj=0, axis=1) / 255.0 + 0.01

    return test_data_inputs, test_data_targets
