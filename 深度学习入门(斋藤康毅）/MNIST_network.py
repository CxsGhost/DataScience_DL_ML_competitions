"""
该网络使用的是softmax和cross entropy loss。不同与之前使用sigmoid和MSE的网络，对小部分数据的训练预测效果却并不好
以下是某次训练中损失函数的值（学习率0.01），可以看出确实是在下降的，所以首先保证了在实现上没有问题，可能是该网络对小型数据不敏感造成的
还没有进行全部数据的训练，以后再试
（学习率0.01）
2.2656619435333827
2.265626447098826
2.2655383052708387
2.265427692839141
2.26530160020836
2.265199750054375
2.2651183886395416
2.2650182321783126
"""
import numpy as np
import pandas as pd
import common_layers
import copy
from collections import OrderedDict

# 读取数据到dataframe
train_data = pd.read_csv("E:/py/科学计算与机器学习/神经网络编程（塔里克拉希德)/MNIST/CUT_MNIST/mnist_train_0.csv", header=None)
for i in range(1, 12):
    part_data = pd.read_csv("E:/py/科学计算与机器学习/神经网络编程（塔里克拉希德)/MNIST/CUT_MNIST/mnist_train_{}.csv".format(i), header=None)
    train_data = pd.concat([train_data, part_data], axis=0)

# dataframe合并后，索引不会自动合并，需要重新标记
train_data = train_data.reset_index(drop=True)
print("数据合并完毕！")

# 把label和input先分离
train_data_targets = train_data.iloc[:, 0].values
train_data_inputs = train_data.values
train_data_inputs = np.delete(train_data_inputs, obj=0, axis=1)
train_data_inputs = np.transpose(train_data_inputs, axes=(1, 0))

# 缩放input的数据范围，来适用于sigmoid
train_data_inputs = train_data_inputs / 255.0 + 0.01

# 把target转化为标记矩阵，然后转置
data_targets = np.zeros(shape=(len(train_data_targets), 10))
for t in range(len(train_data_targets)):
    data_targets[t][train_data_targets[t]] = 1
train_data_targets = data_targets
train_data_targets = np.transpose(train_data_targets, axes=(1, 0))

# 读取并处理测试数据集
test_data = pd.read_csv("E:/py/科学计算与机器学习/神经网络编程（塔里克拉希德)/MNIST/MNIST_all/mnist_test.csv", header=None)
test_data_targets = test_data.iloc[:, 0].values
test_data_inputs = np.delete(test_data.values, obj=0, axis=1) / 255.0 + 0.01
test_data_inputs = np.transpose(test_data_inputs, axes=(1, 0))


for lr in [0.03]:
    w1 = np.random.normal(loc=0, scale=1 / np.sqrt(784), size=(50, 784))
    w2 = np.random.normal(loc=0, scale=1 / np.sqrt(50), size=(10, 50))
    b1 = np.zeros(shape=(50, 1))
    b2 = np.zeros(shape=(10, 1))

    all_layers = OrderedDict()
    all_layers['input_Affine'] = common_layers.AffineLayer()
    all_layers['hidden1_sigmoid'] = common_layers.SigmoidLayer()
    all_layers['Affine_softmax'] = common_layers.AffineLayer()

    output_layer = common_layers.SoftmaxWithLoss()

    learn_rate = lr
    for number in range(0, 30000, 32):
        x = copy.deepcopy(train_data_inputs[:, number:number + 32])
        y = copy.deepcopy(train_data_targets[:, number:number + 32])
        x = all_layers['input_Affine'].forward_propagation(x, weight=w1, biases_weight=b1)
        x = all_layers['hidden1_sigmoid'].forward_propagation(x)
        x = all_layers['Affine_softmax'].forward_propagation(x, weight=w2, biases_weight=b2)

        loss = output_layer.forward_propagation(x, y)
        dout = output_layer.backward_propagation()
        print(loss)

        back_layers = list(all_layers.values())
        back_layers.reverse()
        for back in back_layers:
            dout = back.backward_propagation(dout)

        w1 -= all_layers['input_Affine'].d_weight * learn_rate
        b1 -= all_layers['input_Affine'].d_biases_weight * learn_rate
        w2 -= all_layers['Affine_softmax'].d_weight * learn_rate
        b2 -= all_layers['Affine_softmax'].d_biases_weight * learn_rate

    test_x = copy.deepcopy(test_data_inputs)
    test_y = copy.deepcopy(test_data_targets)
    test_x = all_layers['input_Affine'].forward_propagation(test_x, weight=w1, biases_weight=b1)
    test_x = all_layers['hidden1_sigmoid'].forward_propagation(test_x)
    test_x = all_layers['Affine_softmax'].forward_propagation(test_x, weight=w2, biases_weight=b2)

    output_layer.forward_propagation(test_x, test_y)

    acc_1 = np.argmax(output_layer.output, axis=0)
    acc_2 = test_y
    accuracy = 0
    for acc in zip(acc_1, acc_2):
        if acc[1] == acc[0]:
            accuracy += 1
    print(f'正确率{accuracy / 100}%')

