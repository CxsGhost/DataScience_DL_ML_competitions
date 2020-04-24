import numpy as np
import scipy.special
import pandas as pd


class NeuralNetwork:

    def __init__(self, inputnodes, outputnodes,
                 hiddennodes, learningrate):
        # 设置节点数量属性
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # 按照正态分布设置初始权重矩阵
        self.wih = np.random.normal(loc=0.0,
                                    scale=1 / np.sqrt(self.innodes),
                                    size=(self.hnodes, self.innodes))
        self.who = np.random.normal(loc=0.0,
                                    scale=1 / np.sqrt(self.hnodes),
                                    size=(self.onodes, self.hnodes))

        # 这里scipy.special.expit()，是sigmoid函数，用于正向激活，numpy中并未提供
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list_1, target_list):
        # 转置输入数据，以及目标数据
        input_1 = np.array(input_list_1, ndmin=2)
        input_1 = np.transpose(input_1, axes=(1, 0))
        targets = np.array(target_list, ndmin=2)
        targets = np.transpose(targets, axes=(1, 0))

        # 计算隐藏层输出，以及最终输出
        hidden_inputs_1 = np.dot(self.wih, input_1)
        hidden_outputs_1 = self.activation_function(hidden_inputs_1)
        final_inputs_1 = np.dot(self.who, hidden_outputs_1)
        final_outputs_1 = self.activation_function(final_inputs_1)

        # 计算输出层损失，并反向传播给隐藏层
        output_errors = targets - final_outputs_1
        hidden_errors = np.dot(np.transpose(self.who, axes=(1, 0)), output_errors)

        # 对两组权重进行梯度下降，E(out - target) * sigmoid * ( 1 - sigmoid ) *(矩阵点积) O(hidden)
        self.who += self.lr * np.dot((output_errors * final_outputs_1 * (1.0 - final_outputs_1)),
                                     np.transpose(hidden_outputs_1, axes=(1, 0)))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs_1 * (1.0 - hidden_outputs_1)),
                                     np.transpose(input_1, axes=(1, 0)))

    def query(self, inputs_list_2):
        # 把输入转化成矩阵并转置
        inputs_2 = np.array(inputs_list_2, ndmin=2)
        inputs_2 = np.transpose(inputs_2, axes=(1, 0))

        # 计算输出结果
        hidden_inputs_2 = np.dot(self.wih, inputs_2)
        hidden_outputs_2 = self.activation_function(hidden_inputs_2)
        final_inputs_2 = np.dot(self.who, hidden_outputs_2)
        final_outputs_2 = self.activation_function(final_inputs_2)

        return final_outputs_2


# 读取数据到dataframe
train_data = pd.read_csv("MNIST_all/mnist_train.csv", header=None)

# 把label和input先分离
train_data_targets = train_data.iloc[:, 0].values
train_data_inputs = train_data.values
train_data_inputs = np.delete(train_data_inputs, obj=0, axis=1)

# 缩放input的数据范围，来适用于sigmoid
train_data_inputs = train_data_inputs / 255.0 + 0.01

# 把target转化为标记矩阵，然后转置
data_targets = np.zeros(shape=(len(train_data_targets), 10))
for t in range(len(train_data_targets)):
    data_targets[t][train_data_targets[t]] = 0.99
train_data_targets = data_targets

# 确定每层的节点数量，学习率
input_nodes = 784
hidden_nodes = 10
output_nodes = 10

# 经测试发现，学习率最优值约在0.1到0.2之间
learn_rate = 0.15

# 创建NeuralNetwork实例，并逐条数据训练
DNN = NeuralNetwork(input_nodes, output_nodes,
                    hidden_nodes, learn_rate)
print("神经网络搭建完成，开始训练....")

number = 0
while True:
    # 对原始数据，旋转后的数据依次进行训练
    for each_train_data in zip(train_data_inputs, train_data_targets):
        each_input = each_train_data[0]
        each_target = each_train_data[1]
        DNN.train(each_input, each_target)
        number += 1
    if number == 5:  # 训练5次,这时候正确率差不多是峰值
        break
print("训练完成，测试效果")

# 读取并处理测试数据集
test_data = pd.read_csv("MNIST_all/mnist_test.csv", header=None)
test_data_targets = test_data.iloc[:, 0].values
test_data_inputs = np.delete(test_data.values, obj=0, axis=1) / 255.0 + 0.01

# 查看预测效果
pre_right = 0
for each_test in range(len(test_data_targets)):
    pre_target = DNN.query(test_data_inputs[each_test])
    if pre_target.argmax() == test_data_targets[each_test]:
        pre_right += 1

accuracy = pre_right / len(test_data_targets) * 100
print("正确率为：{0:.2f}%".format(accuracy))









