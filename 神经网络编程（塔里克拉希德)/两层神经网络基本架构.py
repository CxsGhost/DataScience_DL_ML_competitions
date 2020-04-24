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
                                    scale=np.sqrt(self.innodes),
                                    size=(self.innodes, self.hnodes))
        self.who = np.random.normal(loc=0.0,
                                    scale=np.sqrt(self.hnodes),
                                    size=(self.hnodes, self.onodes))

        # 这里scipy.special.expit()，是sigmoid函数，numpy中并未提供
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list_1, target_list):
        # 转置输入数据，以及目标数据
        input_1 = np.array([input_list_1], ndmin=2)
        input_1 = np.transpose(input_1, axes=(1, 0))
        targets = np.array([target_list], ndmin=2)
        targets = np.transpose(targets, axes=(1, 0))

        # 计算隐藏层输出，以及最终输出
        hidden_inputs_1 = np.dot(self.wih, input_1)
        hidden_outputs_1 = self.activation_function(hidden_inputs_1)
        final_inputs_1 = np.dot(self.who, hidden_outputs_1)
        final_outputs_1 = self.activation_function(final_inputs_1)

        # 计算输出层损失，并反向传播给隐藏层
        output_errors = targets - final_outputs_1
        hidden_errors = np.dot(np.transpose(self.who, axes=(1, 0)), output_errors)

        # 对两层权重进行梯度下降
        self.who += self.lr * np.dot((output_errors * final_outputs_1 * (1.0 - final_outputs_1)),
                                     np.transpose(hidden_outputs_1, axes=(1, 0)))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs_1 * (1.0 - hidden_outputs_1)),
                                     np.transpose(input_1, axes=(1, 0)))

    def query(self, inputs_list_2):
        # 把输入转化成矩阵并转置
        inputs_2 = np.array([inputs_list_2], ndmin=2)
        inputs_2 = np.transpose(inputs_2, axes=(1, 0))

        # 计算输出结果
        hidden_inputs_2 = np.dot(self.wih, inputs_2)
        hidden_outputs_2 = self.activation_function(hidden_inputs_2)
        final_inputs_2 = np.dot(self.who, hidden_outputs_2)
        final_outputs_2 = self.activation_function(final_inputs_2)

        return final_outputs_2
