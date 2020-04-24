import numpy as np
import scipy.special

# 整个项目最精髓的库，可以把图片矩阵进行左右偏转，使得拥有更多的数据集，获得更好的模型
import scipy.ndimage.interpolation

import matplotlib.pyplot as plt
import data_preprocessing


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
        # scipy.special.logit()， 用于反向激活
        self.inverse_activation_function = lambda y: scipy.special.logit(y)

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

    @staticmethod
    def accuracy():
        # 查看预测效果
        pre_right = 0
        for each_test in range(len(test_data_targets)):
            pre_target = DNN.query(test_data_inputs[each_test])
            if pre_target.argmax() == test_data_targets[each_test]:
                pre_right += 1

        accuracy = pre_right / len(test_data_targets) * 100
        print("学习率：{}， 隐藏层节点数：{}".format(lr, nodes))
        print("正确率为：{0:.2f}%\n".format(accuracy))

    def back_query(self, back_inputs_list):
        back_inputs = np.array(back_inputs_list, ndmin=2)

        # 计算输出层的反向输出，以及隐藏层的反向输入，并缩放范围至sigmoid函数的范围内！！！
        back_output = self.inverse_activation_function(back_inputs)
        back_hidden_inputs = np.dot(back_output, self.who)
        back_hidden_inputs -= np.min(back_hidden_inputs)
        back_hidden_inputs /= np.max(back_hidden_inputs) - np.min(back_hidden_inputs)
        # 防止从反向激活函数输出为-inf
        back_hidden_inputs = back_hidden_inputs * 0.98 + 0.01

        # 计算隐藏层的反向输出，及输入层的反向输入，最终反向输出
        back_hidden_outputs = self.inverse_activation_function(back_hidden_inputs)
        back_final_outputs = np.dot(back_hidden_outputs, self.wih)
        back_final_outputs -= np.min(back_final_outputs)
        back_final_outputs /= np.max(back_final_outputs)
        """
        首先进行归一化是没什么问题的
        因为数据可能存在极小的值导致出现inf,所以要加0.01保底
        但是只单纯加上0.01又有超出1的可能
        所以先乘0.98，再加0.01无论如何不会超出1，还能保底
        """
        back_final_outputs = back_final_outputs * 0.98 + 0.01

        return back_final_outputs

    @staticmethod
    def back_outputs_visualize(target, b_f_o):
        # 在训练时对输入数据做了放缩处理，这里要进行相反的处理恢复原始数据形态
        visual_array = (b_f_o.reshape(28, 28) - 0.01) * 255
        plt.figure()
        plt.imshow(visual_array, cmap="gray", interpolation=None)
        plt.title("the number is :{}".format(target))
        plt.show()

    @staticmethod
    def back_query_show():
        # 预先准备好十个数字的期望输出
        back_input_data = {0: [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                           1: [0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                           2: [0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                           3: [0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                           4: [0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01],
                           5: [0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01],
                           6: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01],
                           7: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01],
                           8: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01],
                           9: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99]}

        for b_i in back_input_data.items():
            b_o = DNN.back_query(b_i[1])
            DNN.back_outputs_visualize(b_i[0], b_o)


# 得到经过预处理的成熟数据
train_data_inputs, train_data_targets = data_preprocessing.get_mature_data()

for lr in [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]:
    for nodes in range(100, 800, 100):
        # 确定每层的节点数量，学习率
        input_nodes = 784
        hidden_nodes = nodes
        output_nodes = 10

        # 经测试发现，学习率最优值约在0.1到0.2之间
        learn_rate = lr

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
                # 把图像旋转后再次进行训练
                """
                rotate函数中
                cval：类型：scalar（标量）---如果模式为“常量”，则填充输入的过去边缘的值， 默认值为0.0
                reshape：类型：boolean（布尔型）---如果reshape为true，则调整输出形状，以便输入数组完全包含在输出中。
                默认为True，如果是False则可以防止挤压数组导致的“断层”
                其他参数参见收藏夹文章
                这里旋转+-10度是最好的选择，转的过大会导致拟合效果下降，整个数据混乱不堪
                """
                each_input_plus10 = scipy.ndimage.interpolation.rotate(each_input.reshape(28, 28),
                                                                       10, cval=0.01, reshape=False).reshape(1, -1)
                DNN.train(each_input_plus10, each_target)
                each_input_minus10 = scipy.ndimage.interpolation.rotate(each_input.reshape(28, 28),
                                                                        -10, cval=0.01, reshape=False).reshape(1, -1)
                DNN.train(each_input_minus10, each_target)

            number += 1
            if number == 10:  # 训练10次,这时候正确率差不多是峰值
                break
        print("训练完成，测试效果")

        # 得到成熟的测试数据
        test_data_inputs, test_data_targets = data_preprocessing.read_segmentation_test_x_labels()

        # 查看预测效果
        DNN.accuracy()

        # 进行反向查询
        DNN.back_query_show()


"""
神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.01，隐藏层节点100，正确率为：97.30%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.01，隐藏层节点200，正确率为：97.70%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.01，隐藏层节点300，正确率为：97.79%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.01，隐藏层节点400，正确率为：97.68%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.01，隐藏层节点500，正确率为：97.74%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.01，隐藏层节点600，正确率为：97.64%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.01，隐藏层节点700，正确率为：97.55%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.03，隐藏层节点100，正确率为：97.30%

神经网络搭建完成，开始训练....
训练完成，测试效果
学习率：0.03，隐藏层节点200，正确率为：97.49%
"""
