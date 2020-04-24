import numpy as np
import common_functions


class AffineLayer:
    def __init__(self):

        # 保存正向传播时留下的train-x，权重，后续用来计算反向传播
        self.x = None
        self.weight = None
        # 初始偏置单元设为1有助于快速训练
        self.b = 0
        self.biases_weight = None
        self.d_x = None
        self.d_weight = None
        self.d_biases_weight = None

        # 用来保存mini-batch的大小，反向传播中用来计算平均梯度（反向传播将会进行重排，故要提前保存）
        self.batch_size = None

    def forward_propagation(self, x, weight, biases_weight):

        # 存留数据，mini-batch大小，进行正向传播
        self.x = x
        # 一向以引用来传递参数的Python，在init的时候竟然开始复制了，只能每次训练重新赋值权重
        self.weight = weight
        self.biases_weight = biases_weight
        self.batch_size = self.x.shape[1]

        # 偏置权重只有一列，但由于numpy的广播性，偏置会被加到各个列
        return np.dot(self.weight, self.x) + self.b * self.biases_weight

    def backward_propagation(self, d_output):

        """
        他妈的我说怎么每次mini-batch结果都和狗屎一样烂，原来reshape根本不好用，胡乱重排，打乱原有顺序
        还是乖乖的一个个乘把，也别整什么四维矩阵张量乘法了
        """
        # 因为是mini-batch，为了遵循反向传播的规则，需要重排成四维矩阵，进行两层矩阵乘法
        # self.x = np.transpose(self.x, axes=(1, 0))
        # self.x = self.x.reshape((1, self.x.shape[0], 1, -1))
        # # 因为向上一层传播还要用到下一层传来的梯度，所以重排不覆盖原矩阵
        # d_output_copy = np.reshape(d_output, newshape=(d_output.shape[1], 1, -1, 1), order='F')
        #
        # # 计算权重的梯度，因为每个数据要对应其梯度，故得到的四维矩阵对角线上的梯度矩阵才是我们需要的，trace加和后平均
        # self.d_weight = np.matmul(d_output_copy, self.x).trace() / self.batch_size

        self.d_weight = np.dot(d_output[:, 0].reshape((-1, 1)), self.x[:, 0].reshape((1, -1)))
        for e in range(1, self.batch_size):
            self.d_weight += np.dot(d_output[:, e].reshape((-1, 1)), self.x[:, e].reshape((1, -1)))
        self.d_weight /= self.batch_size

        # 计算本层偏置权重，因为bias输入恒等于1，相当于把权重矩阵直接加上去，梯度就等于1乘下一层梯度,sum出来的值会变成一维，转回去
        self.d_biases_weight = (np.sum(d_output, axis=1) / self.batch_size).reshape(-1, 1)

        # 计算传向上一层的梯度
        self.weight = np.transpose(self.weight, axes=(1, 0))
        self.d_x = np.dot(self.weight, d_output)
        print(self.d_x.shape)

        return self.d_x


class SigmoidLayer:
    def __init__(self):
        self.output = None

    def forward_propagation(self, x):

        # 保存输出，后续用来反向传播梯度
        self.output = common_functions.Sigmoid(x)
        return self.output

    def backward_propagation(self, d_output):

        # 根据已经求得的偏导表达式直接计算
        return self.output * (1.0 - self.output) * d_output


class SoftmaxWithLoss:
    def __init__(self):
        self.output = None
        self.target = None
        self.loss = None

    def forward_propagation(self, x, target):

        # 保存输出及目标数据
        self.output = common_functions.SoftMax(x)
        self.target = target
        self.loss = common_functions.Cross_Entropy(self.output, self.target)

        # 返回平均交叉熵误差
        return np.sum(self.loss) / self.output.shape[1]

    def backward_propagation(self):

        # 根据计算出的交叉熵和softmax的反向传播公式
        return self.output - self.target


class ReLULayer:
    def __init__(self):
        # ReLU需要保存x，用于后续反向传播梯度
        self.x = None
        self.output = None

    def forward_propagation(self, x):
        self.x = x
        self.output = common_functions.ReLU(x)

        return self.output

    def backward_propagation(self, d_output):

        # 输入x大于0的偏导为1，否则为0
        self.x = (self.x > 0).astype(np.int)
        return self.x * d_output


class TanhLayer:
    def __init__(self):
        pass

    def forward_propagation(self):
        pass

    def backward_propagation(self):
        pass



