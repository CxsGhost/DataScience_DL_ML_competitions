import numpy as np


def SoftMax(x):
    x = np.exp(x)
    sum_ex = np.sum(x, axis=0)
    return x / sum_ex


def Sigmoid(x):
    x = np.exp(-1 * x)
    return 1 / (1 + x)


def Tanh(x):
    numerator = np.exp(x) - np.exp(-1 * x)
    denominator = np.exp(x) + np.exp(-1 * x)
    return numerator / denominator


def ReLU(x):
    x1 = (x > 0).astype(np.int)
    return x * x1


def Cross_Entropy(y, t):
    y = np.log(y)
    loss = y * t
    return -1 * loss




