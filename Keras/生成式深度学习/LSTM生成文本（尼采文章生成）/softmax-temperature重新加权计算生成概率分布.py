"""
详见书229
"""
import numpy as np


# temperature是一个因子，用于定量描述输出分布的熵,更高的温度对应熵更大，生成的数据更具创造性，更少的数据结构
def reweight_distributing(original_distributing, temperature=0.5):
    distribution = np.log(original_distributing) / temperature
    distribution = np.exp(distribution)

    # 返回原始分布加权后的结果，因为加权后可能和不等于1，所以还要除以和，得到新的分布
    return distribution / np.sum(distribution)