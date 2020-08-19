import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs


data = pd.read_excel('判断矩阵.xlsx', index_col=0, sheet_name='二级权重判断矩阵')
data_1 = pd.read_excel('判断矩阵.xlsx', header=None, sheet_name='一级权重判断矩阵')
mat_1 = data_1.values
mat = data.values

V = {0: '非常差', 1: '差', 2: '一般', 3: '好', 4: '非常好'}
a1_score = [3.44, 3.10, 2.95]
a2_score = [4.35, 4.78, 4.11, 3.25]
a3_score = [2.56, 3.93]


# 一致性检验
def isConsist(F):
    n = np.shape(F)[0]
    a, b = eigs(F, 1)
    # print('最大特征值：{}'.format(a[0]))
    maxlam = a[0].real
    CI = (maxlam - n) / (n - 1)
    # print('CI值为：{}'.format(CI))
    RI = np.array([0, 0, 0.52, 0.89, 1.12, 1.26, 1.36,
                   1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59])
    CR = CI / RI[n-1]
    if CR < 0.1:
        return bool(1)
    else:
        return bool(0)


# print('二级判断矩阵一致性检验：{}'.format(isConsist(mat)))
# print('一级判断矩阵一致性检验：{}'.format(isConsist(mat_1)))
#

def cal_weights(input_matrix):
    input_matrix = np.array(input_matrix)
    n, n1 = input_matrix.shape
    assert n == n1, '不是一个方阵'
    # for i in range(n):
    #     for j in range(n):
    #         if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
    #             raise ValueError('不是反互对称矩阵')

    eigenvalues, eigenvectors = np.linalg.eig(input_matrix)

    max_idx = np.argmax(eigenvalues)
    eigen = eigenvectors[:, max_idx].real
    eigen = eigen / eigen.sum()
    return eigen


weight = cal_weights(mat)
A = cal_weights(mat_1)
A1 = np.array(weight[0:3])
A2 = np.array(weight[3:7])
A3 = np.array(weight[-2:])
A1 = A1 / np.sum(A1)
A2 = A2 / np.sum(A2)
A3 = A3 / np.sum(A3)

# print('\n第一个一级指标下二级指标的权重：{}'.format(A1))
# print('第二个一级指标下二级指标的权重：{}'.format(A2))
# print('第三个一级指标下二级指标的权重：{}'.format(A3))
# print('三个一级指标权重：{}\n'.format(A))
# print('三个一级指标权重加和：{}\n'.format(np.sum(A)))


def v1(x):
    if x <= 1:
        return 1
    elif x >= 2:
        return 0
    else:
        return (2 - x) / ( 2 - 1)


def v2(x):
    if x <= 1:
        return 0
    elif 1 < x <= 2:
        return (x - 1) / (2 - 1)
    elif 2 < x < 3:
        return (3 - x) / (3 - 2)
    else:
        return 0


def v3(x):
    if x <= 2:
        return 0
    elif 2 < x <= 3:
        return (x - 2) / (3 - 2)
    elif 3 < x < 4:
        return (4 - x) / (4 - 3)
    else:
        return 0


def v4(x):
    if x <= 3:
        return 0
    elif 3 < x <= 4:
        return (x - 3) / (4 - 3)
    elif 4 < x < 5:
        return (5 - x) / (5 - 4)
    else:
        return 0


def v5(x):
    if x <= 4:
        return 0
    elif 4 < x < 5:
        return (x - 4) / (5 - 4)
    else:
        return 1

writer = pd.ExcelWriter('矩阵.xlsx')
R1 = np.array([list(map(v1, a1_score)),
               list(map(v2, a1_score)),
               list(map(v3, a1_score)),
               list(map(v4, a1_score)),
               list(map(v5, a1_score))])
R1 = np.transpose(R1, axes=(1, 0))
r1 = pd.DataFrame(R1).to_excel(writer, sheet_name='R1')
R2 = np.array([list(map(v1, a2_score)),
               list(map(v2, a2_score)),
               list(map(v3, a2_score)),
               list(map(v4, a2_score)),
               list(map(v5, a2_score))])
R2 = np.transpose(R2, axes=(1, 0))
r2 = pd.DataFrame(R2).to_excel(writer, sheet_name='R2')
R3 = np.array([list(map(v1, a3_score)),
               list(map(v2, a3_score)),
               list(map(v3, a3_score)),
               list(map(v4, a3_score)),
               list(map(v5, a3_score))])
R3 = np.transpose(R3, axes=(1, 0))
r3 = pd.DataFrame(R3).to_excel(writer, sheet_name='R3')

B1 = np.dot(A1, R1)
B2 = np.dot(A2, R2)
B3 = np.dot(A3, R3)

pd.DataFrame(B1).to_excel(writer, sheet_name='B1')
pd.DataFrame(B2).to_excel(writer, sheet_name='B2')
pd.DataFrame(B3).to_excel(writer, sheet_name='B3')
R = np.array([B1, B2, B3])
pd.DataFrame(R).to_excel(writer, sheet_name='R')
B = np.dot(A, R)
pd.DataFrame(B).to_excel(writer, sheet_name='B')
writer.save()
writer.close()

print('对五个评语的隶属程度：')
for re in range(5):
    print(V[re], ':', B[re])

result = V[int(np.argmax(B))]
print('评价结果为：{}'.format(result))

pd.DataFrame(np.arange(1, 19, 1))
pd.ExcelWriter('你说叫什么好.xlsx')
project = pd.DataFrame(np.arange(1, 10, 9))
project.to_excel(writer, sheet_name='你说叫神马')
writer.save()
writer.close()
