import numpy as np
import geatpy as ea
import pandas as pd
import matplotlib.pyplot as plt

def target():
    pass


def strict():
    pass

# 差分进化算法要求问题的模版
class MyProblem(ea.Problem):
    def __init__(self):
        name = 'MyProblem'
        M = 1  # 初始化M（目标维数）
        max_or_min = [1]  # 初始化max_or_min（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 5  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        low_b = [165, 185, 225, 245, 65]  # 决策变量下界
        up_b = [185, 205, 245, 265, 100]  # 决策变量上界
        low_bin = [1 for _ in range(Dim)]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        up_bin = [1 for _ in range(Dim)]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        ea.Problem.__init__(self, name, M, max_or_min, Dim, varTypes, low_b, up_b, low_bin, up_bin)

    def aimFunc(self, pop):  # 要优化的目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        temp_1 = Vars[:, 0]
        temp_2 = Vars[:, 1]
        temp_3 = Vars[:, 2]
        temp_4 = Vars[:, 3]
        velocity = Vars[:, -1]

        # 计算目标函数值
        pop.ObjV = target()

        # 采用可行性法则处理约束条件
        strict_1 = None
        strict_2 = None
        strict_3 = None
        strict_4 = None
        strict_5 = None
        strict_6 = None
        strict_7 = None
        pop.CV = np.hstack([strict_1, strict_2, strict_3,
                            strict_4, strict_5, strict_6,
                            strict_7])


problem = MyProblem()

# 种群设置
Encoding = 'RI'  # 编码方式，实数整数混合编码
N_IND = 150  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, N_IND)  # 实例化种群对象

# 设置算法参数
myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)  # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 500  # 最大进化代数
myAlgorithm.mutOper.F = 0.5  # 差分进化中的缩放因子F
myAlgorithm.recOper.XOVR = 0.7  # 重组概率

# 调用算法进行种群进化
[population, obj_trace, var_trace] = myAlgorithm.run()

# 输出结果
best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])  # 记录最优种群个体是在哪一代
best_ObjV = obj_trace[best_gen, 1]
print('最优(最小)的目标函数值为：{}'.format(best_ObjV))
print('最优的决策变量(温度、速度）值为：')
for i in range(var_trace.shape[1]):
    print('{0: %.3f}'.format(var_trace[best_gen, i]))
print('有效进化代数：{}'.format(obj_trace.shape[0]))
print('最优的一代是第：{} 代'.format(best_gen + 1))
print('评价次数：{}'.format(myAlgorithm.evalsNum))
