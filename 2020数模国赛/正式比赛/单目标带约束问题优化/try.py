import numpy as np
import geatpy as ea
"""
    该案例展示了一个带等式约束的连续型决策变量最大化目标的单目标优化问题。
    该函数存在多个欺骗性很强的局部最优点。
    max f = 4*x1 + 2*x2 + x3
    s.t.
    2*x1 + x2 - 1 <= 0
    x1 + 2*x3 - 2 <= 0
    x1 + x2 + x3 - 1 == 0
    0 <= x1,x2 <= 1
    0 < x3 < 2
"""
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 3 # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0,0,0] # 决策变量下界
        ub = [1,1,2] # 决策变量上界
        lbin = [1,1,0] # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,0] # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        pop.ObjV = 4*x1 + 2*x2 + x3 # 计算目标函数值，赋值给pop种群对象的ObjV属性
        # 采用可行性法则处理约束
        pop.CV = np.hstack([2*x1 + x2 - 1,
                        x1 + 2*x3 - 2,
                        np.abs(x1 + x2 + x3 - 1)])


if __name__ == '__main__':
    """================================实例化问题对象==========================="""
    problem = MyProblem() # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'       # 编码方式
    NIND = 100            # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 500  # 最大进化代数
    myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = 0.7  # 重组概率
    """===========================调用算法模板进行种群进化======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板

    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])  # 记录最优种群个体是在哪一代
    best_ObjV = obj_trace[best_gen, 1]
    print('最优(最小)的目标函数值为：{}'.format(best_ObjV))
    print('最优的决策变量(温度、速度）值为：')
    for i in range(var_trace.shape[1]):
        print(var_trace[best_gen, i])
    print('有效进化代数：{}'.format(obj_trace.shape[0]))
    print('最优的一代是第：{} 代'.format(best_gen + 1))
    print('评价次数：{}'.format(myAlgorithm.evalsNum))