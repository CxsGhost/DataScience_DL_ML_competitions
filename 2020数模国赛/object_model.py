import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
g = 10


# 定义浮标类，及其初始化函数
class Drogue:
    def __init__(self):
        self.d = 2
        self.height = 2
        self.G = 1000 * g
        self.sink_depth = None
        self.theta = None
        self.buoyancy = None
        self.wind_force = None
        self.T = None
        self.V = None
        self.max_weight = None
        self.water_force = None


# 定义钢管类，及其初始化函数
class SteelTube:
    def __init__(self):
        self.l = 1
        self.d = 5e-2
        self.G = 10 * g
        self.buoyancy = (1.025e3 * np.pi *
                         np.square(self.d) *
                         self.l * g) / 4
        self.T = None
        self.theta = None
        self.water_force = None


# 定义铁桶类，及其初始化函数
class SteelBucket:
    def __init__(self):
        self.l = 1
        self.d = 3e-1
        self.G = 100 * g
        self.theta = None
        self.alpha = None
        self.buoyancy = (1.025e3 * np.pi *
                         np.square(self.d) *
                         self.l * g) / 4
        self.T = None
        self.water_force = None


# 定义锚链单元，及其初始化函数
class ChainUnit:
    def __init__(self, unit_weight, unit_length, unit_diameter = None):
        self.l = unit_length
        self.G = unit_weight * g
        self.T = None
        self.theta = None
        self.water_force = None
        self.d = unit_diameter
        self.unit_species = {0.078: 1, 0.105: 2, 0.12: 3, 0.15: 4, 0.18: 5}


# 定义锚类，及其初始化函数
class Anchor:
    def __init__(self):
        self.G = 600 * g


# 定义该模型类（该类模型通用框架）
class Model:

    # 初始化函数为创建模型中各个对象的实例，以及各个外界参数
    def __init__(self):
        self.drogue = Drogue()
        self.tubes = [SteelTube() for _ in range(4)]
        self.bucket = SteelBucket()
        self.anchor = Anchor()
        self.globe_G = None
        self.wind_velocity = None
        self.water_depth = None
        self.water_velocity = None
        self.water_Rho = 1.025e3
        self.globe_max_weight = None

        # 用于存放锚链单元
        self.chains = []

    # 问题一求解函数，函数接收参数为该模型的各种参数，例如海水深度，锚链型号等
    def solve_1(self, chain_length=None, unit_length=None, per_weight=None,
                water_depth=None, sink_depth=None,
                wind_velocity=None, globe_weight=None):

        # 计算锚链单元长度，单元个数
        unit_length = unit_length / 1000
        unit_weight = per_weight * unit_length
        unit_numbers = int(chain_length / unit_length)

        # 将锚链单元实例添加至模型
        self.chains.clear()
        for _ in range(unit_numbers):
            self.chains.append(ChainUnit(unit_weight=unit_weight, unit_length=unit_length))

        # 将各个参数赋给模型对应的属性
        self.water_depth = water_depth
        self.globe_G = globe_weight * g
        self.wind_velocity = wind_velocity

        # 赋值吃水深度
        self.drogue.sink_depth = sink_depth

        # 计算入水体积，浮力，风力
        self.drogue.V = (np.pi * np.square(self.drogue.d) * self.drogue.sink_depth) / 4
        self.drogue.buoyancy = self.water_Rho * self.drogue.V * g
        self.drogue.wind_force = 0.625 * (self.drogue.height -
                                          self.drogue.sink_depth) * self.drogue.d * np.square(self.wind_velocity)

        # 计算并赋值浮标对第一节钢管的拉力及其角度
        self.drogue.theta = np.arctan((self.drogue.buoyancy - self.drogue.G) / self.drogue.wind_force)
        self.drogue.T = self.drogue.wind_force / np.cos(self.drogue.theta)

        # 计算并赋值每节钢管对下一节钢管的拉力及其角度
        self.tubes[0].theta = np.arctan((self.drogue.T * np.sin(self.drogue.theta) +
                                         self.tubes[0].buoyancy - self.tubes[0].G) /
                                        (self.drogue.T * np.cos(self.drogue.theta)))
        self.tubes[0].T = (self.drogue.T * np.cos(self.drogue.theta)) / np.cos(self.tubes[0].theta)
        for i in range(1, 4):
            self.tubes[i].theta = np.arctan((self.tubes[i - 1].T * np.sin(self.tubes[i - 1].theta) +
                                             self.tubes[i].buoyancy - self.tubes[i].G) /
                                            (self.tubes[i - 1].T * np.cos(self.tubes[i - 1].theta)))
            self.tubes[i].T = (self.tubes[i - 1].T *
                               np.cos(self.tubes[i - 1].theta)) / np.cos(self.tubes[i].theta)

        # 计算并赋值钢桶对第一节锚链单元的拉力及其角度
        self.bucket.theta = np.arctan((self.tubes[3].T * np.sin(self.tubes[3].theta) +
                                       self.bucket.buoyancy - self.bucket.G - self.globe_G) /
                                      (self.tubes[3].T * np.cos(self.tubes[3].theta)))
        self.bucket.T = (self.tubes[3].T * np.cos(self.tubes[3].theta)) / np.cos(self.bucket.theta)

        # 计算并赋值钢桶自身的倾斜角度
        self.bucket.alpha = - np.arctan((self.bucket.T * np.cos(self.bucket.theta)) /
                                        ((self.bucket.buoyancy - self.bucket.G) / 2 -
                                         self.globe_G - self.bucket.T *
                                        np.sin(self.bucket.theta)))

        # 计算并赋值每节锚链单元对下一节锚链单元的拉力及其角度
        self.chains[0].theta = np.arctan((self.bucket.T * np.sin(self.bucket.theta) -
                                          self.chains[0].G) / (self.bucket.T *
                                         np.cos(self.bucket.theta)))
        self.chains[0].T = (self.bucket.T * np.cos(self.bucket.theta)) / np.cos(self.chains[0].theta)
        for j in range(1, unit_numbers):
            self.chains[j].theta = np.arctan((self.chains[j - 1].T *
                                              np.sin(self.chains[j - 1].theta) -
                                              self.chains[j].G) /
                                             (self.chains[j - 1].T *
                                              np.cos(self.chains[j - 1].theta)))
            self.chains[j].T = (self.chains[j - 1].T *
                                np.cos(self.chains[j - 1].theta)) / np.cos(self.chains[j].theta)

        # 计算在所给参数下的海水深度
        water_depth = 0
        water_depth += self.drogue.sink_depth
        water_depth += self.tubes[0].l * np.sin(self.drogue.theta)
        for i in range(1, 4):
            water_depth += self.tubes[i].l * np.sin(self.tubes[i - 1].theta)
        water_depth += self.bucket.l * np.sin(self.bucket.theta)
        water_depth += self.chains[0].l * np.sin(self.bucket.theta)
        for j in range(1, len(self.chains)):

            # 锚链会有拖地的情况，故只计算未拖地的长度
            if self.chains[j - 1].theta > 0:
                water_depth += self.chains[j].l * np.sin(self.chains[j - 1].theta)

        # 设定迭代完成的最大误差条件，符合条件再计算游动区域，最后输出结果
        delta = 0.01
        if abs(self.water_depth - water_depth) <= delta:

            # 计算游动区域
            r = self.tubes[0].l * np.cos(self.drogue.theta)
            for i in range(1, 4):
                r += self.tubes[i].l * np.cos(self.tubes[i - 1].theta)
            r += self.bucket.l * np.sin(self.bucket.alpha)
            r += self.chains[0].l * np.cos(self.bucket.theta)
            for j in range(1, len(self.chains)):

                # 仅计算未拖地的锚链单元
                if self.chains[j - 1].theta > 0:
                    r += self.chains[j].l * np.cos(self.chains[j - 1].theta)

            # 输出结果
            print('\n-------------以下是风速{}m/s时的求解结果-----------'.format(wind_velocity))
            print('-------吃水深度约为：{}'.format(self.drogue.sink_depth))
            print('-------此时计算所的海水深度：{}'.format(water_depth))
            print('-------此时海水深度与真实深度误差小于：{}'.format(delta))
            print('-------浮标游动区域半径为：{}'.format(r))
            print('钢管1的倾斜角度为：{}'.format((np.pi / 2 - self.drogue.theta) / (np.pi * 2) * 360))
            for p in range(1, 4):
                print('钢管{}的倾斜角度为：{}'.format(p + 1, (np.pi / 2 - self.tubes[p - 1].theta) / (np.pi * 2) * 360))
            print('钢桶的倾斜角度为：{}\n'.format(self.bucket.alpha / (np.pi * 2) * 360))

            # 绘制图像，首先将列表倒序
            self.chains.reverse()
            x = []
            y = []
            for unit in self.chains:
                if unit.theta > 0:
                    x.append(unit.l * np.cos(unit.theta))
                    y.append(unit.l * np.sin(unit.theta))

            # 依次进行累加形成数据点
            x = np.cumsum(np.array(x))
            y = np.cumsum(np.array(y))
            x_ = []
            y_ = []

            # 每隔5个点选取一个点(并且手动将最后一个点加入）
            for i in range(0, len(x), 5):
                x_.append(x[i])
                y_.append(y[i])
            x_.append(x[-1])
            y_.append(y[-1])

            # 设置字体，防止绘图中文显示乱码，调整长宽比，防止失真
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.figure(figsize=(int(max(x_)) + 1, int(max(y_)) + 1))
            plt.plot(x_, y_, color='b', marker='o', linestyle='-', label='Drogue')
            plt.xticks(np.arange(0, max(x_) + 1, 1))
            plt.yticks(np.arange(0, max(y) + 1, 1))
            plt.xlabel('锚链水平方向长度（单位：米）')
            plt.ylabel('锚链竖直方向长度（单位：米）')
            plt.title('风速{}时锚链模拟图(每间隔五个点取一个点)'.format(wind_velocity))
            plt.legend()
            plt.savefig('{}跨步5.png'.format(wind_velocity))
            plt.show()
            return 1

    # 问题二第二问的求解函数
    def solve_2_pre(self):
        self.chains.clear()
        for _ in range(210):
            self.chains.append(ChainUnit(unit_weight=0.735, unit_length=0.105))
        self.wind_velocity = 36
        self.water_depth = 18

        # 浮标完全入水时，重物球质量达到上限
        buoyancy_1 = (self.water_Rho * np.pi *
                      np.square(self.drogue.d) *
                      self.drogue.height) / 4
        self.globe_max_weight = buoyancy_1 + self.bucket.buoyancy + self.tubes[0].buoyancy * 4

    def solve_2(self, sink_depth=None,  globe_weight=None):

        self.globe_G = globe_weight * g
        self.drogue.sink_depth = sink_depth

        # 计算入水体积，浮力，风力
        self.drogue.V = (np.pi * np.square(self.drogue.d) * self.drogue.sink_depth) / 4
        self.drogue.buoyancy = self.water_Rho * self.drogue.V * g
        self.drogue.wind_force = 0.625 * (self.drogue.height -
                                          self.drogue.sink_depth) * self.drogue.d * np.square(self.wind_velocity)

        # 计算并赋值浮标对第一节钢管的拉力及其角度
        self.drogue.theta = np.arctan((self.drogue.buoyancy - self.drogue.G) / self.drogue.wind_force)
        self.drogue.T = self.drogue.wind_force / np.cos(self.drogue.theta)

        # 计算并赋值每节钢管对下一节钢管的拉力及其角度
        self.tubes[0].theta = np.arctan((self.drogue.T * np.sin(self.drogue.theta) +
                                         self.tubes[0].buoyancy - self.tubes[0].G) /
                                        (self.drogue.T * np.cos(self.drogue.theta)))
        self.tubes[0].T = (self.drogue.T * np.cos(self.drogue.theta)) / np.cos(self.tubes[0].theta)
        for i in range(1, 4):
            self.tubes[i].theta = np.arctan((self.tubes[i - 1].T * np.sin(self.tubes[i - 1].theta) +
                                             self.tubes[i].buoyancy - self.tubes[i].G) /
                                            (self.tubes[i - 1].T * np.cos(self.tubes[i - 1].theta)))
            self.tubes[i].T = (self.tubes[i - 1].T *
                               np.cos(self.tubes[i - 1].theta)) / np.cos(self.tubes[i].theta)

        # 计算并赋值钢桶对第一节锚链单元的拉力及其角度
        self.bucket.theta = np.arctan((self.tubes[3].T * np.sin(self.tubes[3].theta) +
                                       self.bucket.buoyancy - self.bucket.G - self.globe_G) /
                                      (self.tubes[3].T * np.cos(self.tubes[3].theta)))
        self.bucket.T = (self.tubes[3].T * np.cos(self.tubes[3].theta)) / np.cos(self.bucket.theta)

        # 计算并赋值钢桶自身的倾斜角度
        self.bucket.alpha = - np.arctan((self.bucket.T * np.cos(self.bucket.theta)) /
                                        ((self.bucket.buoyancy - self.bucket.G) / 2 -
                                         self.globe_G - self.bucket.T *
                                         np.sin(self.bucket.theta)))

        # 计算并赋值每节锚链单元对下一节锚链单元的拉力及其角度
        self.chains[0].theta = np.arctan((self.bucket.T * np.sin(self.bucket.theta) -
                                          self.chains[0].G) / (self.bucket.T *
                                                               np.cos(self.bucket.theta)))
        self.chains[0].T = (self.bucket.T * np.cos(self.bucket.theta)) / np.cos(self.chains[0].theta)
        for j in range(1, len(self.chains)):
            self.chains[j].theta = np.arctan((self.chains[j - 1].T *
                                             np.sin(self.chains[j - 1].theta) -
                                              self.chains[j].G) /
                                             (self.chains[j - 1].T *
                                              np.cos(self.chains[j - 1].theta)))
            self.chains[j].T = (self.chains[j - 1].T *
                                np.cos(self.chains[j - 1].theta)) / np.cos(self.chains[j].theta)

        # 计算在所给参数下的海水深度
        water_depth = 0
        water_depth += self.drogue.sink_depth
        water_depth += self.tubes[0].l * np.sin(self.drogue.theta)
        for i in range(1, 4):
            water_depth += self.tubes[i].l * np.sin(self.tubes[i - 1].theta)
        water_depth += self.bucket.l * np.sin(self.bucket.theta)
        water_depth += self.chains[0].l * np.sin(self.bucket.theta)
        for j in range(1, len(self.chains)):

            # 锚链会有拖地的情况，故只计算未拖地的长度
            if self.chains[j - 1].theta > 0:
                water_depth += self.chains[j].l * np.sin(self.chains[j - 1].theta)

        # 首先判断深度是否匹配，如果匹配则进一步判断
        if abs(water_depth - self.water_depth) <= 1:
            alpha = (5 - self.bucket.alpha / (2 * np.pi) * 360 > 0)
            if (self.chains[-1].theta < 0) and (alpha > 0):

                # 输出结果
                print('\n重物球的质量约为：{}kg 时符合条件'.format(globe_weight))
                print('钢桶的倾斜角度为：{}'.format(self.bucket.alpha / (np.pi * 2) * 360))
                print('此时锚链拖地')
                return 1
            elif (16 - self.chains[-1].theta / (2 * np.pi) * 360 > 0) and (alpha > 0):

                # 输出结果
                print('重物球的质量约为：{}kg 时符合条件'.format(globe_weight))
                print('钢桶的倾斜角度为：{}'.format(self.bucket.alpha / (np.pi * 2) * 360))
                print('最后一个锚链单元倾斜角度为：{}\n'.format(self.chains[-2].theta / (np.pi * 2) * 360))
                return 1

    # 问题三求解函数
    def solve_3_out(self, chain_length, unit_length, water_depth, sink_depth, globe_weight):
        # 计算游动区域
        r = self.tubes[0].l * np.cos(self.drogue.theta)
        for i in range(1, 4):
            r += self.tubes[i].l * np.cos(self.tubes[i - 1].theta)
        r += self.bucket.l * np.sin(self.bucket.alpha)
        r += self.chains[0].l * np.cos(self.bucket.theta)
        for j in range(1, len(self.chains)):

            # 仅计算未拖地的锚链单元
            if self.chains[j - 1].theta > 0:
                r += self.chains[j].l * np.cos(self.chains[j - 1].theta)

        # 输出结果
        x_h = self.chains[0].unit_species[unit_length]
        c_d = chain_length
        w_d = water_depth
        s_d = sink_depth
        t_1 = (np.pi / 2 - self.drogue.theta) / (2 * np.pi) * 360
        t_2 = (np.pi / 2 - self.tubes[0].theta) / (2 * np.pi) * 360
        t_3 = (np.pi / 2 - self.tubes[1].theta) / (2 * np.pi) * 360
        t_4 = (np.pi / 2 - self.tubes[2].theta) / (2 * np.pi) * 360
        g_t = self.bucket.alpha / (2 * np.pi) * 360
        z_m = self.chains[-2].theta / (np.pi * 2) * 360
        print('\n锚链型号：{}'.format(x_h))
        print('锚链长度：{}'.format(c_d))
        print('重物球质量：{}'.format(weight))
        print('此时计算所得海水深度:{}'.format(w_d))
        print('浮标吃水深度：{}'.format(s_d))
        print('浮标游动区域：{}'.format(r))
        print('第1节钢管的倾斜角：{}'.format(t_1))
        print('第2节钢管的倾斜角：{}'.format(t_2))
        print('第3节钢管的倾斜角：{}'.format(t_3))
        print('第4节钢管的倾斜角：{}'.format(t_4))
        print('钢桶倾斜角度：{}'.format(g_t))
        print('最后一个锚链单元倾斜角度为：{}\n'.format(z_m))
        return np.array([[x_h, c_d, weight, w_d, s_d, r, t_1, t_2, t_3, t_4, g_t, z_m]])

    def solve_3(self, chain_length=None, unit_length=None, per_weight=None,
                unit_diameter=None, sink_depth=None, globe_weight=None):

        # 计算锚链单元长度，单元个数
        unit_length = unit_length / 1000
        unit_weight = per_weight * unit_length
        unit_numbers = int(chain_length / unit_length)

        # 将锚链单元实例添加至模型
        self.chains.clear()
        for _ in range(unit_numbers):
            self.chains.append(ChainUnit(unit_weight=unit_weight,
                                         unit_length=unit_length,
                                         unit_diameter=unit_diameter))

        # 将各个参数赋给模型对应的属性
        self.water_velocity = 1.5
        self.globe_G = globe_weight * g
        self.wind_velocity = 36

        # 赋值吃水深度
        self.drogue.sink_depth = sink_depth

        # 计算入水体积，浮力，风力，水流力
        self.drogue.V = (np.pi * np.square(self.drogue.d) * self.drogue.sink_depth) / 4
        self.drogue.buoyancy = self.water_Rho * self.drogue.V * g
        self.drogue.wind_force = 0.625 * (self.drogue.height -
                                          self.drogue.sink_depth) * self.drogue.d * np.square(self.wind_velocity)
        self.drogue.water_force = 374 * self.drogue.d * self.drogue.sink_depth * np.square(self.water_velocity)

        # 计算并赋值浮标对第一节钢管的拉力及其角度
        self.drogue.theta = np.arctan((self.drogue.buoyancy - self.drogue.G) /
                                      (self.drogue.wind_force + self.drogue.water_force))
        self.drogue.T = (self.drogue.wind_force +
                         self.drogue.water_force) / np.cos(self.drogue.theta)

        # 钢管浮力，拉力及其角度
        self.tubes[0].water_force = (374 * self.tubes[0].d * self.tubes[0].l *
                                     np.sin(self.drogue.theta)) * np.square(self.water_velocity)
        self.tubes[0].theta = np.arctan((self.drogue.T * np.sin(self.drogue.theta) +
                                         self.tubes[0].buoyancy - self.tubes[0].G) /
                                        (self.drogue.T * np.cos(self.drogue.theta) + self.tubes[0].water_force))
        self.tubes[0].T = (self.drogue.T * np.cos(self.drogue.theta) +
                           self.tubes[0].water_force) / np.cos(self.tubes[0].theta)
        for i in range(1, 4):
            self.tubes[i].water_force = (374 * self.tubes[i].d * self.tubes[i].l *
                                         np.sin(self.tubes[i - 1].theta)) * np.square(self.water_velocity)
            self.tubes[i].theta = np.arctan((self.tubes[i - 1].T * np.sin(self.tubes[i - 1].theta) +
                                             self.tubes[i].buoyancy - self.tubes[i].G) /
                                            (self.tubes[i - 1].T * np.cos(self.tubes[i - 1].theta) +
                                             self.tubes[i].water_force))
            self.tubes[i].T = (self.tubes[i - 1].T *
                               np.cos(self.tubes[i - 1].theta) +
                               self.tubes[i].water_force) / np.cos(self.tubes[i].theta)

        # 计算钢桶浮力，拉力，及其角度
        self.bucket.water_force = (374 * self.bucket.d * self.bucket.l *
                                   np.sin(self.tubes[-1].theta)) * np.square(self.water_velocity)
        self.bucket.theta = np.arctan((self.tubes[3].T * np.sin(self.tubes[3].theta) +
                                       self.bucket.buoyancy - self.bucket.G - self.globe_G) /
                                      (self.tubes[3].T * np.cos(self.tubes[3].theta) + self.bucket.water_force))
        self.bucket.T = (self.tubes[3].T * np.cos(self.tubes[3].theta) +
                         self.bucket.water_force) / np.cos(self.bucket.theta)

        # 计算并赋值钢桶自身的倾斜角度
        self.bucket.alpha = - np.arctan((self.bucket.T * np.cos(self.bucket.theta)) /
                                        ((self.bucket.buoyancy - self.bucket.G) / 2 -
                                         self.globe_G - self.bucket.T *
                                         np.sin(self.bucket.theta) - self.bucket.water_force / 2))

        # 锚链拉力及其角度,浮力
        self.chains[0].water_force = (374 * self.chains[0].d * self.chains[0].l *
                                      np.sin(self.bucket.theta)) * np.square(self.water_velocity)
        self.chains[0].theta = np.arctan((self.bucket.T * np.sin(self.bucket.theta) -
                                          self.chains[0].G) / (self.bucket.T *
                                                               np.cos(self.bucket.theta) + self.chains[0].water_force))
        self.chains[0].T = (self.bucket.T * np.cos(self.bucket.theta) +
                            self.chains[0].water_force) / np.cos(self.chains[0].theta)
        for j in range(1, unit_numbers):
            self.chains[j].water_force = (374 * self.chains[j].d * self.chains[j].l *
                                          np.sin(self.chains[j - 1].theta)) * np.square(self.water_velocity)
            self.chains[j].theta = np.arctan((self.chains[j - 1].T *
                                              np.sin(self.chains[j - 1].theta) -
                                              self.chains[j].G) /
                                             (self.chains[j - 1].T *
                                              np.cos(self.chains[j - 1].theta) + self.chains[j].water_force))
            self.chains[j].T = (self.chains[j - 1].T *
                                np.cos(self.chains[j - 1].theta) + self.chains[j].water_force) / np.cos(self.chains[j].theta)

        # 计算在所给参数下的海水深度
        water_depth = 0
        water_depth += self.drogue.sink_depth
        water_depth += self.tubes[0].l * np.sin(self.drogue.theta)
        for i in range(1, 4):
            water_depth += self.tubes[i].l * np.sin(self.tubes[i - 1].theta)
        water_depth += self.bucket.l * np.sin(self.bucket.theta)
        water_depth += self.chains[0].l * np.sin(self.bucket.theta)
        for j in range(1, len(self.chains)):

            # 锚链会有拖地的情况，故只计算未拖地的长度
            if self.chains[j - 1].theta > 0:
                water_depth += self.chains[j].l * np.sin(self.chains[j - 1].theta)

        # 水深限制在约16到20米
        if 15.9 <= water_depth <= 20.1:

            # 检查两个角度是否符合,符合则输出
            alpha = (5 - self.bucket.alpha / (2 * np.pi) * 360 > 0)
            if (self.chains[-1].theta < 0) and (alpha > 0):
                return 1, self.solve_3_out(chain_length, unit_length, water_depth, sink_depth, globe_weight)

            elif (16 - self.chains[-1].theta / (2 * np.pi) * 360 > 0) and (alpha > 0):
                return 1, self.solve_3_out(chain_length, unit_length, water_depth, sink_depth, globe_weight)

        return 0, 0


# 实例化模型
model = Model()


# 求解风速12m/s时
for depth in np.arange(0.001, 2, 0.0001):
    stop = model.solve_1(chain_length=22.05, unit_length=105, per_weight=7,
                         water_depth=18, wind_velocity=12,
                         globe_weight=1200, sink_depth=depth)
    if stop:
        break


# 求解风速24m/s时
for depth in np.arange(0.001, 2, 0.00001):
    stop = model.solve_1(chain_length=22.05, unit_length=105, per_weight=7,
                         water_depth=18, wind_velocity=24,
                         globe_weight=1200, sink_depth=depth)
    if stop:
        break


# 求解风速36m/s时
for depth in np.arange(0.001, 2, 0.0001):
    stop = model.solve_1(chain_length=22.05, unit_length=105, per_weight=7,
                         water_depth=18, wind_velocity=36,
                         globe_weight=1200, sink_depth=depth)
    if stop:
        break


# 求解重物球质量范围
model.solve_2_pre()

# 设置字体，防止绘图中文显示乱码，调整长宽比，防止失真
plt.rcParams['font.sans-serif'] = ['SimHei']
ax = plt.figure(figsize=(11, 9)).gca()
ax.set_xticks(np.arange(0, 18, 2))
ax.set_yticks(np.arange(0, 23, 2))
ax.set_xlabel('锚链水平方向长度（单位：米）')
ax.set_ylabel('锚链竖直方向长度（单位：米）')
ax.set_title('36m/s风速下，不同重物球的锚链模拟图')

# 以下是两个控制参数，stop决定改重量是否符合，number决定是否绘图
stop = 0
number = 0
for weight in np.arange(1200, model.globe_max_weight, 100):
    for depth in np.arange(0.74, 2, 0.0001):
        stop = model.solve_2(sink_depth=depth, globe_weight=weight)
        if stop:
            number += 1
            break
    if stop and (number % 2):
        stop = 0
        # 绘制图像，首先将列表倒序
        model.chains.reverse()
        x = []
        y = []
        for unit in model.chains:
            if unit.theta > 0:
                x.append(unit.l * np.cos(unit.theta))
                y.append(unit.l * np.sin(unit.theta))
            else:
                x.append(model.chains[0].l)
                y.append(0)

        # 依次进行累加形成数据点
        x = np.cumsum(np.array(x))
        y = np.cumsum(np.array(y))

        # 每隔5个点选取一个点
        x_ = []
        y_ = []
        for i in range(0, len(x), 5):
            x_.append(x[i])
            y_.append(y[i])
        x_.append(x[-1])
        y_.append(y[-1])

        plt.plot(x_, y_, color='b', marker='o', linestyle='-')

plt.savefig('question2.png')
plt.show()


# 求解第三问
# 将各种类型锚链的参数存入表格
unit_species = np.array([[78, 3.2, 0.022788],
                         [105, 7, 0.033704],
                         [120, 12.5, 0.045039],
                         [150, 19.5, 0.056253],
                         [180, 28.12, 0.067552]])

# 计算锚链最短长度
chain_length_min = 20 - model.drogue.height - 4 * model.tubes[0].l - model.bucket.l

# 用于存放可使用的方案
project = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

write = (0, 0)
for chain in unit_species:
    for length in np.arange(chain_length_min, 25, 1):
        for weight in np.arange(1800, model.globe_max_weight, 100):
            for depth in np.arange(0.74, 2, 0.01):
                write = model.solve_3(chain_length=length, unit_length=chain[0], per_weight=chain[1],
                                      unit_diameter=chain[2], sink_depth=depth, globe_weight=weight)
                if write[0]:
                    project = np.concatenate([project, write[1]], axis=0)

# 将迭代所得方案存入excel
project_excel = pd.DataFrame(project)
writer = pd.ExcelWriter('project.xlsx')
project_excel.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()






















