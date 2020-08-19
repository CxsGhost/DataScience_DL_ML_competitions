import numpy as np
import pandas as pd

# 读取之前计算的斜面总辐射量
cant_data = pd.read_excel('问题1斜面太阳辐射总量及相关参数.xlsx', header=0)
cant_data = cant_data.values
cant_data = cant_data[:, -1]

# 读取东南西北各个辐射量
data = pd.read_excel('weather_data.xls', sheet_name='逐时气象参数',
                     index_col=0, header=0)
data = data.values
north_data = data[:, -1]
west_data = data[:, -2]
south_data = data[:, -3]
east_data = data[:, -4]


class Battery:
    def __init__(self, name, power, l, w, eta, price):
        self.name = name
        self.power = power
        self.length = l
        self.width = w
        self.s = l * w / 1e6
        self.eta = eta
        self.price = price
        self.cost = None
        self.profit = None
        self.dian = None


# 所有电池参数
batteries = {'A1': Battery('A1', 215, 1580, 808, 0.1684, 14.9),
             'A2': Battery('A2', 325, 1956, 991, 0.1664, 14.9),
             'A3': Battery('A3', 200, 1580, 808, 0.1870, 14.9),
             'A4': Battery('A4', 270, 1651, 992, 0.1650, 14.9),
             'A5': Battery('A5', 245, 1650, 991, 0.1498, 14.9),
             'A6': Battery('A6', 295, 1956, 991, 0.1511, 14.9),
             'B1': Battery('B1', 265, 1650, 991, 0.1621, 12.5),
             'B2': Battery('B2', 320, 1956, 991, 0.1639, 12.5),
             'B3': Battery('B3', 210, 1482, 992, 0.1598, 12.5),
             'B4': Battery('B4', 240, 1640, 992, 0.1480, 12.5),
             'B5': Battery('B5', 280, 1956, 992, 0.1598, 12.5),
             'B6': Battery('B6', 295, 1956, 992, 0.1520, 12.5),
             'B7': Battery('B7', 250, 1668, 1000, 0.1499, 12.5),
             'C1': Battery('C1', 100, 1300, 1100, 0.0699, 4.8),
             'C2': Battery('C2', 58, 1321, 711, 0.0617, 4.8),
             'C3': Battery('C3', 100, 1414, 1114, 0.0635, 4.8),
             'C4': Battery('C4', 90, 1400, 1100, 0.0584, 4.8),
             'C5': Battery('C5', 100, 1400, 1100, 0.0649, 4.8),
             'C6': Battery('C6', 4, 355, 310, 0.0363, 4.8),
             'C7': Battery('C7', 4, 615, 180, 0.0363, 4.8),
             'C8': Battery('C8', 8, 615, 355, 0.0366, 4.8),
             'C9': Battery('C9', 12, 920, 355, 0.0366, 4.8),
             'C10': Battery('C10', 12, 818, 355, 0.0413, 4.8),
             'C11': Battery('C11', 50, 1645, 712, 0.0427, 4.8)}

for b in batteries.items():
    b[1].cost = (b[1].power * b[1].price) / (b[1].length * b[1].width * 1e-6)


# 计算单位净利润，并排序
def battery_sort(data_, dict_):
    p = 0.5
    rank_names = []
    rank_profits = []
    rank_gets = []
    rank_cost = []
    for ba in dict_.items():
        if ba[1].price == 4.8:
            mat1 = data_ >= 30
            mat1 = mat1.astype(np.int)
            energy = np.sum(mat1 * data_)
        else:
            mat1 = data_ >= 80
            mat1 = mat1.astype(np.int)
            energy = np.sum(mat1 * data_)
        power = energy * ba[1].eta
        E = power * (10 + 0.9 * 15 + 0.8 + 10)
        ba[1].dian = E
        zs = E / 1e3 * p
        profit = zs - ba[1].cost
        rank_names.append(ba[1].name)
        rank_profits.append(profit)
        rank_gets.append(zs)
        rank_cost.append(ba[1].cost)
    rank_data = pd.DataFrame(data=rank_names, columns=['电池类型'])
    rank_cost = pd.Series(rank_cost, name='单位面积成本')
    rank_gets = pd.Series(rank_gets, name='单位面积收益')
    rank_profits = pd.Series(rank_profits, name='单位面积利润')
    rank_data = pd.concat([rank_data, rank_cost, rank_gets, rank_profits], axis=1)
    rank_data = rank_data.sort_values('单位面积利润', ascending=False)
    return rank_data


# north_sort = battery_sort(north_data, batteries)
south_sort = battery_sort(south_data, batteries)
nan = 6 * batteries['B3'].dian * 0.94  + (2 * batteries['C2'].dian + 10 * batteries['C10'].dian) * 0.86
# east_sort = battery_sort(east_data, batteries)
west_sort = battery_sort(west_data, batteries)
we = (10 * batteries['C1'].dian + 6 * batteries['C2'].dian) * 0.94 + (batteries['C8'].dian + 3 * batteries['C10'].dian) * 0.84
cant_sort = battery_sort(cant_data, batteries)
cant = 40 * batteries['A3'].dian * 0.9 + (3 * batteries['A3'].dian + 8 * batteries['C10'].dian) * 0.86
print((nan+ we + cant) / 1000)


# 利润排序数据生成Excel
writer = pd.ExcelWriter('电池各墙面利润排序.xlsx')
north_sort.to_excel(writer, sheet_name='北面墙电池利润排序')
south_sort.to_excel(writer, sheet_name='南面墙电池利润排序')
east_sort.to_excel(writer, sheet_name='东面墙电池利润排序')
west_sort.to_excel(writer, sheet_name='西面墙电池利润排序')
cant_sort.to_excel(writer, sheet_name='屋顶电池利润排序')

writer.save()
writer.close()

# 构造电池利润顺序列表
south_sort = south_sort.values[:, [0, -1]]
west_sort = west_sort.values[:, [0, -1]]
east_sort = east_sort.values[:, [0, -1]]
cant_sort = cant_sort.values[:, [0, -1]]

battery_list = []


def laying(length, width, sort_list):
    h = 0
    s = 0
    h_h = 0
    h_s = 0
    s_s = 0
    s_h = 0
    if length <= 0 or width <= 0:
        return 0
    else:
        for bat in sort_list:
            if batteries[bat].profit <= 0:
                continue
            if length - batteries[bat].length >= 0 and width - batteries[bat].width >= 0:
                battery_list.append(batteries[bat].name)
                h = 1
                h_h = batteries[bat].profit * batteries[bat].s + \
                      laying(length - batteries[bat].length, batteries[bat].width, sort_list) + \
                      laying(length, width - batteries[bat].width, sort_list)
                h_s = batteries[bat].profit * batteries[bat].s + \
                      laying(batteries[bat].length, width - batteries[bat].width, sort_list) + \
                      laying(length - batteries[bat].length, width, sort_list)
            if length - batteries[bat].width >= 0 and width - batteries[bat].length >= 0:
                battery_list.append(batteries[bat].name)
                s = 1
                s_s = batteries[bat].profit * batteries[bat].s + \
                      laying(batteries[bat].width, width - batteries[bat].length, sort_list) + \
                      laying(length - batteries[bat].width, width, sort_list)
                s_h = batteries[bat].profit * batteries[bat].s + \
                      laying(length - batteries[bat].width, batteries[bat].length, sort_list) + \
                      laying(length, width - batteries[bat].length, sort_list)
            if h and s:
                h_max = max(h_h, h_s)
                s_max = max(s_s, s_h)
                if h_max >= s_max:
                    return h_max
                else:
                    return s_max
            if h:
                h_max = max(h_h, h_s)
                return h_max
            if s:
                s_max = max(s_s, s_h)
                return s_max
        return 0


# 西面墙
west_l = 3000
west_w = 2800
for p in west_sort:
    batteries[p[0]].profit = p[1]
west_sort = west_sort[:, 0]
print('\n西面墙最大利润：{}，及此时可用电池板'.format(laying(west_l, west_w, west_sort)))
for i in set(battery_list):
    print(i)
print('\n')
battery_list.clear()

# # 东面墙
# east_l = 7100
# east_w = 2500 + 700
# for p in east_sort:
#     batteries[p[0]].profit = p[1]
# east_sort = east_sort[:, 0]
# print('忽略门时，东面墙最大利润：{}，及此时可用电池板'.format(laying(east_l, east_w, east_sort)))
# for i in set(battery_list):
#     print(i)
# print('\n')
# battery_list.clear()

# 南面墙区域1
south_1_l = 9000
south_1_w = 2800
for p in south_sort:
    batteries[p[0]].profit = p[1]
south_sort = south_sort[:, 0]
print('忽略圆形窗户时，南面墙区域1最大利润：{}，及此时可用电池板'.format(laying(south_1_l, south_1_w, south_sort)))
for i in set(battery_list):
    print(i)
print('\n')
battery_list.clear()

# # 南面墙区域2
# south_2_l = 3600
# south_2_w = 700
# print('忽略门时，南面墙区域2最大利润：{}，及此时可所用电池板'.format(laying(south_2_l, south_2_w, south_sort)))
# for i in set(battery_list):
#     print(i)
# print('\n')
# battery_list.clear()
#
# # 南面墙区域3
# south_3_l = 3000
# south_3_w = 3200
# print('忽略方形窗户时，南面墙区域3最大利润：{}，及此时可所用电池板'.format(laying(south_3_l, south_3_w, south_sort)))
# for i in set(battery_list):
#     print(i)
# print('\n')
# battery_list.clear()


# 由于屋顶空间较大，之前的递归时间过长，故重写递归方式
def laying_roof(length, width, sort_list):
    if length <= 0 or width <= 0:
        return 0
    else:
        for bat in sort_list:
            if batteries[bat].profit <= 0:
                continue
            if length - batteries[bat].length >= 0 and width - batteries[bat].width >= 0:
                battery_list.append(batteries[bat].name)
                h_h = batteries[bat].profit * batteries[bat].s + \
                      laying_roof(length - batteries[bat].length, batteries[bat].width, sort_list) + \
                      laying_roof(length, width - batteries[bat].width, sort_list)
                h_s = batteries[bat].profit * batteries[bat].s + \
                      laying_roof(batteries[bat].length, width - batteries[bat].width, sort_list) + \
                      laying_roof(length - batteries[bat].length, width, sort_list)
                return max(h_h, h_s)
            elif length - batteries[bat].width >= 0 and width - batteries[bat].length >= 0:
                battery_list.append(batteries[bat].name)
                s_s = batteries[bat].profit * batteries[bat].s + \
                      laying_roof(batteries[bat].width, width - batteries[bat].length, sort_list) + \
                      laying_roof(length - batteries[bat].width, width, sort_list)
                s_h = batteries[bat].profit * batteries[bat].s + \
                      laying_roof(length - batteries[bat].width, batteries[bat].length, sort_list) + \
                      laying_roof(length, width - batteries[bat].length, sort_list)
                return max(s_s, s_h)
        return 0


# 南屋顶
roof_l = 9000
roof_w = 3600
for p in cant_sort:
    batteries[p[0]].profit = p[1]
cant_sort = cant_sort[:, 0]
print('忽略天窗时，屋顶最大利润：{}，及此时可用电池板'.format(laying_roof(roof_l, roof_w, cant_sort)))
for i in set(battery_list):
    print(i)
print('\n')
battery_list.clear()


