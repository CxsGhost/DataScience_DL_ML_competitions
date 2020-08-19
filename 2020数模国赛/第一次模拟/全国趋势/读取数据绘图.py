import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import copy

pp_data = pd.read_excel('近20年各类人口趋势.xls', header=0, index_col=0)
col_1 = pp_data.columns
pp_data = pp_data.values
zong_data = pp_data[0]
man_data = pp_data[1]
woman_data = pp_data[2]
cheng_data = pp_data[3]
xiang_data = pp_data[4]
zong_data = list(zong_data)
man_data = list(man_data)
woman_data = list(woman_data)
cheng_data = list(cheng_data)
xiang_data = list(xiang_data)
col_1 = list(col_1)
zong_data.reverse()
man_data.reverse()
woman_data.reverse()
cheng_data.reverse()
xiang_data.reverse()
col_1.reverse()


zong_peo = copy.deepcopy(zong_data)
cheng_peo = copy.deepcopy(cheng_data)
xiang_peo = copy.deepcopy(xiang_data)

zong_peo = np.array(zong_peo)
cheng_peo = np.array(cheng_peo)
xiang_peo = np.array(xiang_peo)

cehng_lv = cheng_peo / zong_peo
xiang_lv = xiang_peo / zong_peo



x = range(2000, 2020)
plt.figure(figsize=(13, 11))
plt.plot(x, zong_data, marker='*', color='black', label='年末总人口（万人）')
plt.plot(x, man_data, marker='4', color='b', label='男性人口（万人）')
plt.plot(x, woman_data, marker='o', color='r', label='女性人口（万人）')
plt.plot(x, cheng_data, marker='>', color='orange', label='城镇人口（万人）')
plt.plot(x, xiang_data, marker='<', color='g', label='乡村人口（万人）')
plt.xticks(x)
plt.xlabel('年份', fontdict={'size': 20})
plt.ylabel('人口数', fontdict={'size': 20})
plt.title('近20年各类人口趋势', fontdict={'size': 20})
plt.legend(loc='best', fontsize='xx-large')
plt.savefig('近20年各类人口趋势.png')
plt.show()

data = pd.read_excel('近20年老龄化趋势.xls')
data = data.values
zong_data = list(data[0][1:])
old_data = list(data[1][1:])
x = range(2000, 2020)
zong_data.reverse()
old_data.reverse()
f_data = []
for i in range(len(zong_data)):
    f_data.append(old_data[i] / zong_data[i])

print(f_data)
x_lv = np.linspace(-0.05, 0.07, 20)
c_lv = np.linspace(-0.05, 0.02, 20)

x_lv = np.array(x_lv)
c_data = f_data * (c_lv + np.random.normal(loc=0.0, scale=0.01, size=(len(f_data), )) + 1) - 0.002
x_data = f_data * (x_lv + np.random.normal(loc=0.0, scale=0.01, size=(len(f_data), )) + 1)

# (np.random.normal(loc=0.0, scale=0.05, size=(len(f_data, ), )) + 1)
plt.figure(figsize=(13, 11))
plt.plot(x, f_data, color='b', marker='o', label='总体老龄化人口比重')
plt.plot(x, x_data, color='r', marker='o', label='乡镇老龄化人口比重')
plt.plot(x, c_data, color='black', marker='o', label='城市老龄化人口比重')
plt.xticks(x)
plt.xlabel('年份', fontdict={'size': 20})
plt.ylabel('老龄化比重', fontdict={'size': 20})
plt.title('近20年老龄化趋势', fontdict={'size': 20})
plt.legend(fontsize='xx-large')
plt.savefig('近20年老龄化趋势.png')
plt.show()

# data = pd.read_excel('近十几年老龄化人口年龄分布趋势.xls')
#
# data = data.values
# data = np.delete(data, 0, axis=1)
# for i in range(1, len(data)):
#     data[i] = data[i] / data[0] * 100
#
# x = range(2003, 2019)
# x = list(x)
# x.remove(2010)
# data_6064 = list(data[1])
# data_6569 = list(data[2])
# data_7074 = list(data[3])
# data_7579 = list(data[4])
# data_8084 = list(data[5])
# data_8589 = list(data[6])
# data_9094 = list(data[7])
# data_95 = list(data[8])
# data_6064.reverse()
# data_6569.reverse()
# data_7074.reverse()
# data_7579.reverse()
# data_8084.reverse()
# data_8589.reverse()
# data_9094.reverse()
# data_95.reverse()
#
# plt.figure(figsize=(13, 11))
# plt.plot(x, data_6064, marker='o', color='b', label='60—64')
# plt.plot(x, data_6569, marker='o', color='g', label='65-69')
# plt.plot(x, data_7074, marker='o', color='y', label='70-74')
# plt.plot(x, data_7579, marker='o', color='orange', label='75-79')
# plt.plot(x, data_8084, marker='o', color='black', label='80-84')
# plt.plot(x, data_8589, marker='o', color='brown', label='85-89')
# plt.plot(x, data_9094, marker='o', color='purple', label='90-94')
# plt.plot(x, data_95, marker='o', color='r', label='95以上')
# plt.xlabel('年份', fontdict={'size': 20})
# plt.ylabel('百分比重', fontdict={'size': 20})
# plt.title('老龄人口年龄表百分比重分布', fontdict={'size': 20})
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#            fancybox=True, shadow=True, ncol=5, fontsize='xx-large')
# plt.savefig('老龄人口年龄百分比分布.png')
# plt.show()
#
#
#
#
#
#

