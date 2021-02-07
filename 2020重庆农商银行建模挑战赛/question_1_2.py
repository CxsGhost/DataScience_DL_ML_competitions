import pandas as pd
import numpy as np
# from scipy.sparse.linalg import eigs
# import matplotlib.pyplot as plt
import time
import sys

print('读取数据...')
data_3 = None
try:
    data_3 = pd.read_csv('数据包/03_个人客户流水.csv', header=0, dtype={'交易时间': str})
except:
    print('读取失败！请将依赖数据：03_个人客户流水.csv 与程序放在同一文件夹下！')

# 统计账户号
account = data_3['账户号'].unique()
data_3 = data_3.values

print('共有流水数据：{}条'.format(data_3.shape[0]))
print('共有账户号：{}条'.format(len(account)))

# 存放检测到大额交易，大额资金流动的账户
big_day_ck = np.zeros(shape=(len(account), ))
big_day_zr = np.zeros(shape=(len(account), ))
big_day_qk = np.zeros(shape=(len(account), ))
big_day_zc = np.zeros(shape=(len(account), ))
flow = np.array([])

# 存放各账户的信息（如年总收入，投资能力等...）
year_in = np.array([])
clear_property_div = np.array([])
invest_ability = np.array([])
development_ability = np.array([])

action = {'CK': '存款', 'QK': '取款', 'ZR': '转入', 'ZC': '转出', 'QTXF': '其他消费', 'GMLC': '购买理财'}
account_action = data_3[:, 3]
account_name = data_3[:, 0]


def account_detect(account_, ind_):
    global flow, year_in, clear_property_div, \
        invest_ability, development_ability

    # 计算账户总流动资金
    data_ = data_3[account_name == account_]
    ck_ind = data_[:, 3] == 'CK'
    zr_ind = data_[:, 3] == 'ZR'
    all_in = np.sum(data_[ck_ind, 4]) + np.sum(data_[zr_ind, 4])
    all_out = np.sum(data_[data_[:, 3] == 'QK', 4]) + np.sum(data_[data_[:, 3] == 'ZC', 4])
    if all_in and all_out:
        flow = np.append(flow, np.array(np.min([all_in, all_out])))
    else:
        flow = np.append(flow, 0)

    # 对账户各种类型交易进行分类
    ck_data = data_[data_[:, 3] == 'CK']
    zr_data = data_[data_[:, 3] == 'ZR']
    qk_data = data_[data_[:, 3] == 'QK']
    zc_data = data_[data_[:, 3] == 'ZC']
    ck_dates = ck_data[:, 1]
    qk_dates = qk_data[:, 1]
    zc_dates = zc_data[:, 1]
    zr_dates = zr_data[:, 1]

    gmlc_data = data_[data_[:, 3] == 'GMLC']
    qtxf_data = data_[data_[:, 3] == 'QTXF']

    gmlc_expense = np.sum(gmlc_data[:, 4])
    qtxf_expense = np.sum(qtxf_data[:, 4])

    # 一四季度收入
    four_season = 0
    first_season = 0

    # 计算用于评价用户信用等级的指标
    if all_in:
        year_in_money = all_in
        year_clear_in_money = all_in - all_out - gmlc_expense - qtxf_expense
        property_div = year_clear_in_money / year_in_money
        year_out = all_out + gmlc_expense + qtxf_expense
        if year_out:
            invest_ab = gmlc_expense / year_out
        else:
            invest_ab = 0
        in_ind = (ck_ind.astype(np.int) + zr_ind.astype(np.int)) >= 1
        for data in data_[in_ind]:
            if 101 <= data[1] - 20180000 <= 331:
                first_season += data[4]
            elif 901 <= data[1] - 20180000 <= 1231:
                four_season += data[4]
        development_ab = four_season - first_season
        development_ab = development_ab / np.power(10, len(str(abs(int(development_ab)))) - 1)
    else:
        year_in_money = all_out + gmlc_expense + qtxf_expense
        if not year_in_money:
            year_in_money = 100
        year_clear_in_money = 0
        property_div = 0
        year_out = all_out + gmlc_expense + qtxf_expense
        if year_out:
            invest_ab = gmlc_expense / year_out
        else:
            invest_ab = 0
        for data in data_:
            if 101 <= data[1] - 20180000 <= 331:
                first_season += data[4]
            elif 901 <= data[1] - 20180000 <= 1231:
                four_season += data[4]
        development_ab = four_season - first_season
        development_ab = development_ab / np.power(10, len(str(abs(int(development_ab)))) - 1)

    # 记录各账户各属性
    year_in = np.append(year_in, year_in_money)
    invest_ability = np.append(invest_ability, invest_ab)
    clear_property_div = np.append(clear_property_div, property_div)
    development_ability = np.append(development_ability, development_ab)

    # 下面判断账户是否有洗钱嫌疑行为

    # 判断有无日大额存款
    for date in ck_dates:
        day_ck = np.sum(ck_data[ck_data[:, 1] == date, 4])
        if day_ck >= 200000:
            big_day_ck[ind_] = 1
            break

    # 有无日大额取款
    for date in qk_dates:
        day_qk = np.sum(qk_data[qk_data[:, 1] == date, 4])
        if day_qk >= 200000:
            big_day_qk[ind_] = 1
            break

    # 有无日大额转入
    for date in zr_dates:
        day_zr = np.sum(zr_data[zr_data[:, 1] == date, 4])
        if day_zr >= 200000:
            big_day_zr[ind_] = 1

    # 有无日大额转出
    for date in zc_dates:
        day_zc = np.sum(zc_data[zc_data[:, 1] == date, 4])
        if day_zc >= 200000:
            big_day_zc[ind_] = 1


print('开始计算账户各个信息指标、筛选可疑账户...')
st = time.time()
print('正在处理第000000', end='')
for ac in range(1000):
    print('\b\b\b\b\b\b{:6d}'.format(ac), end='')
    account_detect(account[ac], ac)
print('处理完成')
print(time.time() - st)
sys.exit(0)



# # 把各个指标写入Excel并生成
# print('用于评价客户信用的指标写入Excel并生成：问题1：客户信用评价指标.xlsx')
# writer = pd.ExcelWriter('问题1：客户信用评价指标.xlsx')
# confidence_data = np.concatenate([[account], [year_in],
#                                   [clear_property_div], [invest_ability],
#                                   [development_ability]], axis=0).T
# confidence_data = pd.DataFrame(confidence_data,
#                                columns=['账户号', '年收入', '净资产比率', '投资能力', '发展能力'])
# confidence_data.to_excel(writer, sheet_name='客户信用评价指标')
# writer.save()
# writer.close()
#
# x = range(len(flow))
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(10, 8))
# plt.plot(x, flow, color='b', label='各账户资金总流动量')
# plt.ylabel('资金总流动量')
# plt.title('账户资金总流动量分布')
# plt.legend()
# plt.savefig('账户资金总流动量分布.png')
# plt.show()
#
#
# # 整理初步筛选结果，写入Excel生成
# print('处理初步筛选结果...写入Excel并生成： 问题2：洗钱嫌疑初步筛选结果.xlsx')
# writer = pd.ExcelWriter('问题2：洗钱嫌疑初步筛选结果.xlsx')
# big_flow_standard = 500000
# big_flow = flow >= big_flow_standard
# big_flow_account = pd.DataFrame(account[big_flow], columns=['账户号'])
# big_flow_account.to_excel(writer, sheet_name='资金总流动量超标的账户')
# big_flow = big_flow.astype(np.int)
#
# big_day_ck_account = big_day_ck == 1
# big_day_ck_account = pd.DataFrame(account[big_day_ck_account], columns=['账户号'])
# big_day_ck_account.to_excel(writer, sheet_name='突然某日累计存款超过20万元的账户')
#
# big_day_qk_account = big_day_qk == 1
# big_day_qk_account = pd.DataFrame(account[big_day_qk_account], columns=['账户号'])
# big_day_qk_account.to_excel(writer, sheet_name='突然某日累计取款超过20万元的账户')
#
# big_day_zr_account = big_day_zr == 1
# big_day_zr_account = pd.DataFrame(account[big_day_zr_account], columns=['账户号'])
# big_day_zr_account.to_excel(writer, sheet_name='突然某日累计转入超过20万元的账户')
#
# big_day_zc_account = big_day_zc == 1
# big_day_zc_account = pd.DataFrame(account[big_day_zc_account], columns=['账户号'])
# big_day_zc_account.to_excel(writer, sheet_name='突然某日累计转出超过20万元的账户')
#
# writer.save()
# writer.close()
#
# # 筛选结果矩阵化，用于后续计算总得分
# one_hot_data = np.concatenate([[big_day_ck], [big_day_qk], [big_day_zr],
#                                [big_day_zc], [big_flow]], axis=0)
#
# print('层析分析法计算各指标权重...进一步确定账户洗钱可能性')
#
# # 权重决定矩阵
# mat = np.array([[1, 1, 1 / 5, 1 / 5, 1 / 7],
#                 [1, 1, 1 / 5, 1 / 5, 1 / 7],
#                 [5, 5, 1, 1, 5 / 7],
#                 [5, 5, 1, 1, 5 / 7],
#                 [7, 7, 7 / 5, 7 / 5, 1]])
#
#
# # 一致性检验
# def isConsist(F):
#     n = np.shape(F)[0]
#     a, b = eigs(F, 1)
#     maxlam = a[0].real
#     CI = (maxlam - n) / (n - 1)
#     RI = np.array([0, 0, 0.52, 0.89, 1.12, 1.26, 1.36,
#                    1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59])
#     CR = CI / RI[n-1]
#     if CR < 0.1:
#         return bool(1)
#     else:
#         return bool(0)
#
#
# print('一致性检验：{}'.format(isConsist(mat)))
#
#
# # 根据决定矩阵计算各指标权重
# def cal_weights(input_matrix):
#     input_matrix = np.array(input_matrix)
#     n, n1 = input_matrix.shape
#     assert n == n1, '不是一个方阵'
#     eigenvalues, eigenvectors = np.linalg.eig(input_matrix)
#
#     max_idx = np.argmax(eigenvalues)
#     eigen = eigenvectors[:, max_idx].real
#     eigen = eigen / eigen.sum()
#     return eigen
#
#
# # 计算各指标权重
# action_list = ['存款', '取款', '转入', '转出', '总流动']
# weights = cal_weights(mat)
# weight_dict = dict(zip(action_list, weights))
# for item in weight_dict.items():
#     print('{}的权重为：{}'.format(item[0], item[1]))
#
#
# # 各账户最终得分
# print('计算各账户最终得分...写入Excel并生成：问题2：各账户洗钱嫌疑最终得分.xlsx')
# writer = pd.ExcelWriter('问题2：各账户洗钱嫌疑最终得分.xlsx')
# final_score = np.dot(np.array([weights]), one_hot_data)
# final_data = pd.DataFrame(np.concatenate([[account], final_score], axis=0).T,
#                           columns=['账户号', '最终得分'])
# final_data.to_excel(writer, sheet_name='各账户洗钱嫌疑最终得分')
# writer.save()
# writer.close()
