import pandas as pd
import numpy as np

print('读取客户评价指标原始数据...')
data = pd.read_excel('客户信用评价指标.xlsx', index_col=0)
data = data.sort_values('年收入', ascending=True)
data = data.values

# 发展能力标准化
data[:, 4] = (data[:, 4] - np.min(data[:, 4])) / (np.max(data[:, 4]) - np.min(data[:, 4]))

print('按年收入进行客户分类...')
year_in = data[:, 1]
qw_ind = []
bw_ind = []
sw_ind = []
w_ind = []
low_w_ind = []
for i in range(len(year_in)):
    if 9999999 < year_in[i] <= 99999999:
        qw_ind.append(i)
    elif 999999 < year_in[i] <= 9999999:
        bw_ind.append(i)
    elif 99999 < year_in[i] <= 999999:
        sw_ind.append(i)
    elif 9999 < year_in[i] <= 99999:
        w_ind.append(i)
    else:
        low_w_ind.append(i)

qw_data = data[qw_ind]
bw_data = data[bw_ind]
sw_data = data[sw_ind]
w_data = data[w_ind]
low_w_data = data[low_w_ind]


# 对资产归一化
def scaler(data_):
    data_[:, 1] = (data_[:, 1] - np.min(data_[:, 1])) / (np.max(data_[:, 1]) - np.min(data_[:, 1]))
    return data_


print('对资产归一化...')
qw_data = scaler(qw_data)
bw_data = scaler(bw_data)
w_data = scaler(w_data)
sw_data = scaler(sw_data)
low_w_data = scaler(low_w_data)

# # 处理后的数据写入excel
# print('写入生成excel：客户按收入分级.xlsx')
# writer = pd.ExcelWriter('客户按收入分级.xlsx')
# qw_data_ = pd.DataFrame(qw_data, columns=['账户号', '年收入', '净资产比率', '投资能力', '发展能力'])
# qw_data_.to_excel(writer, sheet_name='千万级客户')
#
# bw_data_ = pd.DataFrame(bw_data, columns=['账户号', '年收入', '净资产比率', '投资能力', '发展能力'])
# bw_data_.to_excel(writer, sheet_name='百万级客户')
#
# sw_data_ = pd.DataFrame(sw_data, columns=['账户号', '年收入', '净资产比率', '投资能力', '发展能力'])
# sw_data_.to_excel(writer, sheet_name='十万级客户')
#
# w_data_ = pd.DataFrame(w_data, columns=['账户号', '年收入', '净资产比率', '投资能力', '发展能力'])
# w_data_.to_excel(writer, sheet_name='万级客户')
#
# low_w_data_ = pd.DataFrame(low_w_data, columns=['账户号', '年收入', '净资产比率', '投资能力', '发展能力'])
# low_w_data_.to_excel(writer, sheet_name='千元及以下客户')
#
# writer.save()
# writer.close()

print('开始模型计算...')


# 白化权函数
def ranker_11(data_):
    score = []
    for d in data_:
        if d < 0.1:
            score.append(1)
        elif 0.1 <= d <= 0.2:
            score.append((0.2 - d) / 0.1)
        else:
            score.append(0)
    return score

def ranker_12(data_):
    score = []
    for d in data_:
        if 0.1 <= d <= 0.2:
            score.append((d - 0.1) / 0.1)
        elif 0.2 <= d <= 0.55:
            score.append((0.55 - d) / 0.35)
        else:
            score.append(0)
    return score

def ranker_13(data_):
    score = []
    for d in data_:
        if 0.2 <= d <= 0.55:
            score.append((d - 0.2) / 0.35)
        elif 0.55 <= d <= 0.8:
            score.append((0.8 - d) / 0.25)
        else:
            score.append(0)
    return score

def ranker_14(data_):
    score = []
    for d in data_:
        if d < 0.55:
            score.append(0)
        elif 0.55 <= d <= 0.8:
            score.append((d - 0.55) / 0.25)
        else:
            score.append(1)
    return score

def ranker_21(data_):
    score = []
    for d in data_:
        if d < 0.1:
            score.append(1)
        elif 0.1 <= d <= 0.3:
            score.append((0.3 - d) / 0.2)
        else:
            score.append(0)
    return score

def ranker_22(data_):
    score = []
    for d in data_:
        if 0.1 <= d <= 0.3:
            score.append((d - 0.1) / 0.2)
        elif 0.3 <= d <= 0.7:
            score.append((0.7 - d) / 0.4)
        else:
            score.append(0)
    return score

def ranker_23(data_):
    score = []
    for d in data_:
        if 0.3 <= d <= 0.7:
            score.append((d - 0.3) / 0.4)
        elif 0.7 <= d <= 0.9:
            score.append((0.9 - d) / 0.2)
        else:
            score.append(0)
    return score

def ranker_24(data_):
    score = []
    for d in data_:
        if d < 0.7:
            score.append(0)
        elif 0.7 <= d <= 0.9:
            score.append((d - 0.7) / 0.2)
        else:
            score.append(1)
    return score

def ranker_31(data_):
    score = []
    for d in data_:
        if d < 0.01:
            score.append(1)
        elif 0.01 <= d <= 0.05:
            score.append((0.05 - d) / 0.04)
        else:
            score.append(0)
    return score

def ranker_32(data_):
    score = []
    for d in data_:
        if 0.01 <= d <= 0.05:
            score.append((d - 0.01) / 0.04)
        elif 0.05 <= d <= 0.2:
            score.append((0.2 - d)/ 0.15)
        else:
            score.append(0)
    return score

def ranker_33(data_):
    score = []
    for d in data_:
        if 0.05 <= d <= 0.2:
            score.append((d - 0.05) / 0.15)
        elif 0.2 <= d <= 0.31:
            score.append((0.31 - d) / 0.11)
        else:
            score.append(0)
    return score

def ranker_34(data_):
    score = []
    for d in data_:
        if d < 0.2:
            score.append(0)
        elif 0.2 <= d <= 0.31:
            score.append((d - 0.2) / 0.11)
        else:
            score.append(1)
    return score

def ranker_41(data_):
    score = []
    for d in data_:
        if d < 0.2:
            score.append(1)
        elif 0.2 <= d <= 0.35:
            score.append((0.35 - d) / 0.15)
        else:
            score.append(0)
    return score

def ranker_42(data_):
    score = []
    for d in data_:
        if 0.2 <= d <= 0.35:
            score.append((d - 0.2) / 0.15)
        elif 0.35 <= d <= 0.65:
            score.append((0.65 - d) / 0.3)
        else:
            score.append(0)
    return score

def ranker_43(data_):
    score = []
    for d in data_:
        if 0.35 <= d <= 0.65:
            score.append((d - 0.35) / 0.3)
        elif 0.65 <= d <= 0.8:
            score.append((0.8 - d) / 0.15)
        else:
            score.append(0)
    return score

def ranker_44(data_):
    score = []
    for d in data_:
        if d < 0.65:
            score.append(0)
        elif 0.65 <= d <= 0.8:
            score.append((d - 0.65) / 0.15)
        else:
            score.append(1)
    return score


ranker_mat = np.array([[ranker_11, ranker_12, ranker_13, ranker_14],
                       [ranker_21, ranker_22, ranker_23, ranker_24],
                       [ranker_31, ranker_32, ranker_33, ranker_34]]
                      )

# 权值矩阵
weight_mat = np.array([[0.243902439, 0.22222222, 0.261904762, 0.284697509],
                       [0.243902439, 0.33333333, 0.33333333, 0.320284698],
                       [0.024390244, 0.05555556, 0.095238095, 0.110320285],
                       [0.487804878, 0.38888889, 0.30952381, 0.284697509]])


def calculate(data_):
    data_ = data_[:, 1:]
    score_mat_1 = np.zeros(shape=(data_.shape[0], 4, 4))
    for ind in range(4):
        for ranker in range(4):
            score_mat_1[:, ind, ranker] = ranker_mat[ind, ranker](data_[:, ind])
    score_mat_2 = np.zeros(shape=(data_.shape[0], 4))
    for sc in range(4):
        score_mat_2[:, sc] = np.dot(score_mat_1[:, :, sc], weight_mat[:, sc].T)
    return score_mat_2


print('评价最终等级...')
# 计算对各个等级的隶属度
qw_result = calculate(qw_data)
bw_result = calculate(bw_data)
sw_result = calculate(sw_data)
w_result = calculate(w_data)
low_w_result = calculate(low_w_data)

qw_data[:, 1:] = qw_result
bw_data[:, 1:] = bw_result
sw_data[:, 1:] = sw_result
w_data[:, 1:] = w_result
low_w_data[:, 1:] = low_w_result

# 确定最终等级
rank_dict = {0: '差', 1: '中', 2: '良', 3: '优'}


def division(result_):
    result = []
    result_ = np.argmax(result_, axis=1)
    for re in result_:
        result.append(rank_dict[re])
    return result


qw_data = np.column_stack((qw_data, np.array(division(qw_result))))
bw_data = np.column_stack((bw_data, np.array(division(bw_result))))
sw_data = np.column_stack((sw_data, np.array(division(sw_result))))
w_data = np.column_stack((w_data, np.array(division(w_result))))
low_w_data = np.column_stack((low_w_data, np.array(division(low_w_result))))


# 最终数据写入excel表格
print('最终数据写入excel并生成：问题1：最终信用风险等级分类结果.xlsx')
writer = pd.ExcelWriter('问题1：最终信用风险等级分类结果.xlsx')
qw_data = pd.DataFrame(qw_data, columns=['账户号', '差', '中', '良', '优', '分类等级'])
qw_data.to_excel(writer, sheet_name='千万级分类结果')

bw_data = pd.DataFrame(bw_data, columns=['账户号', '差', '中', '良', '优', '分类等级'])
bw_data.to_excel(writer, sheet_name='百万级分类结果')

sw_data = pd.DataFrame(sw_data, columns=['账户号', '差', '中', '良', '优', '分类等级'])
sw_data.to_excel(writer, sheet_name='十万级分类结果')

w_data = pd.DataFrame(w_data, columns=['账户号', '差', '中', '良', '优', '分类等级'])
w_data.to_excel(writer, sheet_name='万级分类结果')

low_w_data = pd.DataFrame(low_w_data, columns=['账户号', '差', '中', '良', '优', '分类等级'])
low_w_data.to_excel(writer, sheet_name='千元及以下分类结果')

writer.save()
writer.close()
print('运行完成')
