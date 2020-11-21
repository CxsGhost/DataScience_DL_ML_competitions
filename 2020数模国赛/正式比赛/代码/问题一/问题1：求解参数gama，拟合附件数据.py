import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
plt.rcParams['font.sans-serif'] = ['SimHei']

# 读取附件数据
attach_data = None
try:
    attach_data = pd.read_excel('附件.xlsx', header=0,
                                index_col=None, sheet_name='Sheet1')
except:
    print('请将附件数据与代码放于同一目下！')
attach_data = attach_data.values

length = 435.5 / 100  # 经过的总长度

rou_panel = 1859  # 电路板密度
thick_panel = 0.0015  # 焊接区域厚度

# 电路板比热容三次样条插值
x_temp_panel = [20, 80, 120, 160, 200, 225, 240, 260, 280]
cp_panel = [1100, 1400, 1500, 1550, 1600, 1610, 1640, 1665, 1690]
f_panel_cp = interpolate.interp1d(x_temp_panel, cp_panel, kind='cubic')

# 空气各项指标随温度变化插值
x_temp_air = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350]

# 动力粘度插值
eta_air = [1.81e-5, 1.86e-5, 1.91e-5, 1.96e-5, 2.01e-5, 2.06e-5, 2.11e-5, 2.15e-5,
           2.19e-5, 2.28e-5, 2.37e-5, 2.45e-5, 2.53e-5, 2.6e-5, 2.74e-5, 2.97e-5, 3.14e-5]
f_air_eta = interpolate.interp1d(x_temp_air, eta_air, kind='cubic')

# 导热系数插值
lamb_air = [2.593e-2, 2.675e-2, 2.756e-2, 2.826e-2, 2.896e-2, 2.966e-2, 3.047e-2, 3.128e-2,
            3.21e-2, 3.338e-2, 3.489e-2, 3.64e-2, 3.78e-2, 3.931e-2, 4.268e-2, 4.605e-2, 4.908e-2]
f_air_lamb = interpolate.interp1d(x_temp_air, lamb_air, kind='cubic')

# 定压比热容插值
cp_air = [1013, 1013, 1013, 1017, 1017, 1017, 1022, 1022, 1022, 1026,
          1026, 1026, 1034, 1034, 1043, 1047, 1055]
f_air_cp = interpolate.interp1d(x_temp_air, cp_air, kind='cubic')

# 流体粘度插值
u_air = [1.502e-5, 1.597e-5, 1.693e-5, 1.793e-5, 1.869e-5, 2.002e-5, 2.11e-5, 2.212e-5,
         2.315e-5, 2.539e-5, 2.775e-5, 3.006e-5, 3.248e-5, 3.485e-5, 4.065e-5, 4.829e-5, 5.548e-5]
f_air_u = interpolate.interp1d(x_temp_air, u_air, kind='cubic')


# T-air, 炉内空气温度
def T_air(v_, t_, t1_, t2_, t3_, t4_):
    air_list = []
    way = v_ * t_
    for w in way:
        if 0.0 <= w < 0.25:
            air_list.append((t1_ - 25) / 0.25 * w + 25)
        elif 0.25 <= w < 1.975:
            air_list.append(t1_)
        elif 1.975 <= w < 2.025:
            air_list.append((t2_ - t1_) / 0.05 * w - (t2_ - t1_) / 0.05 * 1.975 + t1_)
        elif 2.025 <= w < 2.33:
            air_list.append(t2_)
        elif 2.33 <= w < 2.38:
            air_list.append((t3_ - t2_) / 0.05 * w - (t3_ - t2_) / 0.05 * 2.33 + t2_)
        elif 2.38 <= w < 2.685:
            air_list.append(t3_)
        elif 2.685 <= w < 2.935:
            air_list.append((t4_ - t3_) / 0.05 * w - (t4_ - t3_) / 0.05 * 2.685 + t3_)
        elif 2.935 <= w < 3.395:
            air_list.append(t4_)
        elif 3.395 <= w < 4.105:
            air_list.append((25 - t4_) / 0.7 * w - (25 - t4_) / 0.7 * 3.395 + t4_)
        else:
            air_list.append(25)
    return np.array(air_list)


# 计算hc，对流热传导率
def calculate_hc(temp_, gama_=None):
    return 0.664 * np.power(gama_, 0.5) * \
           np.power(f_air_eta(temp_) * f_air_cp(temp_), 1 / 3) * \
           np.power(f_air_lamb(temp_), 2 / 3) / np.power(f_air_u(temp_), 0.5)


# 计算求解偏导数
def calculate_ds(hc_, k_, t_air_, temp_):
    yz = hc_ + k_ * (np.square(t_air_) + np.square(temp_)) * (t_air_ + temp_)
    return yz * (t_air_ - temp_) * (1 / (rou_panel * f_panel_cp(temp_) * thick_panel))


def solve_equation(gama=None, v_=70, t1_=175, t2_=195, t3_=235, t4_=255):
    v_ = v_ / 6000  # 转换单位
    second_max = int(length / v_)
    time_list = np.arange(0, second_max + 0.5, 0.5)
    temp_list = [25]
    ds_list = []
    t_air_list = T_air(v_, time_list, t1_, t2_, t3_, t4_)
    t_air_list_h2 = T_air(v_, time_list + 0.25, t1_, t2_, t3_, t4_)
    K = (5.669e-8 * 0.8 * 0.98) / (0.8 + 0.98 - 1)

    # 龙格库塔法求解
    for i in range(len(time_list) - 1):
        hc = calculate_hc(temp_list[i], gama_=gama)
        k1 = calculate_ds(hc, K, t_air_list[i], temp_list[i])
        k2 = calculate_ds(hc, K, t_air_list_h2[i], temp_list[i] + 0.25 * k1)
        k3 = calculate_ds(hc, K, t_air_list_h2[i], temp_list[i] + 0.25 * k2)
        k4 = calculate_ds(hc, K, t_air_list[i + 1], temp_list[i] + 0.5 * k3)
        ds_list.append(k1)
        temp_list.append(temp_list[i] + 0.5 / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    time_list = np.array(time_list)[-len(attach_data):]
    temp_list = np.array(temp_list)[-len(attach_data):]

    # 计算误差
    loss = np.sum(np.abs(attach_data[:, 1] - temp_list)) / len(temp_list)
    return loss, time_list, temp_list


# 在宽泛搜索的前提下以小步长细致搜索
print('正在搜索最优的伽马值...')
ga_x = np.arange(240, 380, 2)
loss_list = []
for ga in ga_x:
    loss_list.append(solve_equation(gama=ga)[0])

best_gama = ga_x[np.argmin(loss_list)]
print('能够最优拟合附件数据的gama为：{}'.format(best_gama))

# 生成gama与loss的关系图
print('生成gama与loss关系图...')
plt.figure(figsize=(8, 6))
plt.plot(ga_x, loss_list, color='b', marker='o', markersize=3)
plt.xlabel('Gama', fontdict={'size': 14})
plt.ylabel('Loss', fontdict={'size': 14})
plt.savefig('gama-loss关系图.png')
plt.show()


loss, time_list, temp_list = solve_equation(gama=best_gama)

# 与附件数据做对比，生成拟合效果图
plt.figure(figsize=(8, 6))
plt.plot(time_list, temp_list, color='b', label='拟合数据')
plt.plot(time_list, attach_data[:, -1], color='r', label='附件数据')
plt.xlabel('时间', fontdict={'size': 13})
plt.ylabel('温度', fontdict={'size': 13})
plt.legend(fontsize='x-large')
plt.savefig('对比图.png')
plt.show()






