import pandas as pd
import numpy as np

data = pd.read_excel('weather_data.xls', sheet_name='逐时气象参数',
                     index_col=0, header=0)
data = data.values

# 水平面总辐射强度
H = data[:, 2]

# 水平面散射辐射强度
Hd = data[:, 3]

# 水平面直射辐射强度
Hb = H - Hd

# 太阳时
hour = data[:, 1]

# 大同市维度
phi = 40.1 / 360 * 2 * np.pi

# 南面屋顶斜角
s = np.arctan(3 / 16)

# 地表反射率
rou = 0.08

# 太阳常数
Isc = 1367

# 太阳修正系数
f = 1 + 0.034 * np.cos(0.9863 * (hour // 24 + 1 - 5))


# 赤纬角
def delta(hour_):
    d = 23.45 * np.sin((2 * np.pi * (284 + hour_ // 24 + 1)) / 365)
    d = d / 360 * 2 * np.pi
    return d


# 斜面日落时角
def ws_wh(delta_):
    wh_ = np.arccos(-np.tan(phi) * np.tan(delta_))
    ws_ = np.arccos(-np.tan(phi - s) * np.tan(delta_))
    return np.minimum(wh_, ws_), wh_


# 水平辐射强度与斜面辐强度之比
def Rb(delta_=None, ws_=None, wh_=None):
    a = np.cos(phi - s) * np.cos(delta_) * np.sin(ws_) + \
        ws_ * np.sin(phi - s) * np.sin(delta_)
    b = np.cos(phi) * np.cos(delta_) * np.sin(wh_) + \
        wh_ * np.sin(phi) * np.sin(delta_)
    return a / b


# 计算辐射总量
delta = delta(hour)
ws, wh = ws_wh(delta)
Rb = Rb(delta_=delta, ws_=ws, wh_=wh)
Ho = 24 / np.pi * f * Isc * (np.cos(phi) * np.cos(delta) * np.sin(wh) +
                             wh * np.sin(phi) * np.sin(delta))
Hbt = Hb * Rb
Hdt = Hd * (Hb / Ho * Rb + 0.5 * (1 - Hb / Ho) * (1 + np.cos(s)))
Hrt = 0.5 * rou * H * (1 - np.cos(s))
Ht = Hbt + Hdt + Hrt

delta = delta / (2 * np.pi) * 360
calculate_data = np.column_stack((delta, ws))
calculate_data = np.column_stack((calculate_data, wh))
calculate_data = np.column_stack((calculate_data, Rb))
calculate_data = np.column_stack((calculate_data, Ho))
calculate_data = np.column_stack((calculate_data, Ht))

# 数据写入Excel，并生成
calculate_data = pd.DataFrame(calculate_data)
calculate_data.columns = ['太阳纬度角delta', '斜面日落时角ws', '水平面日落时角wh',
                          '比例系数Rb', '大气层外水平辐射量Ho', '太阳辐射总量Ht']
writer = pd.ExcelWriter('问题1斜面太阳辐射总量及相关参数(1).xlsx')
calculate_data.to_excel(writer, float_format='%.5f')
writer.save()
writer.close()









