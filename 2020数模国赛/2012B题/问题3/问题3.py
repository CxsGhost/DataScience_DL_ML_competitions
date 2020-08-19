import pandas as pd
import numpy as np

class House:
    def __init__(self):
        self.x1 = None
        self.x2 = None
        self.h1 = None
        self.h2 = None
        self.phi = 34 / 360 * 2 * np.pi
        self.s1 = None
        self.s6 = None
        self.s3 = None
        self.s4 = None


house = House()



data = pd.read_excel('weather_data.xls', sheet_name='逐时气象参数',
                     index_col=0, header=0)
data = data.values

north_data = np.sum(data[:, -1])
west_data = np.sum(data[:, -2])
south_data = np.sum(data[:, -3])
east_data = np.sum(data[:, -4])

H = data[:, 2]
Hd = data[:, 3]
Hb = H - Hd
hour = data[:, 1]
phi = 40.1 / 360 * 2 * np.pi
rou = 0.08
Isc = 1367
f = 1 + 0.034 * np.cos(0.9863 * (hour // 24 + 1 - 5))


# 赤纬角
def delta(hour_):
    d = 23.45 * np.sin((2 * np.pi * (284 + hour_ // 24 + 1)) / 365)
    d = d / 360 * 2 * np.pi
    return d


# 斜面日落时角
def ws_wh(delta_, s):
    wh_ = np.arccos(-np.tan(phi) * np.tan(delta_))
    ws_ = np.arccos(-np.tan(phi - s) * np.tan(delta_))
    return np.minimum(wh_, ws_), wh_


# 水平辐射强度与斜面辐强度之比
def Rb(s, delta_=None, ws_=None, wh_=None):
    a = np.cos(phi - s) * np.cos(delta_) * np.sin(ws_) + \
        ws_ * np.sin(phi - s) * np.sin(delta_)
    b = np.cos(phi) * np.cos(delta_) * np.sin(wh_) + \
        wh_ * np.sin(phi) * np.sin(delta_)
    return a / b


delta = delta(hour)


# 计算辐射总量
def calculate(s):
    ws, wh = ws_wh(delta, s)
    Rb_ = Rb(s, delta_=delta, ws_=ws, wh_=wh)
    Ho = 24 / np.pi * f * Isc * (np.cos(phi) * np.cos(delta) * np.sin(wh) +
                                 wh * np.sin(phi) * np.sin(delta))
    Hbt = Hb * Rb_
    Hdt = Hd * (Hb / Ho * Rb_ + 0.5 * (1 - Hb / Ho) * (1 + np.cos(s)))
    Hrt = 0.5 * rou * H * (1 - np.cos(s))
    Ht = Hbt + Hdt + Hrt
    return Ht

cant_data = np.sum(calculate(house.phi))

p = 34 / 360 * 2 * np.pi
def search(x1, x2, h1, h2, c1, c3, c4, c6):
    s1 = x1 * h1
    s2 = x2 * h2
    s3 = x1 * h2
    s4 = 0.5 * (h1 + h2) * x2
    s5 = (h2 - h1) * x1 / np.sin(phi)
    s6 = 0.5 * (h1 + h2) * x2
    b = (c1 + c6 + c3 + c4) / s2
    b1 = c1 / s1
    b3 = c3 / s3
    b4 = c4 / s4
    b6 = c6 / s6
    if s2 + s3 + s4 <= 74 and b >= 0.2 and \
        b1 <= 0.5 and b3 <= 0.3 and b4 <= 0.35 and b6 <= 0.35:
        h_all = s1 * south_data + s3 * north_data + \
                s4 * east_data + s6 * west_data + s5 * cant_data
        return h_all
    else:
        return 0

cs = []
h = []
number = 1
for x1 in np.arange(3, 15.1, 3):
    for x2 in np.arange(3, 15.1, 3):
        for h1 in np.arange(2.8, 5.5, 0.4):
            for h2 in np.arange(2.8, 5.5, 0.4):
                for c1 in np.arange(1, 40.5, 5):
                    for c3 in np.arange(1, 28.35, 5):
                        for c4 in np.arange(1, 28.35, 5):
                            for c6 in np.arange(1, 28.35, 5):
                                result = search(x1, x2, h1, h2, c1, c3, c4, c6)
                                if result:
                                    cs.append([x1, x2, h1, h2, c1, c3, c4, c6])
                                    h.append(result)
cs = cs[np.argmax(h)]
house.x1 = cs[0]
house.x2 = cs[1]
house.h1 = cs[2]
house.h2 = cs[3]
house.s1 = cs[4]
house.s3 = cs[5]
house.s4 = cs[6]
house.s6 = cs[7]
print('最优房屋规格参数为：')
for i in cs:
    print(i)
