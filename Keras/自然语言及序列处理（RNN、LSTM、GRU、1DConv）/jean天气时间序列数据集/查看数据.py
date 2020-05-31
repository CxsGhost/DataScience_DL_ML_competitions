# 观察jena天气数据集
import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'E:/py/科学计算与机器学习/Keras/自然语言及序列处理（RNN、LSTM、GRU、1DConv）' \
           '/jean天气时间序列数据集/jena_climate_2009_2016.csv'

data_file_name = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

# 舍去数字索引，使用时间列做数据索引
data = pd.read_csv(data_file_name, index_col=1)

print(len(data.index))
# 共420551个数据

print(len(data.columns))
# 共有15个指标，第一个是日期，后14个是与天气相关的指标

# 读取为numpy数据
data = data.values
print(data.shape)

# 取出温度指标
temperature = data[:, 1]

plt.plot(range(len(temperature)), temperature)

# 每10分钟记录一次，所以一天144次，10天1440次
plt.plot(range(1440), temperature[: 1440], color='g', linestyle='-', marker='o', markersize=2)

plt.show()

"""
如果按年度序列来看，可以看到明显的周期性
但是10天的数据，看起来变化比较凌乱
我们来试试以天为尺度在，这个数据是否是可预测的
"""






















