import matplotlib.pyplot as plt
import numpy as np

for idx, color in enumerate("rbgy"):  # 此处是不同的颜色
    plt.subplot(2, 2, 1, facecolor=color)
# 使用过大的参数将会有如下警告：
# E:\py\PyCharm 2019.2.3\helpers\pycharm_matplotlib_backend\backend_interagg.py:64: UserWarning:
# Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations
# 用户警告：未应用紧密布局。紧凑的布局不能使轴的高度小到足以容纳所有轴的装饰
# 简单来说就是y轴被挤压的太小太短，可能不足以显示所有y值

x = np.linspace(0, 10, 10000)
y = np.sin(x)
a = plt.subplot(2, 2, 1)  # 先创建图表，下面一行接着是函数，若果有两个，则后一个会替换前一个
print(a)  # 此处将返回一个图表对象
plt.plot(x, y, "r", label="$hahaha$")  # 两个钱的意思是“斜体”
plt.legend()
plt.ylim(-3.0, 3.0)

# 321分别是行数，列数，位置。左上角第一个开始是1
# 三个参数可以分开写，也可以合并起来这样写

plt.show()
