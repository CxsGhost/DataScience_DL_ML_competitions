import matplotlib.pyplot as plt
import numpy as np
data = np.array([i for i in range(25)])
data_1 = data.reshape(5, 5)
print(data_1)

plt.figure(figsize=(8, 8))
plt.bar(data_1[0, :], data_1[1, :], color=["r", "b", "y"],  width=0.5, label='data')
# bar函数用于绘制柱状统计图，其中的width参数用于控制柱的宽度
plt.xlabel("person")
plt.ylabel('number')
plt.ylim(-30, 30)

plt.legend()
plt.show()
