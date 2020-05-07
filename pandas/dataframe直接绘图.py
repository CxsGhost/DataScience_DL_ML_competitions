import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6],
                   'www': [7, 8, 7]})
print(df)

df.plot()  # 会自动根据行列标签进行绘图
plt.show()

plt.savefig('df.png')  # 可以保存图像，PDF或者PNG格式

df.plot()
plt.title('wwwww')
plt.xlabel('ooo')
plt.ylabel('111111')  # 自行设置xy轴的标签
plt.show()

# 上述绘图中，r1 r2 r3只是不同的数据点，c1 c2 c3是三个不同特征，三条线
'-----------------'

# 上述绘制的是普通线形图
# 还可以绘制别的图
df.plot(kind='hist')  # 直方图
df.plot(kind='box')  # 箱线图（可以观测离群值）
plt.show()

