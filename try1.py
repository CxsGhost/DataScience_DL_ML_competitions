import os
base_path = "E:/py/科学计算与机器学习/2021挑战杯/数据"
data_list = ['hua_nan_deal', 'w0003_deal']
classes = os.path.join(base_path, data_list[0])
classes = os.path.join(classes, 'txt/classes.txt')
classes = open(classes, "r+", encoding='utf-8').readlines()
classes = list(classes)
for c in range(len(classes)):
    classes[c] = classes[c].strip()
print(classes)
#%%

data_static = [0 for _ in range(13)]
for f in data_list:
    for d in os.listdir(os.path.join(base_path, f+'/txt')):
        if d == 'classes.txt':
            continue
        file = open(os.path.join(base_path, f+'/txt/'+d), 'r+', encoding='utf-8')
        data = file.readlines()
        for dd in range(len(data)):
            data_static[int(data[dd][0])] += 1
print(data_static)

#%%

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
x = classes
y = data_static

plt.figure(figsize=(8, 5.5))
plt.bar(x, y)
plt.title('数据统计')
plt.ylabel('数量')
plt.xticks(rotation=45)
plt.savefig('数据统计.png')
plt.show()





