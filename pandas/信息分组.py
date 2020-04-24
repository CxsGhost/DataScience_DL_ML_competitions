import pandas as pd

df1 = pd.DataFrame({'name': ['john doe', 'al smith', 'sam black', 'john doe'],
                    'pos': ['1B', 'C', 'P', '2B'],
                    'year': [2000, 2004, 2008, 2003]})

groups = df1.groupby('year')  # 以某个列表标签进行数据分组,元组形式存储
for year, data in groups:
    print(year)
    print(data)
'----------------------------------'
group1 = groups.get_group(2003)  # 此函数取其中的特定组
print(group1)
'--------------------'
print(groups.sum(axis=1))  # 可以指定行列
print(groups.mean())
print(groups.median())
# 以列标签为基准，计算每年每种数据的平均，中位数， 求和
# 但必须是数字才行
'--------------------------------------------------'
no2000 = groups.filter(lambda x: x.name > 2000)
print(no2000)
# 过滤器，根据分类的标签，进行条件过滤，lambda返回false的过滤掉
'---------------------------------------------'
groups_2 = df1.groupby(['year', 'pos'])  # 可以选择多个列标签分类，类别元组返回，下面j
for j, k in groups_2:
    print(j)

fil = groups_2.filter(lambda x: x.name[0] > 2000)  # 以其中某个标签过滤
print(fil)
'----------------------------------------------'

