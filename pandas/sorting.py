import pandas as pd
import numpy as np


df = pd.DataFrame({
    'playerID': ['bettsmo01', 'canoro01', 'cruzne02', 'ortizda01', 'cruzne02'],
    'yearID': [2016, 2016, 2016, 2016, 2017],
    'teamID': ['BOS', 'SEA', 'SEA', 'BOS', 'SEA'],
    'HR': [31, 39, 43, 38, 39]})

# 单元素排序
df1 = df.sort_values('yearID', ascending=False)
print(df1)
# False是按照升序排序

# 多元素排序
df2 = df.sort_values(['yearID', 'HR'], ascending=[True, False])
print(df2)
# 先按照year排序，然后在每个year内部，再按照HR进行排序
# 在前的元素优先等级高
