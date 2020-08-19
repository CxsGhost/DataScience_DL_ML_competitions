import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig, OnlineHostType

CurrentConfig.ONLINE_HOST = 'pyecharts-assets-master/pyecharts-assets-master/assets/'

data = pd.read_excel('浙江人口数据.xlsx', header=0)
data = data.values
print(data)
data_18 = []
data_10 = []
for i in range(len(data)):
    data_18.append(tuple((data[i][0].replace(' ', ''), data[i][2])))
    data_10.append(tuple((data[i][0].replace(' ', ''), data[i][4])))
print(data_18)
print(data_10)

map_ = Map()
map_.add("", data_18, "浙江", zoom=1)
map_.set_global_opts(
    title_opts=opts.TitleOpts(title="2018浙江各市老龄化情况",
                              subtitle="数据来源：浙江统计局",
                              pos_right="center",
                              pos_top="5%"),
    visualmap_opts=opts.VisualMapOpts(min_=min(data[:, 2]),
                                      max_=max(data[:, 2]),
                                      range_color=["#C9ECDD", "#0B614B"]),
    )
map_.render("2018浙江各市老龄化情况.html")

map_ = Map()
map_.add("", data_10, "浙江", zoom=1)
map_.set_global_opts(
    title_opts=opts.TitleOpts(title="2010浙江各市老龄化情况",
                              subtitle="数据来源：浙江统计局",
                              pos_right="center",
                              pos_top="5%"),
    visualmap_opts=opts.VisualMapOpts(min_=min(data[:, -1]),
                                      max_=max(data[:, -1]),
                                      range_color=["#FFFFFF",  "#088A68"]),
    )
map_.render("2010浙江各市老龄化情况.html")