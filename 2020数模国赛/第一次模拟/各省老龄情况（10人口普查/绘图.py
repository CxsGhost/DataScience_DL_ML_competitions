import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig

CurrentConfig.ONLINE_HOST = 'pyecharts-assets-master/pyecharts-assets-master/assets/'

all_data = pd.read_excel('各省老龄化情况.xlsx', header=0)
all_data = all_data.values

zong_data = []
man_data = []
women_data = []
wm_data = []

for i in range(1, len(all_data)):
    zong_data.append(tuple((all_data[i][0].replace(' ', ''), all_data[i][6])))
    man_data.append(tuple((all_data[i][0].replace(' ', ''), all_data[i][7])))
    women_data.append(tuple((all_data[i][0].replace(' ', ''), all_data[i][8])))
    wm_data.append(tuple((all_data[i][0].replace(' ', ''), all_data[i][9])))


zong_data = [('北京', 12.54), ('天津', 13.02), ('河北', 13.0), ('山西', 11.53), ('内蒙古', 11.48), ('辽宁', 15.43), ('吉林', 13.21), ('黑龙江', 13.03), ('上海', 15.07), ('江苏', 15.99), ('浙江', 13.89), ('安徽', 15.01), ('福建', 11.42), ('江西', 11.44), ('山东', 14.75), ('河南', 12.73), ('湖北', 13.93), ('湖南', 14.54), ('广东', 9.73), ('广西', 13.12), ('海南', 11.33), ('重庆', 17.42), ('四川', 16.3), ('贵州', 12.84), ('云南', 11.06), ('西藏', 7.67), ('陕西', 12.85), ('甘肃', 12.44), ('青海', 9.45), ('宁夏', 9.67), ('新疆', 9.66)]
map_ = Map()
map_.add("", zong_data, "china", zoom=1)
map_.set_global_opts(
    title_opts=opts.TitleOpts(title="各省老龄化总体情况",
                              subtitle="数据来源：国家统计局",
                              pos_right="center",
                              pos_top="5%"),
    visualmap_opts=opts.VisualMapOpts(max_=max(all_data[:, 6]),
                                      min_=min(all_data[:, 6]),
                                      range_color=["#E0ECF8", "#045FB4"]),
    )
map_.render("各省老龄化总体情况.html")

map_ = Map()
map_.add("", man_data, "china", zoom=1)
map_.set_global_opts(
    title_opts=opts.TitleOpts(title="各省男性老龄化情况",
                              subtitle="数据来源：国家统计局",
                              pos_right="center",
                              pos_top="5%"),
    visualmap_opts=opts.VisualMapOpts(max_=max(all_data[:, 7]),
                                      min_=min(all_data[:, 7]),
                                      range_color=["#E0ECF8", "#045FB4"]),
    )
map_.render("各省男性老龄化情况.html")

map_ = Map()
map_.add("", women_data, "china", zoom=1)
map_.set_global_opts(
    title_opts=opts.TitleOpts(title="各省女性老龄化情况",
                              subtitle="数据来源：国家统计局",
                              pos_right="center",
                              pos_top="5%"),
    visualmap_opts=opts.VisualMapOpts(max_=max(all_data[:, 8]),
                                      min_=min(all_data[:, 8]),
                                      range_color=["#FBEFEF", '#FA5858', "#DF0101"]),
    )
map_.render("各省女性老龄化情况.html")

map_ = Map()
map_.add("", wm_data, "china", zoom=1)
map_.set_global_opts(
    title_opts=opts.TitleOpts(title="各省男女老龄化比例(女比男)",
                              subtitle="数据来源：国家统计局",
                              pos_right="center",
                              pos_top="5%"),
    visualmap_opts=opts.VisualMapOpts(min_=min(all_data[:, 9]),
                                      max_=max(all_data[:, 9]),
                                      range_color=["#88d3ce", "#6e45e2"]),
    )
map_.render("各省男女老龄化比例.html")

