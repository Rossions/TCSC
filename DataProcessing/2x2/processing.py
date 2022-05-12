import pandas as pd
import numpy as np
import json

trace_file = "/data/2x2/sumoTrace.xlsx"


def distinct(list1):
    list2 = []
    for i in list1:
        if not i in list2:
            list2.append(i)
    return list2


trace_data = pd.read_excel(trace_file)
# 分组
grouped = trace_data.groupby(by="id")
trace_all_list = []
# 按照分组将各个车辆轨迹中交通信号节点进行提取
for name, group in grouped:
    trace_list = []
    for data in group.itertuples():
        lane_name = getattr(data, "lane")
        if lane_name.startswith(":"):
            lane_name = lane_name.split("_")[0][1:]
            trace_list.append(lane_name)
            trace_list = distinct(trace_list)
    trace_all_list.append(trace_list)
with open('trace.txt', 'w', encoding='utf-8') as f:
    for i in trace_all_list:
        for j in i:
            f.write(j)
            f.write(' ')
        f.write('\n')
    f.close()
