import pandas as pd
import time
import datetime
from dateutil.parser import parse


def time_processing(timeStamp):
    timeStamp = parse(str(timeStamp))
    timeStamp = time.mktime(timeStamp.timetuple())
    timeArray = time.localtime(timeStamp)
    # 时间时区设置转换
    base_time = datetime.datetime(timeArray[0], timeArray[1], timeArray[2], 0, 0, 0)
    # 获取当日日期定位到00:00:00
    base_time = time.mktime(base_time.timetuple())
    # base_time转变为时间戳格式
    return timeStamp - base_time


def list_processing(list_data):
    for i in range(len(list_data)):
        while (i + 1 < len(list_data)) and (list_data[i] == list_data[i + 1]):
            list_data.pop(i + 1)
    list_data = str(' '.join(list_data))
    return list_data


data_file = 'C:/Users/14777/codeStation/TCSC/DataProcessing\data\chengdu\ps_20161101_e'
names = ["driver_id", "id", "time", "lone"]
data = pd.read_csv(data_file, encoding='utf-8', names=names, header=None, index_col=False)
data = data.groupby(by="id")
with open("../data/chengdu/chengdu_route.rou.xml", "w") as f:
    print('''<?xml version="1.0" encoding="UTF-8"?>''', file=f)
    print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">''', file=f)
    for name, group in data:
        time_data = str(group["time"].values[0])
        time_data = time_processing(time_data)
        lane_list = group["lone"].tolist()
        lane_list = list_processing(lane_list)
        print('''    <vehicle id="{}" depart="{}">'''.format(name[-6:], time_data), file=f)
        print('''        <route edges="{}"/>'''.format(lane_list), file=f)
        print('''    </vehicle>''', file=f)
    print('''</routes>''', file=f)
