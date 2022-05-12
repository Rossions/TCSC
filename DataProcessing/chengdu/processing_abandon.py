import numpy as np
import pandas as pd
import datetime, time


# 处理输入时间戳，当前汽车驶入时间戳转化为sumo中以秒为单位
def time_processing(timeStamp):
    timeArray = time.localtime(timeStamp)
    # 时间时区设置转换
    base_time = datetime.datetime(timeArray[0], timeArray[1], timeArray[2], 0, 0, 0)
    # 获取当日日期定位到00:00:00
    base_time = time.mktime(base_time.timetuple())
    # base_time转变为时间戳格式
    return timeStamp - base_time


def create_trip_file(data_file="../data/chengdu/20161116.csv"):
    names = ["id", "start_time", "end_time", "time?", "from_lane", "to_lane"]
    data = pd.read_csv(data_file, header=None, names=names, index_col=False)
    # 行索引命名，列索生成
    data = data.sort_values(by='start_time', ascending=True)
    # 排序升序排序
    with open("../data/chengdu/20161116_trips.trips.xml", mode="w") as f:
        print('''<?xml version="1.0" encoding="UTF-8"?>
    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
        ''', file=f)
        for index, data_line in data.iterrows():
            data_line["start_time"] = time_processing(data_line["start_time"])
            print(
                '''    <trip id="{}" depart="{}" from="{}" to="{}"/>'''.format(data_line['id'], data_line['start_time'],
                                                                               data_line['from_lane'],
                                                                               data_line['to_lane']),
                file=f)
            print(
                '''    <trip id="{}" depart="{}" from="{}" to="{}"/>'''.format(data_line['id'], data_line['start_time'],
                                                                               data_line['from_lane'],
                                                                               data_line['to_lane']), )
        print('''</routes>''', file=f)
