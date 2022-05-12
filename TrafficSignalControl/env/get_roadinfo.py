import traci


# 去掉id_list中冒号开头的路段
def remove_colon(id_list):
    id_list = list(id_list)
    id_list_new = []
    for name in id_list:
        if name[0] != ':':
            id_list_new.append(name)
    return id_list_new


# 获取edge上的各种车辆信息(reward)。
# 利用字典嵌套存储每个edge上的车辆信息。
# 返回值：edge_info_dict
def get_road_info():
    edge_info = {}
    edge_list = traci.edge.getIDList()
    edge_list = remove_colon(edge_list)
    for edge in edge_list:
        # 获取道路上车辆数量
        vehicle_number = traci.edge.getLastStepVehicleNumber(edge)
        # 获取道路上车辆平均速度
        mean_speed = traci.edge.getLastStepMeanSpeed(edge)
        # 获取道路被车辆占用的时间百分比
        occupancy = traci.edge.getLastStepOccupancy(edge)
        # 获取道路上车辆等待时间
        wait_time = traci.edge.getWaitingTime(edge)
        # 获取道路上车辆平均长度
        mean_car_length = traci.edge.getLastStepLength(edge)
        # 获取道路上停车车辆数量
        parking_number = traci.edge.getLastStepHaltingNumber(edge)
        edge_info[edge] = {'vehicle_number': vehicle_number, 'mean_speed': mean_speed, 'occupancy': occupancy,
                           'wait_time': wait_time, 'mean_car_length': mean_car_length, 'parking_number': parking_number}
    return edge_info
