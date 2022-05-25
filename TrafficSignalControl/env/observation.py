from traci import trafficlight

# 如果交叉路口的速度低于 70km/h 的阈值（可通过选项tls.minor-left.max-speed 进行配置），
# 则允许在迎面而来的直行交通同时左转，但必须让行。这称为 绿色小写，在状态定义中用小写的g表示。否则，
# 左转流必须使用受保护的左转阶段.
# 如果由于没有专用的转弯车道而无法建造这样的阶段，则无论如何都允许绿色未成年人，但会发出警告

# 默认设置信号灯相位
# 直相
# 左转阶段（仅当有专用左转车道时）
# 垂直于第一个方向的直线相位
# 与第一个方向正交的方向的左转阶段（仅当有专用的左转车道时）
# 如果有超过 4 条道路在交叉路口相遇，则会生成额外的绿灯阶段
def get_observation(self):
    traffic_light  = trafficlight.getIDList()
    for id in traffic_light:

