import traci
import os
import sys
import optparse
from sumolib import checkBinary
from traci import trafficlight
from traci import simulation


def get_reward(tls_id, last_phase):
    """
    Get the reward of the current state.
    reward designed as waiting vehicle number of the current phase
    :param tls_id:
    :return:reward
    """
    reward = 0
    lane_list = traci.trafficlight.getControlledLanes(tls_id)
    for lane in lane_list:
        reward += - traci.lane.getLastStepHaltingNumber(lane)
    if traci.trafficlight.getPhase(tls_id) == last_phase:
        # 判断动作是否与上一动作一致。
        return reward * 10
    else:
        return reward


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
def get_observation(tls_id):
    current_phase = trafficlight.getPhase(tls_id)
    waiting_vehicle_num = []
    new_lane_list = []
    lane_list = trafficlight.getControlledLanes(tls_id)
    [new_lane_list.append(i) for i in lane_list if i not in new_lane_list]
    for lane in new_lane_list:
        waiting_vehicle_num.append(traci.lane.getLastStepHaltingNumber(lane))
    time = simulation.getTime()
    observation = waiting_vehicle_num
    observation.append(current_phase)
    observation.append(time)
    return observation


# 判断sumo接口是否存在
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


class SumoEnvironment(object):
    observation_space = 6
    action_space = 5
    phase_action_space = 4
    duration_action_space = 1
    action_space_range = [15, 90]

    def __init__(self, cfg_file, trips_info_file="data/trip_info.xml"):
        self.cfg_file = cfg_file
        self.trips_info_file = trips_info_file
        self.time_step = 0
        self.observation_space = 6
        self.action_space = 5
        self.action_space_range = [15, 90]
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        options = get_options()
        if options.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", cfg_file,
                     "--tripinfo-output", trips_info_file])

    # 重置环境，回到初始状态。
    def reset(self, tls_id):
        self.time_step = 0

        traci.load(["-c", self.cfg_file, "--tripinfo-output", self.trips_info_file])
        return get_observation(tls_id)

    # step()用于执行一个动作，最后返回一个元组（observation, reward, done, info）
    # observation (object): 智能体执行动作a后的状态，也就是所谓的“下一步状态s’ ”
    # reward (浮点数) : 智能体执行动作a后获得的奖励
    # done (布尔值): 判断episode是否结束，即s’是否是最终状态？是，则done=True；否，则done=False。
    # info (字典): 一些辅助诊断信息（有助于调试，也可用于学习），一般用不到。
    def step(self, tls_id, action_phase, phase_duration):
        last_action_phase = traci.trafficlight.getPhase(tls_id)
        traci.trafficlight.setPhase(tls_id, action_phase)
        self.time_step += phase_duration
        traci.simulationStep(self.time_step)
        done = traci.simulation.getMinExpectedNumber() < 0
        info = traci.simulation.getTime()
        observation = get_observation(tls_id)
        reward = get_reward(tls_id, last_action_phase)
        return observation, reward, done, info

    def close(self):
        traci.close()
