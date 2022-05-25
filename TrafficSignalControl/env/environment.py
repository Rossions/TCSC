import traci
import os
import sys
import optparse
from sumolib import checkBinary


# 判断sumo接口是否存在
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options



class SumoEnvironment(object):
    def __init__(self, cfg_file, trips_info_file="data/trip_info.xml"):
        self.cfg_file = cfg_file
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
    def reset(self):
        traci.load(["-c", self.cfg_file])

    # step()用于执行一个动作，最后返回一个元组（observation, reward, done, info）
    # observation (object): 智能体执行动作a后的状态，也就是所谓的“下一步状态s’ ”
    # reward (浮点数) : 智能体执行动作a后获得的奖励
    # done (布尔值): 判断episode是否结束，即s’是否是最终状态？是，则done=True；否，则done=False。
    # info (字典): 一些辅助诊断信息（有助于调试，也可用于学习），一般用不到。
    def step(self, tls_id, action_phase, phase_duration):
        done = traci.simulation.getMinExpectedNumber() > 0
        info = traci.simulation.getTime()
        observation = get_road_info()
        reward = None
        return observation, reward, done, info

    def sumo_close(self):
        traci.close()
