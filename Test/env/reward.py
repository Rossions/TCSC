import traci


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
