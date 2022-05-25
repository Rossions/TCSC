import traci


def get_reward(tls_id):
    """
    Get the reward of the current state.
    reward designed as waiting vehicle number of the current phase
    :param tls_id:
    :return:reward
    """
    reward = 0
    lane_list = traci.trafficlight.getControlledLanes(tls_id)
    for lane in lane_list:
        reward += traci.lane.getLastStepHaltingNumber(lane)
    return reward