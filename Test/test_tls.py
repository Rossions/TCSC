import os
import sys
import optparse
from sumolib import checkBinary
import traci
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def main():
    phase = [0, 1, 2, 3]
    duration = [5, 10]
    id = traci.trafficlight.getIDList()[0]
    step = 0
    action_phase = random.choice(phase)
    action_duration = random.choice(duration)
    traci.trafficlight.setPhase(id, action_phase)
    tag = step
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if int(action_duration) == (step - tag):
            tag = step
            action_phase = random.choice(phase)
            action_duration = random.choice(duration)
            traci.trafficlight.setPhase(id, action_phase)

        step += 1
    traci.close()
    sys.stdout.flush()


if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "C:/Users/14777/codeStation/TCSC/Test/data/cfg.sumocfg"])
    main()
