import os
import sys
import optparse
from sumolib import checkBinary
import traci
from reward import get_reward
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
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        id = traci.trafficlight.getIDList()[0]
        # traci.simulation.
        step += 1
    traci.close()
    sys.stdout.flush()


if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "C:/Users/14777/codeStation/TCSC/DataProcessing/data/2x2/cfg.sumocfg"])
    main()
