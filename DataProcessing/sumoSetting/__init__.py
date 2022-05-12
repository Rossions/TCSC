import traci
import sumolib
import numpy as np
import os
import sys
import optparse
from sumolib import checkBinary


class SumoEnvironment(object):
    def __init__(self, sumo_config_file):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        options = self.__get_options()
        if options.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", sumo_config_file,
                     "--tripinfo-output", "data/tripinfo.xml"])

    @staticmethod
    def __get_options():
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options
