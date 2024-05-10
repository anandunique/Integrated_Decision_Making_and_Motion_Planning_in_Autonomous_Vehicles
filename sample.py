import os
import sys
import optparse
import time
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def run():
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # Advance the simulation
            traci.simulationStep()
            time.sleep(0.5)
            vid = "ego_vehicle"
            vid2 = "vehicle5"
            # traci.vehicle.setSpeed(vid2, 60)
            # traci.vehicle.setSpeed("vehicle3", 75)
            # traci.vehicle.setSpeedMode(vid, 5)  # Disable speed safety checks
            # Example usage:
            ego_vehicle_id = "ego_vehicle"
            front_vehicle_id = "vehicle6"

            distance = get_distance(ego_vehicle_id, front_vehicle_id)
            print("Distance between ego vehicle and front vehicle:", distance)

        if traci.simulation.getCollidingVehiclesNumber() > 0:
            # If collision occurred, return a negative reward
            print('Collision occurred')
        else:
            print('No collision')
            # get_state()

    except traci.exceptions.FatalTraCIError:
        print("Error occurred with TraCI.")
    


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def check(vehID):
    print("vehicles", traci.vehicle.getIDList())
    print("vehicle count", traci.vehicle.getIDCount())
    print("examining", vehID)
    print("speed", traci.vehicle.getSpeed(vehID))
    print("speedLat", traci.vehicle.getLateralSpeed(vehID))
    print("speed w/o traci", traci.vehicle.getSpeedWithoutTraCI(vehID))
    print("acceleration", traci.vehicle.getAcceleration(vehID))
    print("pos", traci.vehicle.getPosition(vehID))
    print("pos3D",traci.vehicle.getPosition3D(vehID))
    print("angle", traci.vehicle.getAngle(vehID))
    print("road", traci.vehicle.getRoadID(vehID))
    print("lane", traci.vehicle.getLaneID(vehID))
    print("laneIndex", traci.vehicle.getLaneIndex(vehID))
    print("type", traci.vehicle.getTypeID(vehID))
    print("routeID", traci.vehicle.getRouteID(vehID))
    print("routeIndex", traci.vehicle.getRouteIndex(vehID))
    print("route", traci.vehicle.getRoute(vehID))
    print("lanePos", traci.vehicle.getLanePosition(vehID))
    print("color", traci.vehicle.getColor(vehID))
    print("bestLanes", traci.vehicle.getBestLanes(vehID))
    print("CO2", traci.vehicle.getCO2Emission(vehID))
    print("CO", traci.vehicle.getCOEmission(vehID))
    print("HC", traci.vehicle.getHCEmission(vehID))
    print("PMx", traci.vehicle.getPMxEmission(vehID))


def get_state():
    av_position = traci.vehicle.getPosition("ego_vehicle")
    av_speed = traci.vehicle.getSpeed("ego_vehicle")
    av_distance = traci.vehicle.getDistance("ego_vehicle")

    other_vehicles = list(traci.vehicle.getIDList())  # Convert tuple to list
    other_vehicles.remove("ego_vehicle")  # Remove AV from the list
    
    presence_of_vehicles = 1 if len(other_vehicles) > 0 else 0
    
    # Initialize a list to store relative speeds of other vehicles
    relative_speeds = []
    
    for vehicle_id in other_vehicles:
        # Get speed of other vehicle
        other_vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
        
        # Calculate relative speed (difference between AV speed and other vehicle speed)
        relative_speed = av_speed - other_vehicle_speed
        
        # Append relative speed to the list
        relative_speeds.append(relative_speed)
    
    # Calculate average relative speed
    avg_relative_speed = np.mean(relative_speeds) if relative_speeds else 0.0
    
    
    
    # Print the state for testing purposes
    print("AV Position:", av_position)
    print("AV Speed:", av_speed)
    print("AV Distance:", av_distance)
    print("Presence of Vehicles:", presence_of_vehicles)
    print("Average Relative Speed:", avg_relative_speed)


import math
def get_distance(vehicle1_id, vehicle2_id):
    # Get the positions of the two vehicles
    vehicle1_position = traci.vehicle.getPosition(vehicle1_id)
    vehicle2_position = traci.vehicle.getPosition(vehicle2_id)
    
    # Calculate the Euclidean distance between their positions
    distance = math.sqrt((vehicle2_position[0] - vehicle1_position[0]) ** 2 + (vehicle2_position[1] - vehicle1_position[1]) ** 2)
    
    return distance





    
# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')





   # Start SUMO simulation
    traci.start([sumoBinary, "-c", "dqn.sumocfg",  "--collision.mingap-factor", "1", "--collision.action", "warn", "--collision.stoptime", "10", "--tripinfo-output", "tripinfo.xml"])

    num_simulations = 5

    for _ in range(num_simulations):
        # Run the simulation
        run()
        print('Simulation completed. Resetting environment.')

        # Reset the simulation environment
        traci.load(["-c", "dqn.sumocfg", "--start", "--quit-on-end", "--collision.mingap-factor", "1", "--collision.action", "warn", "--collision.stoptime", "10", "--tripinfo-output", "tripinfo.xml"])

    # Close the simulation environment after all simulations are complete
    traci.close()

    
