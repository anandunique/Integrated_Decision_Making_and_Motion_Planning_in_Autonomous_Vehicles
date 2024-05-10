import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import traci
import os
import sys
import optparse
import time
from sumolib import checkBinary
import math

# Ensure SUMO_HOME is set up correctly
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Set TensorFlow logging level to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# DQN Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Increased memory size to store more experiences
        self.memory = deque(maxlen=5000)  
        # Slow down epsilon decay rate
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Maintain some exploration
        self.epsilon_decay = 0.99  # Reduced decay rate
        self.learning_rate = 0.001
        self.model = self._build_model()

    # Build a simple neural network with 3 layers
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Linear activation for Q-values
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Store experience in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose an action based on epsilon-greedy strategy
    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Exploration
            random_action = random.randrange(self.action_size)
            print("Exploring with action:", random_action)  # Debug log
            return random_action
        
        # Exploitation: Predict the best action based on the current state
        act_values = self.model.predict(state)
        best_action = np.argmax(act_values[0])
        print("Exploiting with action:", best_action)  # Debug log
        return best_action  # returns action with the highest Q-value

    # Train the model using experience replay
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:  # Update target Q-value if not terminal
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)  # Get current Q-values
            target_f[0][action] = target  # Update the Q-value for the selected action
            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train the model
        
        # Update epsilon for exploration-exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon for reduced exploration


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

    return [av_position[0], av_position[1], av_speed, av_distance, presence_of_vehicles, avg_relative_speed]


def get_distance(vehicle1_id, vehicle2_id):
    # Get the positions of the two vehicles
    vehicle1_position = traci.vehicle.getPosition(vehicle1_id)
    vehicle2_position = traci.vehicle.getPosition(vehicle2_id)
    
    # Calculate the Euclidean distance between their positions
    distance = math.sqrt((vehicle2_position[0] - vehicle1_position[0]) ** 2 + (vehicle2_position[1] - vehicle1_position[1]) ** 2)
    
    return distance

# Reward function to avoid collisions and promote safe distances
def get_reward():
    try:
        ego_vehicle_id = "ego_vehicle"
        
        if ego_vehicle_id not in traci.vehicle.getIDList():
            return 0
        
        front_vehicle_id = "vehicle6"
        
        if front_vehicle_id not in traci.vehicle.getIDList():
            return 0
        


        # Calculate the distance between the ego vehicle and the front vehicle
        distance = get_distance(ego_vehicle_id, front_vehicle_id)

        # Check for collisions
        collisions = traci.simulation.getCollidingVehiclesNumber()

        # Reward logic with adjustments for clarity
        if collisions > 0:
            reward = -100  # Strong penalty for collisions
        elif distance <= 5:  # Unsafe distance
            reward = -50
        else:
            reward = 1  # Small positive reward for safe behavior

        return reward
    
    except traci.exceptions.TraCIException as e:
        print("Error occurred while getting reward:", e)
        return 0

def is_done():
    # Define the ID for the ego vehicle
    ego_vehicle_id = "ego_vehicle"
    
    # Get the current simulation time
    simulation_time = traci.simulation.getCurrentTime() / 1000  # Convert to seconds

    # Check if the ego vehicle still exists
    if ego_vehicle_id not in traci.vehicle.getIDList():
        # If the simulation time is greater than 10 seconds and the ego vehicle is missing, end simulation
        if simulation_time > 10:
            return False

    # Check if the simulation has no more expected vehicles and the simulation time is greater than 10 seconds
    if traci.simulation.getMinExpectedNumber() == 0:
        if simulation_time > 10:
            return False

    return True  # Continue simulation


def get_lane_index(ego_vehicle_id):
    # Get the current lane index of the ego vehicle
    lane_id = traci.vehicle.getLaneID(ego_vehicle_id)
    return int(lane_id.split("_")[1])  # Extract the lane index from the lane ID

def get_total_lanes(ego_vehicle_id):
    # Get the total number of lanes for the current road
    edge_id = traci.vehicle.getRoadID(ego_vehicle_id)
    return traci.edge.getLaneNumber(edge_id)

def can_turn_left(ego_vehicle_id):
    # Check if there's a lane to the left and if it's safe to turn
    current_lane = get_lane_index(ego_vehicle_id)
    if current_lane == 0:
        return False  # No lane to the left
    return True  # If there's a lane, it's possible to change

def can_turn_right(ego_vehicle_id):
    # Check if there's a lane to the right and if it's safe to turn
    current_lane = get_lane_index(ego_vehicle_id)
    total_lanes = get_total_lanes(ego_vehicle_id)
    if current_lane == total_lanes - 1:
        return False  # No lane to the right
    return True  # If there's a lane, it's possible to change

def get_traffic_density(edge_id):
    # Calculate traffic density on a specific edge
    num_vehicles = len(traci.edge.getLastStepVehicleIDs(edge_id))
    edge_length = traci.edge.getLength(edge_id)
    return num_vehicles / edge_length  # Density calculation

def can_turn_left(ego_vehicle_id):
    # Check if there's a vehicle on the left
    left_lane = traci.vehicle.getLaneIndex(ego_vehicle_id) - 1
    if left_lane < 0:
        return False  # No left lane to turn into
    nearby_vehicles = traci.vehicle.getLaneID(ego_vehicle_id)
    return len(nearby_vehicles) == 0  # Return True if lane is empty

def can_turn_right(ego_vehicle_id):
    # Check if there's a lane to the right and if it's safe to turn
    current_lane = traci.vehicle.getLaneIndex(ego_vehicle_id)
    edge_id = traci.vehicle.getRoadID(ego_vehicle_id)  # Get current road ID
    total_lanes = traci.edge.getLaneNumber(edge_id)  # Get total lanes on this edge
    
    if current_lane == total_lanes - 1:
        return False  # No lane to the right
    
    # Additional safety check: ensure there's no vehicle in the right lane
    right_lane_id = f"{edge_id}_{current_lane + 1}"  # Lane ID to the right
    nearby_vehicles = traci.lane.getLastStepVehicleIDs(right_lane_id)  # Get vehicles in the right lane
    
    return len(nearby_vehicles) == 0  # Return True if lane is empty


def imminent_collision(ego_vehicle_id):
    # Get the leader (vehicle ahead) of the ego vehicle
    leader_info = traci.vehicle.getLeader(ego_vehicle_id)
    if leader_info:
        leader_id, distance_to_leader = leader_info  # Unpack leader ID and distance
        return distance_to_leader < 10  # Collision risk if within 10 meters
    return False  # No imminent collision

def get_distance_to_front_vehicle(ego_vehicle_id):
    # Get the leader (front vehicle) of the ego vehicle
    leader_info = traci.vehicle.getLeader(ego_vehicle_id)
    if leader_info:
        leader_id, distance_to_leader = leader_info  # Unpack leader ID and distance
        return distance_to_leader  # Return the distance to the front vehicle
    return float('inf')  # If no front vehicle, return infinity

def take_action(action):
    ego_vehicle_id = "ego_vehicle"  # ID of the ego vehicle
    normal_speed = 50  # Default speed
    reduced_speed = 25  # Reduced speed for safety
    cruise_speed = 50  # Desired cruising speed
    safe_distance = 50  # Safe distance for adaptive cruise control

    try:
        # Ensure the ego vehicle exists
        if ego_vehicle_id not in traci.vehicle.getIDList():
            print("Ego vehicle not found in simulation.")
            return

        if action == 0:  # Accelerate to normal speed
            traci.vehicle.setSpeed(ego_vehicle_id, normal_speed)

        elif action == 1:  # Change lane to left with safety check
            if can_turn_left(ego_vehicle_id):  # Only change lanes if safe
                traci.vehicle.changeLane(ego_vehicle_id, 0, 1)  # Leftmost lane

        elif action == 2:  # Change lane to right with safety check
            if can_turn_right(ego_vehicle_id):  # Only change lanes if safe
                traci.vehicle.changeLane(ego_vehicle_id, 1, 1)  # Rightmost lane

        elif action == 3:  # Slow down gradually
            current_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            new_speed = max(current_speed - 5, 0)  # Decrease speed by 5 m/s
            traci.vehicle.setSpeed(ego_vehicle_id, new_speed)
        
        
        elif action == 4:  # Emergency deceleration to avoid collisions
            if imminent_collision(ego_vehicle_id):  # Check for imminent collisions
                traci.vehicle.setEmergencyDecel(ego_vehicle_id, 9.0)  # Set high deceleration rate

        elif action == 5:  # Adaptive cruise control
            distance_to_front_vehicle = get_distance_to_front_vehicle(ego_vehicle_id)  # Distance check
            if distance_to_front_vehicle < safe_distance:  # Too close
                traci.vehicle.setSpeed(ego_vehicle_id, reduced_speed)  # Slow down
            else:
                traci.vehicle.setSpeed(ego_vehicle_id, cruise_speed)  # Maintain speed

        elif action == 6:  # Adjust speed based on traffic density
            edge_id = traci.vehicle.getLaneID(ego_vehicle_id)  # Get current lane
            traffic_density = get_traffic_density(edge_id)  # Example traffic density
            if traffic_density > 0.7:  # High traffic, slow down
                traci.vehicle.setSpeed(ego_vehicle_id, reduced_speed)
            else:  # Normal traffic
                traci.vehicle.setSpeed(ego_vehicle_id, normal_speed)

        else:
            print("Invalid action! No action taken.")

    except traci.exceptions.TraCIException as e:
        print("Error occurred while taking action:", e)



# Simulation runner with reward logging to a file
def run_simulation():
    state_size = 6  # Position (x, y), Speed, Distance, Presence of Other Vehicles
    action_size = 6  # Action size corresponds to the possible actions
    batch_size = 20  # Batch size for experience replay
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    num_simulations = 500  # Number of simulations to run
    
    # Open a file for logging rewards
    log_file_path = "reward_log.txt"  # Define the log file path
    with open(log_file_path, "w") as reward_log:  # Open file in write mode
        reward_log.write("Step,Total Reward,Collisions,Safe Gaps\n")  # Header for log file
        
        # Initialize tracking variables
        total_rewards = []  # Store total reward for each simulation
        collisions = []  # Track -100 rewards (due to collisions)
        safe_gaps = []  # Track -50 rewards (due to unsafe gaps)
        
        for _ in range(num_simulations):
            # Load simulation configuration
            traci.load([
                "-c", "dqn.sumocfg",
                "--start",
                "--quit-on-end",
                "--collision.mingap-factor", "1",
                "--collision.action", "warn",
                "--collision.stoptime", "1",
                "--tripinfo-output", "tripinfo.xml"
            ])
            
            traci.simulationStep()  # Reset simulation state
            step = 1  # Track simulation steps
            total_reward = 0  # Initialize total reward for this simulation
            collision_count = 0  # Reset collision count
            safe_gap_count = 0  # Reset safe gap count
            
            state = np.reshape(get_state(), [1, state_size])  # Get initial state
            
            done = False  # Simulation loop control flag
            
            while not done:
                action = agent.act(state)  # Get action from the agent
                
                # Perform the chosen action
                take_action(action)
                
                # Get the next state and reshape it
                next_state = np.reshape(get_state(), [1, state_size])
                
                # Get the reward and update the total reward
                reward = get_reward()  # Get reward
                
                total_reward += reward  # Accumulate reward for this simulation
                
                # Track collisions and unsafe gaps
                if reward == -100:
                    collision_count += 1  # Count collisions
                elif reward == -50:
                    safe_gap_count += 1  # Count unsafe gaps
                
                # Step the simulation forward
                traci.simulationStep()
                time.sleep(0.5)
                # Update the done condition
                done = not is_done()  # Continue while not done
                
                # Store experience in memory and replay with a larger batch size
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) >= batch_size:
                    agent.replay(batch_size)  # Replay experience for training
                
                state = next_state  # Update the state
                
                step += 1  # Increment step count
            
            # Store data in the lists
            total_rewards.append(total_reward)  # Store total reward for this simulation
            collisions.append(collision_count)  # Store collision count
            safe_gaps.append(safe_gap_count)  # Store unsafe gap count
            
            # Log the rewards for this simulation
            reward_log.write(f"{step},{total_reward},{collision_count},{safe_gap_count}\n")  # Log with step information
            
        # Log the final results for all simulations
        print("All Total Rewards:", total_rewards)  # Summary of total rewards
        print("Collisions:", collisions)  # Summary of collisions
        print("Unsafe Gaps:", safe_gaps)  # Summary of unsafe gaps
    
    # Close SUMO connection
    traci.close()




# Function to get options for SUMO GUI or command-line
def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="Run the command-line version of SUMO")
    options, args = opt_parser.parse_args()
    return options

# Main function to run the simulation
if __name__ == "__main__":
    options = get_options()
    
    # Choose SUMO binary based on GUI option
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # Start SUMO simulation with specified options
    traci.start([sumoBinary, "-c", "dqn.sumocfg", "--collision.mingap-factor", "1", "--collision.action", "warn", "--collision.stoptime", "10", "--tripinfo-output", "tripinfo.xml"])
    
    # Set speed for vehicle6 to simulate traffic
    traci.vehicle.setSpeed("vehicle6", 60)
    
    # Run the simulation
    run_simulation()
