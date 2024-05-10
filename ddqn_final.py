import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import traci 
import math
import os
import sys
import optparse
import time
from sumolib import checkBinary



# Ensure SUMO_HOME is set up correctly
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Set TensorFlow logging level to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Double DQN Agent Class
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Experience replay memory
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.learning_rate = 0.001  # Learning rate
        self.update_target_network_freq = 10  # Frequency to update target network
        self.model = self._build_model()  # Main Q-network
        self.target_model = self._build_model()  # Target Q-network
        self.update_target_network()  # Initialize target network

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())  # Sync target with main network

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Store experience

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Exploration
            return random.randrange(self.action_size)  # Random action
        act_values = self.model.predict(state)  # Predict Q-values
        return np.argmax(act_values[0])  # Best action

    def replay(self, batch_size, episode_num):
        minibatch = random.sample(self.memory, batch_size)  # Sample minibatch
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:  # If not terminal state
                next_action = np.argmax(self.model.predict(next_state)[0])  # Best action from main network
                target += self.gamma * self.target_model.predict(next_state)[0][next_action]  # Use target network for Q-value
            
            target_f = self.model.predict(state)  # Current Q-values
            target_f[0][action] = target  # Update selected action's Q-value
            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train the model

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Reduce epsilon
            
        # Update target network every 10 episodes
        if episode_num % self.update_target_network_freq == 0:
            self.update_target_network()  # Sync target network

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


def run_simulation():
    state_size = 6  # Position (x, y), Speed, Distance, Presence of Other Vehicles
    action_size = 6  # Action size corresponds to the possible actions
    batch_size = 20  # Batch size for experience replay
    
    # Initialize DDQN agent
    agent = DDQNAgent(state_size, action_size)
    
    num_simulations = 500  # Number of simulations to run
    
    # Declare lists to track rewards and specific counts
    total_rewards = []  # List to store total rewards for each simulation
    collision_counts = []  # List to track the number of collisions
    safe_gap_counts = []  # List to track the number of unsafe gaps
    
    # Open a file for logging rewards
    log_file_path = "ddqn.txt"  # Define the log file path
    with open(log_file_path, "w") as reward_log:  # Open file in write mode
        reward_log.write("Step,Total Reward,Collisions,Safe Gaps\n")  # Header for log file
        
        episode_num = 1  # Initialize episode counter
        step = 1  # Track simulation steps
        
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
            
            state = np.reshape(get_state(), [1, state_size])  # Get initial state
            total_reward = 0  # Initialize total reward for this episode
            
            collision_count = 0  # Initialize collision count for this episode
            safe_gap_count = 0  # Initialize unsafe gap count for this episode
            
            done = False  # Simulation loop control flag
            
            while not done:
                action = agent.act(state)  # Get action from the agent
                
                # Perform the chosen action
                take_action(action)
                
                # Get the next state and reshape it
                next_state = np.reshape(get_state(), [1, state_size])
                
                # Get the reward and update the total reward
                reward = get_reward()
                total_reward += reward
                
                # Track collisions and unsafe gaps
                if reward == -100:
                    collision_count += 1  # Count collisions
                elif reward == -50:
                    safe_gap_count += 1  # Count unsafe gaps
                
                # Step the simulation forward
                traci.simulationStep()
                
                # Update the done condition
                done = not is_done()  # Continue while not done
                
                # Store experience in memory and replay with batch size and episode number
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) >= batch_size:
                    agent.replay(batch_size, episode_num)  # Pass episode_num to replay
                
                state = next_state  # Update the state
            
            # Log the reward and counts for each step
            reward_log.write(f"{step},{total_reward},{collision_count},{safe_gap_count}\n")  # Log reward and counts with step number
            
            # Append rewards and counts to their respective lists
            total_rewards.append(total_reward)
            collision_counts.append(collision_count)
            safe_gap_counts.append(safe_gap_count)
            
            step += 1  # Increment step count
            episode_num += 1  # Increment episode count
        
        print("All Total Rewards:", total_rewards)  # Summary of total rewards
    
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