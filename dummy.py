import os
import sys
import optparse
import time
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


from sumolib import checkBinary  # Checks for the binary in environ vars
import traci



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


state_size = 6  
action_size = 3  

# Initialize DQN agent
agent = DQNAgent(state_size, action_size)

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
    



def take_action(action):
    if action == 0:  # accelerate
        traci.vehicle.slowDown("ego_vehicle", 0, 2)
    elif action == 1:  # brake
        traci.vehicle.setSpeed("ego_vehicle", 10)  # Set speed to 10 m/s (adjust as needed)
    else:  # maintain speed
        traci.vehicle.slowDown("ego_vehicle", 0, 0)




def get_reward():
    # Implement reward calculation based on current state and action
    # Example: Give positive reward for maintaining safe distance and negative reward for collisions
    reward = 0
    # Calculate reward based on criteria such as distance and collisions
    return reward

def is_done():
    # Implement episode termination criteria
    # Example: Check if collision occurred or if episode time limit is reached
    done = True
    # Check if episode termination criteria are met
    return done

def run():

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            time.sleep(0.5)
           # Training loop
            batch_size = 1
            EPISODES = 1

            for e in range(EPISODES):
                # Reset environment and get initial state
                traci.simulationStep()
                state = get_state()
                print("State size:", len(state))
                state = np.reshape(state, [1, state_size])
                done = False
                total_reward = 0

                while not done:
                    # Agent takes action
                    action = agent.act(state)
                    print('Action:::::::::::::', action)


                    # Apply action to the SUMO simulation
                    take_action(action)
                    
                    
                    # Environment processes action and returns next state, reward, and done flag
                    traci.simulationStep()
                    next_state = get_state()
                    next_state = np.reshape(next_state, [1, state_size])
                    reward = get_reward()  # Implement a function to calculate the reward
                    done = is_done()  # Implement a function to check if episode is done
                    
                    # Remember the transition
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    
                    # Replay the agent's experiences
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                
                # Print episode statistics
                print("Episode:", e+1, "Total Reward:", total_reward)    

    except traci.exceptions.FatalTraCIError:
        print("Error occurred with TraCI.")


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options



# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "dqn.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()
    print('hello')