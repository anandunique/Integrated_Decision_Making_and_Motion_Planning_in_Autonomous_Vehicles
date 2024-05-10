AUTONOMOUS VEHICLES-INTEGRATING THE DECISION MAKING AND MOTION PLANNING LAYERS IN A HEIRARCHICAL WAY:
Autonomous navigation enables vehicles to make independent driving decisions, drawing from various factors like sensor data, perception, path planning, localization, and vehicle control. While existing research tackles these aspects with advanced algorithms, many fail to recognize their interconnected nature, leading to inconsistencies and unpredictable behavior, especially when navigating complex scenarios like unsignalized intersections or multi-lane merging.
To address this, we propose a hierarchical and double-layered framework for autonomous decision-making and motion planning. The upper layer focuses on decision-making, utilizing Deep Reinforcement Learning (DRL) algorithms such as DQN and DDQN, while the lower layer handles motion planning with Model Predictive Control (MPC). This structure allows for a more cohesive approach to autonomous vehicle navigation, providing the flexibility to adapt to diverse and challenging traffic conditions.
DRIVING SCENARIO SELECTION:
In autonomous vehicle navigation, selecting the driving scenarios for model training is essential. Among these scenarios, intersections and roundabouts are more critical than multilane roads. Intersections without traffic signals, in particular, are especially challenging due to the lack of signals to regulate traffic flow, leading to higher unpredictability. This makes decision-making crucial at unsignalized intersections, where traffic patterns can be highly random.
TOOLS USED:
We used Python as the programming language and the Simulation of Urban Mobility ( SUMO) simulator as a platform to train the agents.
PYTHON MODULES:
Various python modules are used for numerical computation, random operations, Data Structures and Collections, Deep Learning Frameworks like Keras, Simulation and Traffic Control Interface, Operation System Interface and Command Line Parsing.
ENVIRONMENT SET UP:

NET FILE CONFIGURATION:
Netedit is a graphical tool used to create and edit SUMO networks. It allows users to visually construct roads, junctions, and other network elements for simulation in SUMO. In our, unsignalized intersection network, there are 5 nodes (representing junctions) and 8 edges. Each edge consists of two internal lanes, resulting in a total of 16 lanes in the driving environment.
ROUTE FILE CONFIGURATION:
A vehicle type called car is defined, with characteristics such as length, acceleration, deceleration, a safety margin parameter, a maximum speed 
and a time gap.Several routes are defined in the simulation, including the main route vehicle_route, which consists of edges -gneE2 and gneE3. There are also other routes, such as other_route1, other_route2, and other_route3, which use different combinations of edges. The ego vehicle is designated as ego_vehicle. It uses the vehicle_route route and departs at time 0 from a random lane. The ego vehicle is also marked with a distinct color, red (color="1,0,0"), for easy identification.The ego vehicle is set to follow the vehicle_route, suggesting that it will turn left to reach its destination. This route configuration may require navigating through complex intersections or other traffic conditions. 
4.5.1	TRAINING USING DQN + MPC:
1.	The DQN agent is initialized with the following parameters:
2.	State_size: The number of features in the state space (6).
3.	Action_size: The number of possible actions (6).
4.	Memory: Experience replay buffer with a maximum capacity of 5000 experiences.
5.	Gamma: Discount rate for future rewards (0.95).
6.	Epsilon: Initial exploration rate (1.0).
7.	Epsilon_min: Minimum exploration rate (0.05).
8.	Epsilon_decay: Rate at which exploration rate decays (0.99).
9.	Learning_rate: Learning rate for the neural network optimizer (0.001).

TRAINING USING DDQN + MPC:
Training of the DDQN algorithm uses similar neural network architecture, experience replay, reward function and simulation techniques.
The DDQN parameters are, 
1.	Memory size: 5000
2.	Discount rate (gamma): 0.95
3.	Exploration rate (epsilon): Starting value of 1.0, with epsilon decay rate of 0.995 and minimum epsilon of 0.05
4.	Learning rate: 0.001
5.	Update target network frequency: Every 10 episodes.
CONCLUSION:
The hierarchical and double-layered approach to autonomous navigation demonstrates a significant advancement in decision-making and motion planning. By integrating Deep Reinforcement Learning (DRL) algorithms like DQN and DDQN in the upper layer and Model Predictive Control (MPC) in the lower layer, we achieved a robust and adaptive framework. This approach addressed the critical challenge of coherence by effectively coordinating various tasks such as perception, path planning, and vehicle control
FUTURE WORKS:
The framework may be extended to accommodate interactions with multiple autonomous vehicles and diverse road users, such as pedestrians and cyclists. This would involve the development of sophisticated coordination mechanisms to ensure safe and efficient navigation in mixed traffic scenarios. Mechanisms for the framework to adapt and learn in real-time based on evolving traffic conditions and environmental factors can be incorporated. 
