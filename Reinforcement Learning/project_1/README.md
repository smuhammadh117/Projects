# Project 1: Reinforcement Learning for Traffic Light Control

This project explores the application of reinforcement learning (RL) algorithms to optimize traffic light control at a single intersection. The goal of this project is to optimize the traffic signals in the east-west and north-south directions for the most effective flow of traffic throught them; while avoiding threshold exceedances in a custom Gymnasium environment.

---

## Author

**Syed Muhammad Husainie (shusainie3)**  
GT OMSCS - Reinforcement Learning Summer 2025  
GitHub repo: [GT GitHub Repository](https://github.gatech.edu/gt-omscs-rldm/7642RLDMSummer2025shusainie3)

---

## Project Structure


```
├── traffic_environment.py   # Custom Gym environment (TrafficEnv): handles transitions, rewards, terminal states, reset, and rendering
├── traffic_simulator.py     # Defines the probabilistic traffic arrival distributions
├── rl_planners.py           # Implements value iteration and policy iteration
├── rl_agents.py             # Implements Q-Learning and SARSA agents
├── traffic_execution.py     # Entry point: runs the chosen algorithm and saves performance metrics and plots
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Setup Instructions

### Environment

- Python 3.8 or higher


---

### Installation

```bash
pip install -r requirements.txt
```

---

### Running the Code

To run the code, carry out the following command in terminal:


```bash
python traffic_execution.py
```

The traffic_environment.py file includes several reward function options, which are commented for clarity.

To test different reward strategies (e.g., penalizing wait time vs. queue length), uncomment the relevant sections in the reward function.