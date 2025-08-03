import sys
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from traffic_simulator import TrafficSim
from traffic_simulator import TrafficRenderer

# constants for traffic light actions
RED, GREEN = 0, 1

class TrafficEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, max_cars_dir=20, max_cars_total=30, lambda_ns=2, lambda_ew=3, cars_leaving=5, rewards=None, max_steps=1000): # THEY GET
        """
        Initialize the environment with specified parameters.

        Args:
            max_cars_dir (int): Maximum number of cars allowed in a single direction (north-south or east-west).
            max_cars_total (int): Maximum number of cars allowed in total across both directions.
            lambda_ns (int): Poisson rate parameter for car arrivals in the north-south direction.
            lambda_ew (int): Poisson rate parameter for car arrivals in the east-west direction.
            cars_leaving (int): Number of cars leaving the intersection per timestep.
            rewards (dict): Reward values for different traffic states.
            max_steps (int): Maximum number of steps per episode.
        """

        # set the main parameters
        self.max_cars_dir = max_cars_dir
        self.max_cars_total = max_cars_total
        self.lambda_ns = lambda_ns
        self.lambda_ew = lambda_ew
        self.cars_leaving = cars_leaving

        # set the rewards function
        self.rewards = rewards

        # setting max number of steps per episode and keeping track
        self.max_steps = max_steps
        self.current_step = 0

        # two states for each direction (N/S and E/W) and one for the traffic light
        self.nS = (self.max_cars_dir + 1) ** 2 * 2  # number of states
        self.nA = 2  # number of actions (0: keep, 1: switch)

        # define action and observation spaces
        self.action_space = spaces.Discrete(self.nA)
        sizes = [self.max_cars_dir + 1, self.max_cars_dir + 1, 2]
        self.observation_space = spaces.MultiDiscrete(sizes)

        # initial state distribution
        self.isd = np.indices(sizes).reshape(len(sizes), -1).T

        # initial state
        """
        random_index = np.random.choice(self.isd.shape[0])
        self.s = tuple(self.isd[random_index]) # (ns,ew,light)
        """
        self.s = (0,0,1)

        # initialize simulator with environment parameters
        self.sim = TrafficSim(max_cars_dir, lambda_ns, lambda_ew, cars_leaving, self.s[0], self.s[1], self.s[2])
        # initialize renderer in human mode
        self.renderer = TrafficRenderer(self.sim, "human")

        # Reward setup
        self.rewards = rewards



        # determine the transition probability matrix
        print("Building transition matrix...")
        self.P = self._build_transition_prob_matrix()
        print("Transition matrix built.")

    def _build_transition_prob_matrix(self):
        """Build the transition probability matrix."""
        P = {} # Dictionary to hold the transition probabilities for each state-action pair, with size (ns, ew, light) as keys
        for ns in range(self.max_cars_dir + 1):
            for ew in range(self.max_cars_dir + 1):
                for light in [RED, GREEN]:
                    state = (ns, ew, light)
                    P[state] = {action: [] for action in range(self.nA)}
                    for action in range(self.nA):
                        transitions = []
                        for appr_ns in range(8):
                            for appr_ew in range(8):
                                # determine the next state based on action
                                next_light = abs(light - action)
                                next_ns, next_ew, prob_next_state = self.sim.get_updated_wait_cars(ns, ew, next_light, appr_ns, appr_ew)
                                # get reward
                                reward = self.get_rewards(next_ns, next_ew, next_light)
                                done = self.is_terminal(next_ns, next_ew)
                                next_state = (next_ns, next_ew, next_light)
                                # collect all transitions for normalization
                                transitions += [(prob_next_state, next_state, reward, done)]
                        # normalize the probabilities to ensure they sum to 1
                        total_prob = sum([t[0] for t in transitions])
                        transitions = [(p / total_prob, s, r, d) for (p, s, r, d) in transitions]
                        # assign the normalized transitions to the state-action pair
                        P[state][action] = transitions
        return P

    def get_rewards(self, ns, ew, light):

        # Condition 1: adjusting the wait and queue penalties based on the traffic light state
        ###########################################################################################
        # Terminal state
        # if ns > self.max_cars_dir or ew > self.max_cars_dir or ns + ew > self.max_cars_total:
        #     return -10  # Strong penalty for invalid state
        
        # # Penalize both clearing the intersection and wait time
        # wait_penalty = self.sim.wait_time_ns + self.sim.wait_time_ew
        # queue_penalty = ns + ew

        # # Weighted sum of penalties
        # reward = -0.1 * wait_penalty - 0.9 * queue_penalty
        ###########################################################################################


        # Condition 2: unshaped rewards 
        ###########################################################################################
        # return -(ns + ew)
        ###########################################################################################


        # Condition 3: Shaped rewards
        ###########################################################################################
        # Terminal state
        if ns > self.max_cars_dir or ew > self.max_cars_dir or ns + ew > self.max_cars_total:
            return -10  # Strong penalty for invalid state
        
        # Penalize both clearing the intersection and wait time
        wait_penalty = self.sim.wait_time_ns + self.sim.wait_time_ew
        queue_penalty = ns + ew

        # Weighted sum of penalties
        reward = -0.9 * wait_penalty - 0.1 * queue_penalty


        # Reward for helping clear the intersection
        if light == 1:
            reward += min(ns, self.max_cars_dir)/10
        else:
            reward += min(ew, self.max_cars_dir)/10

        return reward
        ###########################################################################################


    def is_terminal(self, ns, ew):
        """
        Check if the state is terminal.

        Args:
            ns (int): Number of cars in the north-south direction.
            ew (int): Number of cars in the east-west direction.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        if ns > self.max_cars_dir or ew > self.max_cars_dir:
            return True
        if ns + ew > self.max_cars_total:
            return True
        # Testing if the intersection is empty
        # if ns+ew == 0:
        #     return True
        # If the car waiting time exceeds a threshold, consider it terminal
        # if self.sim.wait_time_ns > 2 or self.sim.wait_time_ew > 2:
        #     return True
        else:
            return False
        

    def is_truncated(self):
        """
        Check if the maximum number of steps has been reached.

        Returns:
            bool: True if the maximum number of steps has been reached, False otherwise.
        """
        if self.current_step >= self.max_steps:
            return True
        else:
            return False
        

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action (int): The action to take in the environment.

        Returns:
            tuple: A tuple containing:
                - obs (np.ndarray): The new state after taking the action.
                - r (float): The reward received for taking the action.
                - done (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated due to reaching the max number of steps.
                - dict: Additional information, such as the probability of the transition.
        """

        ns, ew, light = self.s

        # Change light
        if action == 1:
            light = 1 - light
            self.sim.light_ns = light

        # Incoming cars
        cars_approaching_ns, cars_approaching_ew = self.sim.get_approaching_cars()

        # Outgoing cars
        if light == 1:  # NS green
            cars_leaving_ns = min(ns, self.cars_leaving)
            cars_leaving_ew = 0
            
        else:  # EW green
            cars_leaving_ns = 0
            cars_leaving_ew = min(ew, self.cars_leaving)

        # Update car counts
        ns = np.clip(ns - cars_leaving_ns + cars_approaching_ns, 0, self.max_cars_dir)
        ew = np.clip(ew - cars_leaving_ew + cars_approaching_ew, 0, self.max_cars_dir)

        self.s = (ns, ew, light)

        self.current_step += 1

        reward = self.get_rewards(ns, ew, light)
        done = self.is_terminal(ns, ew)
        truncated = self.is_truncated()

        if done or truncated:
            self.current_step = 0

        obs = np.array(self.s, dtype=int)

        info = {
            'cars_approaching_ns': cars_approaching_ns,
            'cars_approaching_ew': cars_approaching_ew,
            'cars_leaving_ns': cars_leaving_ns,
            'cars_leaving_ew': cars_leaving_ew
        }

        return obs, reward, done, truncated, info

        

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        # randomly select the number of cars in the NS and EW directions and the traffic light state
        random_index = np.random.choice(self.isd.shape[0])
        # set the initial state
        s = self.isd[random_index]
        self.s = tuple(s) # (ns,ew,light)
        # reset simulator object
        self.sim.reset(*self.s)
        self.current_step = 0
        if return_info:
            return s, {}
        return s

    def render(self, close=False):
        """Render the environment."""
        if close and self.renderer:
            if self.renderer:
                self.renderer.close()
            return

        if self.renderer:
            return self.renderer.render(*self.s)
