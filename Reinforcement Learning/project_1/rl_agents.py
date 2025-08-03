import numpy as np

class Agent:
    """
    Base RL agent class for Q-Learning and SARSA.

    Args:
        env (gym.Env): The environment to train in.
        gamma (float): Discount factor for future rewards.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay factor for epsilon.
        episodes (int): Number of training episodes.
    """

    def __init__(self, env, gamma=0.75, alpha=0.5, epsilon=0.9, epsilon_decay=0.99, episodes=2000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.q_table = np.zeros((*self.env.observation_space.nvec, self.env.action_space.n))

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (tuple): The current environment state

        Returns:
            int: The action chosen by the agent.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space.n)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action according to Q-table

class QLearningAgent(Agent):
    """Q-Learning agent implementing the off-policy TD control."""

    def train(self):
        """
        Train the agent over a fixed number of episodes using the Q-Learning algorithm.
        Note: After training, set epsilon to 0 to deploy the learned policy without further exploration.

        Returns:
            None
        """
        Q = self.q_table

        for ep in range(self.episodes):
            state = tuple(self.env.reset())
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = tuple(next_state)

                td_target = reward + self.gamma * np.max(Q[next_state]) * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] += self.alpha * td_error

                state = next_state

            # Decay epsilon
            self.epsilon *= self.epsilon_decay

class SARSAgent(Agent):
    """SARSA agent implementing the on-policy TD control."""

    def train(self):
        """
        Train the agent over a fixed number of episodes using the SARSA algorithm.
        Note: After training, set epsilon to 0 to deploy the learned policy without further exploration.

        Returns:
            None
        """
        Q = self.q_table # 4D array with shape (N, M, L, A) where N, M, L are the dimensions of the state space and A is the number of actions

        for ep in range(self.episodes):
            state = tuple(self.env.reset())
            done = False
            action = self.choose_action(state)

            while not done:
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = tuple(next_state)
                next_action = self.choose_action(next_state)

                td_target = reward + self.gamma * Q[next_state][next_action] * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] += self.alpha * td_error

                state, action = next_state, next_action

            # Decay epsilon
            self.epsilon *= self.epsilon_decay
