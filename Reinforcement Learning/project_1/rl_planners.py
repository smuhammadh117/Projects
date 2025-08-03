import numpy as np

class ValueIterationPlanner:
    """
    Planner that uses the Value Iteration algorithm to determine the optimal policy for a given environment.

    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping value iteration (determines convergence).
    """
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        sizes = [self.env.max_cars_dir + 1, self.env.max_cars_dir + 1, 2] # ns, ew, and light states
        self.all_states = np.indices(sizes).reshape(len(sizes), -1).T # Listing all possible states in row format e.g., row 1 would be ns0, ew0, light0 and last row would be nsN, ewN, lightN
        self.state_to_index = {tuple(state): index for index, state in enumerate(self.all_states)}
        self.policy = np.random.choice([0, 1], size=np.array(sizes).prod())  # initialize policy arbitrarily - size 1D array with size equal to the number of states
        self.value_function = np.zeros(np.array(sizes).prod()) # 1D array to hold the value function for each state - it assignes one value to each state
        self.value_iteration()

    def value_iteration(self):
        """
        Perform value iteration to compute the optimal value function and policy.
        """
        self.deltas = []
        self.l1_errors = []
        self.rewards = []

        prev_value_function = self.value_function.copy()


        self.q_tracking_states = [self.state_to_index[(0, 0, 0)], self.state_to_index[(10,10,1)]]  # states to track Q-values for
        self.q_values_history = {s: [] for s in self.q_tracking_states}
        

        while True:
            delta = 0
            Q = np.zeros((len(self.all_states), self.env.action_space.n), dtype=np.float64) # 2D array to hold the action **values** for each state, initialized to zero
            # iterate over all states
            for state_index, state_array in enumerate(self.all_states):
                state = tuple(state_array)
                # iterate over all actions
                for a in range(self.env.action_space.n):
                    # calculate the expected value of taking action `a` in state `state`
                    for prob, next_state, reward, done in self.env.P[state][a]:
                        next_state_index = self.state_to_index[tuple(next_state)]
                        # not done is used to ensure that the value of the next state is not considered if the episode is done. The value is 1 if not done
                        Q[state_index, a] += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done)) 
                # update the value function with the best action value
                best_action_value = np.max(Q[state_index])
                delta = max(delta, np.abs(self.value_function[state_index] - best_action_value))
                self.value_function[state_index] = best_action_value

            # track Q-values for specific states to monitor convergence
            for state_index in self.q_tracking_states:
                self.q_values_history[state_index].append(Q[state_index].copy())

            if len(self.deltas) % 5 == 0:
                avg_reward = self.evaluate_policy()
                self.rewards.append(avg_reward)

            # Convergence metrics
            l1_error = np.sum(np.abs(self.value_function - prev_value_function))

            self.deltas.append(delta)
            self.l1_errors.append(l1_error)

            # check for convergence
            if delta < self.theta:
                break
        # extract the policy from the Q-table
        for state_index in range(len(self.all_states)):
            self.policy[state_index] = np.argmax(Q[state_index]) # this assigns the action with the highest value to the policy for that state


    def choose_action(self, state):
        """
        Select the action based on the learned policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action chosen by the policy.
        """
        state_index = self.state_to_index[tuple(state)] # returns just the index from a state
        return self.policy[state_index] # gives the optimal action for that state based on the learned policy
    

    def evaluate_policy(self, n_episodes=10):
        """
        Evaluate current policy by running it in the environment and computing average cumulative reward
        """
        total_rewards = []

        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)  



class PolicyIterationPlanner:
    """
    Planner that uses the Policy Iteration algorithm to determine the optimal policy for a given environment.

    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping policy evaluation.
    """

    def __init__(self, env, gamma=0.9, theta=2e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        sizes = [self.env.max_cars_dir + 1, self.env.max_cars_dir + 1, 2] # ns, ew, and light states
        self.all_states = np.indices(sizes).reshape(len(sizes), -1).T # Listing all possible states in row format e.g., row 1 would be ns0, ew0, light0 and last row would be nsN, ewN, lightN
        self.state_to_index = {tuple(state): index for index, state in enumerate(self.all_states)} # mapping from state to index
        self.policy = np.random.choice([0, 1], size=np.array(sizes).prod())  # initialize policy arbitrarily size 1D array with size equal to the number of states
        self.value_function = np.zeros(np.array(sizes).prod()) # 1D array to hold the value function for each state 


        self.policy_changes = []
        self.l1_errors = []
        self.rewards = []
        self.eval_deltas = []


        self.policy_iteration()

    def evaluate_policy(self):
        """
        Evaluate the current policy by computing the value function for all states.

        Returns:
            None
        """
        previous_value_function = np.copy(self.value_function)  # Store the previous value function for convergence check



        self.q_tracking_states = [self.state_to_index[(0, 0, 0)], self.state_to_index[(10,10,1)]]
        self.q_values_history = {s: [] for s in self.q_tracking_states}

        while True:
            delta = 0
            new_value_function = np.zeros_like(self.value_function)  # Reset the value function for this iteration

            for state_index, state_array in enumerate(self.all_states):
                state = tuple(state_array)
                for prob, next_state, reward, done in self.env.P[state][self.policy[state_index]]:
                    next_state_index = self.state_to_index[tuple(next_state)]
                    new_value_function[state_index] += prob * (reward + self.gamma * previous_value_function[next_state_index] * (not done))

            self.value_function = new_value_function  # Update the value function with the new values
            delta = np.max(np.abs(previous_value_function - self.value_function)) # Calculate the maximum change in value function
            
            self.eval_deltas.append(delta)


            if delta < self.theta:
                break

            previous_value_function = np.copy(self.value_function)  # Update the previous value function for the next iteration

            # L1 error for tracking
            l1_error = np.sum(np.abs(self.value_function - previous_value_function))
            self.l1_errors.append(l1_error)

            self.value_function = new_value_function
                    


    def improve_policy(self):
        """
        Improve the current policy by making it greedy with respect to the current value function.

        Returns:
            bool: True if the policy is stable (i.e., no changes were made), False otherwise.
        """
        Q_improve = np.zeros((len(self.all_states), self.env.action_space.n), dtype=np.float64) # size 2D array to hold the action values for each state, initialized to zero

        for state_index, state_array in enumerate(self.all_states):
            state = tuple(state_array)
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[state][a]:
                    next_state_index = self.state_to_index[tuple(next_state)]
                    Q_improve[state_index, a] += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))

        new_policy = np.argmax(Q_improve, axis=1)


        old_policy = self.policy.copy()
        self.policy = new_policy
        changed = np.sum(old_policy != new_policy)
        self.policy_changes.append(changed)

        stable = np.array_equal(self.policy, new_policy)
        self.policy = new_policy # size 1D array with size equal to the number of states, where each element is the action chosen by the policy for that state
        return stable
    

    def policy_iteration(self):
        """
        Perform policy iteration by alternately evaluating and improving the policy.

        Returns:
            np.ndarray: The final policy after convergence.
        """
        random_actions = np.random.choice(self.env.action_space.n, size=len(self.all_states))  # What is this for? This initializes the policy with random actions for each state
        self.policy = random_actions
        iteration = 0


        while True:
            old_policy = np.copy(self.policy)

            self.evaluate_policy()  # Evaluate the current policy
            self.improve_policy() # Improve the policy based on the current value function

            # Evaluate current policy reward every N iterations
            if iteration % 2 == 0:
                avg_reward = self.evaluate_policy_reward()  # New helper function
                self.rewards.append(avg_reward)

            if np.array_equal(old_policy, self.policy):
                break
            iteration += 1

        return self.value_function, self.policy
    

    def evaluate_policy_reward(self, n_episodes=10):
        total_rewards = []
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)


    def choose_action(self, state):
        """
        Select the action based on the learned policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action chosen by the policy.
        """
        state_index = self.state_to_index[tuple(state)] # returns just the index from a state
        return self.policy[state_index] # gives the optimal action for that state based on the learned policy
