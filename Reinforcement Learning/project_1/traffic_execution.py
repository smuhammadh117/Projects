import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import rl_agents
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

np.random.seed(42)


# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = TrafficEnv(rewards = rewards, max_steps=1000)

# set the RL algorithm to plan or train an agent
rl_algo = "SARSA"  # Options: "Value Iteration", "Policy Iteration", "Q-Learning", "SARSA"


value_it = []
policy_it = []


# initialize the agent and train it
if rl_algo == "Value Iteration":
    print("\nRunning Value Iteration...")
    agent = rl_planners.ValueIterationPlanner(env)

    print("\nLearned Value Iteration Policy (State -> Action):")
    for state_index, state_array in enumerate(agent.all_states):
        state = tuple(state_array)
        action = agent.policy[state_index]
        value_it.append({"ns":state[0], "ew":state[1], "light":state[2], "action":action})
    policy_df = pd.DataFrame(value_it)
    print(policy_df)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot delta
    axs[0].plot(agent.deltas)
    axs[0].set_title("Max Delta per Iteration")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Delta")
    axs[0].grid(True)

    # Plot L1 error
    axs[1].plot(agent.l1_errors)
    axs[1].set_title("L1 Error (Value Function Change)")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("L1 Error")
    axs[1].grid(True)

    # Plot cumulative rewards (evaluated every 5 iterations)
    axs[2].plot(range(0, len(agent.rewards) * 5, 5), agent.rewards)
    axs[2].set_title("Average Cumulative Reward")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Avg Reward")
    axs[2].grid(True)

    # tight layout for better spacing
    plt.tight_layout()
    plt.show()

    # Plot Q-values

    for state_index, q_values_over_time in agent.q_values_history.items():
        q_values_array = np.array(q_values_over_time)  # shape: (iterations, actions)
        for action in range(q_values_array.shape[1]):
            plt.plot(q_values_array[:, action], label=f"State {agent.all_states[state_index]} - Action {action}")
    plt.xlabel("Iteration")
    plt.ylabel("Q-Value")
    plt.title("Q-Values over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Add watermark
    # fig.text(0.5, 0.5, "shusainie3", fontsize=200, color='gray', alpha=0.2,
    #      ha='center', va='center', rotation=45)

    # plt.tight_layout()
    # plt.show()


if rl_algo == "Policy Iteration":
    print("\nRunning Policy Iteration...")
    agent = rl_planners.PolicyIterationPlanner(env)
    print("\nLearned Policy Iteration Policy (State -> Action):")

    for state_index, state_array in enumerate(agent.all_states):
        state = tuple(state_array)
        action = agent.policy[state_index]
        policy_it.append({"ns":state[0], "ew":state[1], "light":state[2], "action":action})
    policy_df = pd.DataFrame(policy_it)

    print(policy_df)


    plt.figure(figsize=(8, 4))
    plt.plot(agent.eval_deltas, marker='o')
    plt.title("Change in Value Function Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Δ Delta Value Function ")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    
    plt.figure(figsize=(8, 4))
    plt.plot(agent.policy_changes, marker='o')
    plt.title("Policy Changes per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Changed Actions")
    plt.grid(True)
    plt.show()





if rl_algo == "Q-Learning":
    print("\nRunning Q-Learning...")
    agent = rl_agents.QLearningAgent(env, gamma=0.75, alpha=0.1, epsilon=0.1, epsilon_decay=0.99, episodes=2000)
    agent.train()

    print("\nLearned Q-Learning Policy (State -> Action):")
    Q = agent.q_table
    all_states = np.array(list(np.ndindex(Q.shape[:-1])))
    q_learning_policy = []
    for state in all_states:
        ns, ew, light = state
        action = int(np.argmax(Q[tuple(state)]))
        q_learning_policy.append({"ns":ns, "ew":ew, "light":light, "action":action})
    policy_df = pd.DataFrame(q_learning_policy)
    print(policy_df)
        
        

if rl_algo == "SARSA":
    print("\nRunning SARSA...")
    agent = rl_agents.SARSAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000)
    agent.train()

    print("\nLearned SARSA Policy (State -> Action):")
    Q = agent.q_table

    all_states = np.array(list(np.ndindex(Q.shape[:-1])))
    sarsa_policy = []
    for state in all_states:
        ns, ew, light = state
        action = int(np.argmax(Q[tuple(state)]))
        sarsa_policy.append({"ns":ns, "ew":ew, "light":light, "action":action})
    policy_df = pd.DataFrame(sarsa_policy)
    print(policy_df)

cmap = ListedColormap(['red', 'green'])

# Heatmap of the learned policy
if policy_df is not None:

    for light_val in [0, 1]:
        pivot_df = policy_df[policy_df['light'] == light_val].pivot(index='ns', columns='ew', values='action')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, cmap='RdYlGn', cbar=True, vmin=0, vmax=1)
        plt.title(f'Policy Heatmap for light = {light_val} (1 = NS green, 0 = EW green)')
        plt.xlabel('EW Cars')
        plt.ylabel('NS Cars')
        plt.show()


# Learning curve for Q-Learning or SARSA
if rl_algo in ["Q-Learning", "SARSA"]:
    plt.plot(agent.q_table.max(axis=-1).flatten())
    plt.title(f'Learning Curve ({rl_algo})')
    plt.xlabel('State Index')
    plt.ylabel('Max Q-Value')
    plt.grid()
    plt.show()

# Q table value heatmap for Q-Learning or SARSA
if rl_algo in ["Q-Learning", "SARSA"]:
    q_values = agent.q_table.max(axis=-1).reshape(env.max_cars_dir + 1, env.max_cars_dir + 1, 2)
    plt.imshow(q_values[:, :, 0], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Max Q-Value for NS Cars')
    plt.title('Q-Values Heatmap for NS Cars')
    plt.xlabel('EW Cars')
    plt.ylabel('NS Cars')
    plt.show()

    plt.imshow(q_values[:, :, 1], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Max Q-Value for EW Cars')
    plt.title('Q-Values Heatmap for EW Cars')
    plt.xlabel('EW Cars')
    plt.ylabel('NS Cars')
    plt.show()



# TODO: Initialize variables to track performance metrics
# Metrics to include:
# 1. Count of instances where car count exceeds critical thresholds (N total cars or M in any direction)
# 2. Average number of cars waiting at the intersection in all directions during a time period
# 3. Maximum continuous time where car count remains below critical thresholds
exceed_count = 0
total_waiting_cars = 0
timesteps = 0
below_threshold_count = 0
current_below_threshold_duration = 0
max_below_threshold_duration = 0

exceed_count_array = []
total_waiting_cars_array = []
timesteps_array = []
below_threshold_count_array = []
current_below_threshold_duration_array = []
max_below_threshold_duration_array = []
reward_total = []

# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}

# TODO: Initialize variables to track environment metrics
# Example: cumulative rewards, episode duration, etc.
cumulative_reward = 0
episode_duration = 0

# set light state variables
RED, GREEN = 0, 1

# run the environment until terminated or truncated
terminated, truncated = False, False
while (not terminated and not truncated):
    # use the agent's policy to choose an action
    action = agent.choose_action(observation)
    # step through the environment with the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Information: {info}")


    # TODO: Update variables to calculate performance and environment metrics based on the new observation

    ns, ew, light = tuple(observation)
    total_cars = ns + ew


    timesteps += 1
    cumulative_reward += reward

    total_waiting_cars += total_cars

    if (ns > env.max_cars_dir or ew > env.max_cars_dir or (ns+ew) > env.max_cars_total):
        below_threshold_count += 1
        exceed_count += 1
        current_below_threshold_duration = 0
    else:
        current_below_threshold_duration += 1
        if current_below_threshold_duration > max_below_threshold_duration:
            max_below_threshold_duration = current_below_threshold_duration

    light_color = "GREEN" if light == GREEN else "RED"
    # print the current state
    print(f"Step: {timesteps}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    # render the environment at each step
    env.render()
    # add a delay to slow down the rendering for better visualization
    time.sleep(0.8)

    total_waiting_cars_array.append(total_waiting_cars)
    timesteps_array.append(timesteps)
    below_threshold_count_array.append(below_threshold_count)
    current_below_threshold_duration_array.append(current_below_threshold_duration)
    max_below_threshold_duration_array.append(max_below_threshold_duration)
    reward_total.append(cumulative_reward)

    # reset the environment if terminated or truncated
    if terminated or truncated:
        print("\nTERMINATED OR TRUNCATED, RESETTING...\n")

        # TODO: Update metrics for completed episode
        print(f"Episode duration: {timesteps} steps")
        print(f"Episode cumulative reward: {cumulative_reward}")
        print(f"Episode threshold exceed count: {exceed_count}")

        if timesteps > 0:
            avg_waiting_cars = total_waiting_cars / timesteps 
        else:
            avg_waiting_cars = 0

        print(f"Episode average cars waiting: {avg_waiting_cars:.2f}")
        print(f"Episode max continuous below threshold duration: {max_below_threshold_duration}")
        


        observation, info = env.reset(), {}

        # TODO: Reset tracking variables for the new episode
        exceed_count = 0
        total_waiting_cars = 0
        timesteps = 0
        cumulative_reward = 0
        current_below_threshold_duration = 0
        max_below_threshold_duration = 0


# TODO: Evaluate performance based on high-level metrics

# Make a pandas DataFrame to summarize the performance metrics



performance_data = {
    "Timesteps": timesteps_array,
    "Total Waiting Cars": total_waiting_cars_array,
    "Below Threshold Count": below_threshold_count_array,
    "Current Below Threshold Duration": current_below_threshold_duration_array,
    "Max Below Threshold Duration": max_below_threshold_duration_array,
    "Total Reward": reward_total
}

print("\nPerformance Metrics Summary:")


performance_df = pd.DataFrame(performance_data)


# Average waiting cars
avg_wait = [total / steps if steps > 0 else 0 for total, steps in zip(total_waiting_cars_array, timesteps_array)]
plt.plot(avg_wait)
plt.title("Average Waiting Cars per Episode")
plt.xlabel("Steps")
plt.ylabel("Avg Waiting Cars")
plt.grid()
plt.show()


# Max Below Threshold Duration
plt.plot(max_below_threshold_duration_array)
plt.title("Max Safe Duration per Episode")
plt.xlabel("Steps")
plt.ylabel("Count")
plt.grid()
plt.show()

# Total Reward per Episode
plt.plot(reward_total)  # Or per episode if tracked
plt.title("Reward per Step")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.grid()
plt.show()



# close the environment
env.render(close=True)