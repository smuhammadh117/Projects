import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy as sp
from networks import FCQV 
from networks import FCDP
import pygame
import matplotlib.pyplot as plt
import seaborn as sns  
import Box2D
from ddpg_agent import DDPGAgent
from noise import SimpleGaussianNoiseStrategy
from replay_buffer import ReplayBuffer
import networks
import gymnasium as gym

# Set device
env = gym.make("LunarLanderContinuous-v3", render_mode="human")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize agent
state_dim = env.observation_space.shape[0]
action_bounds = [env.action_space.low, env.action_space.high]
hidden_dims = (32,32,32,32,32,32,30)
agent = DDPGAgent(state_dim, action_bounds, hidden_dims, gamma = 0.99, tau = 0.0052)


# Initialize replay buffer
# buffer = ReplayBuffer(max_size = 50000, batch_size = 64)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=100000)
# print(buffer)

# Hyperparameters
num_episodes = 2000
max_steps = env.spec.max_episode_steps
print(max_steps)
warmup_steps = 800  # used before training starts



best_avg_reward = -np.inf # starting with infinity to increase reward
reward_history = [] # tracking the rewards for analysis later
loss_history = []

noise_strategy = SimpleGaussianNoiseStrategy(
    action_low=-1,   # min action values
    action_high = 1,    # max action values
    noise_std=0.1                 # standard deviation of the noise
)



# Training step

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    for step in range(max_steps):
        # Select action from policy + noise for exploration
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # unsqueeze to
        # Pass the actual policy model instance to the select_action method
        action = noise_strategy.select_action(agent.online_policy_model, state_tensor.squeeze(0)) # Pass the model instance and squeeze state_tensor
        action = np.clip(action, env.action_space.low, env.action_space.high) # restricting it so the action doesn't go above or below the max or min values

        # Step environment
        next_state, reward, done, truncated, _ = env.step(action)
        done_flag = done or truncated

        # Store transition in replay buffer
        # buffer.store((state, action, reward, next_state, float(done_flag)))
        buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Start training after warmup
        if len(buffer) > warmup_steps:
            # experiences = buffer.sample(batch_size=128)
            # Convert to torch tensors
            # states = torch.tensor(experiences[0], dtype=torch.float32)
            # actions = torch.tensor(experiences[1], dtype=torch.float32)
            # rewards = torch.tensor(experiences[2], dtype=torch.float32).unsqueeze(1)
            # next_states = torch.tensor(experiences[3], dtype=torch.float32)
            # dones = torch.tensor(experiences[4], dtype=torch.float32).unsqueeze(1)

            # states = experiences[0].clone().detach()
            # actions = experiences[1].clone().detach()
            # rewards = experiences[2].clone().detach()
            # next_states = experiences[3].clone().detach()
            # dones = experiences[4].clone().detach()

            # agent.optimize_model((states, actions, rewards, next_states, dones))
            states, actions, rewards, next_states, dones = buffer.sample(batch_size=64)
            loss = agent.optimize_model((states, actions, rewards, next_states, dones))
            loss_history.append(loss.item())

        if done_flag:
            break


    reward_history.append(episode_reward)
    avg_reward = np.mean(reward_history[-50:])  # average last 50 episodes

    print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

    # Saving the model if it is improved to learn the policy
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(agent.online_policy_model.state_dict(), "best_policy_model.pth")
        torch.save(agent.online_value_model.state_dict(), "best_value_model.pth")

    if np.mean(reward_history[-40:]) > 200:
      print(f"Condition reached. Stopping at {episode+1}.")
      break

# Evaluation step

num_eval_episodes = 100
evaluate_rewards = []
successful_landings = 0


for ep in range(num_eval_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = agent.online_policy_model(state_tensor).cpu().numpy()[0]

        # Not adding noise in the evaluation step
        action = np.clip(action, env.action_space.low, env.action_space.high)

        next_state, reward, done, truncated, info = env.step(action)
        done_flag = done or truncated

        state = next_state
        episode_reward += reward

        if done_flag:
            if episode_reward > 200:
                successful_landings += 1
            break


    evaluate_rewards.append(episode_reward)

avg_eval_reward = np.mean(evaluate_rewards)
success_rate = successful_landings / num_eval_episodes



# Metrics
print(f"Evaluation over {num_eval_episodes} episodes:")
print(f"Average Reward: {avg_eval_reward:.2f}")
print(f"Success Rate: {success_rate * 100:.1f}%")



# Plotting results
import matplotlib.pyplot as plt

plt.plot(evaluate_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Evaluation Rewards")
plt.show()



# Plot training reward curve
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(reward_history, label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.legend()

# Plot training loss curve
plt.subplot(1,2,2)
plt.plot(loss_history, label='Loss per Training Step')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.tight_layout()
plt.show()