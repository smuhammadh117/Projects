import gymnasium as gym
import time

env = gym.make('LunarLander-v3', render_mode='human')  # enable rendering window
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  # show animation frame
    action = env.action_space.sample()  # random action
    obs, reward, done_flag, truncated, info = env.step(action)
    done = done_flag or truncated
    total_reward += reward
    time.sleep(0.05)  # slow down for visibility (optional)

print(f"Episode finished with total reward: {total_reward}")
env.close()
