import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

# Implementing the training step for MAPPO

class MAPPOTrainer:
    def __init__(self, env, agent, buffer, num_agents, obs_dim, action_dim,  device='cpu',
                 rollout_length=128, log_interval=1000,
                 gamma=0.99, lam=0.95, max_episodes=10000, min_samples_per_update = 5):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.device = device

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.rollout_length = rollout_length
        self.log_interval = log_interval
        self.gamma = gamma
        self.lam = lam
        self.max_episodes = max_episodes

        # Initialize attributes to store training progress
        self.episode_rewards = []
        self.total_soups = []

        self.min_samples_per_update = min_samples_per_update
        self.episode_event_log = []


        self.placement_in_pot_log = []
        self.current_placement_count = 0

        self.onion_pickup_log = []
        self.dish_pickup_log = []
        self.soup_delivery_log = []

        self.onion_pickup = 0
        self.dish_pickup = 0
        self.soup_delivery_counts = 0

        self.soup_tracker = 0




    def run(self):
        episode = 0
        min_episodes_per_update = self.min_samples_per_update
        update_episode_count = 0


        ptr_count = 0

        while episode < self.max_episodes:
            # done = False
            total_soups = 0
            samples_collected = 0
            total_ep_reward = np.zeros(self.num_agents)



            # obs = self.env.reset() # obs is a dict with key "both_agent_obs"
            self.buffer.reset()





            while samples_collected < min_episodes_per_update:


                done = False
                ep_reward = np.zeros(self.num_agents)
                num_soups_made = 0

                obs = self.env.reset()




                while not done:
                  obs_tensor = torch.tensor(obs["both_agent_obs"], dtype=torch.float32, device=self.device)

                  with torch.no_grad():
                      actions, log_probs, values = self.agent.act(obs_tensor)


                  next_obs, R, done, info = self.env.step(actions)




                  game_stats = self.env.base_env.game_stats
                  r_shaped = info.get("shaped_r_by_agent", [0]*self.num_agents)


                  placement_rewards = []
                  for agent_idx in range(self.num_agents):
                      if r_shaped[agent_idx] == 8:
                        self.current_placement_count += 1






                  # Access event counts for specific types

                  useful_onion_pickups = [0, 0]
                  useful_onion_drops = [0, 0]
                  useful_dish_pickups = [0, 0]
                  optimal_onion_pottings = [0, 0]
                  soup_delivery_counts = [0, 0]
                  onion_pickup = [0, 0]
                  soup_pickup = [0, 0]
                  viable_onion_potting = [0, 0]
                  soup_drop = [0, 0] 
                  dish_pickup = [0, 0]

                  # Get event list from game_stats to add to reward
                  event_keys = {
                      "useful_onion_pickup": useful_onion_pickups,
                      "useful_onion_drop": useful_onion_drops, 
                      "useful_dish_pickup": useful_dish_pickups,
                      "optimal_onion_potting": optimal_onion_pottings,
                      "soup_delivery": soup_delivery_counts,
                      "dish_pickup": dish_pickup,
                      "onion_pickup": onion_pickup,
                      "soup_pickup": soup_pickup,
                      "viable_onion_potting": viable_onion_potting,
                      "soup_drop": soup_drop
                  }


                  for event_name, counter in event_keys.items():
                      event_data = game_stats.get(event_name, None)
                      if event_data:
                          for i, events in enumerate(event_data):
                              counter[i] = len(events)


                  self.onion_pickup += np.sum(onion_pickup)
                  self.dish_pickup += np.sum(dish_pickup)
                  self.soup_delivery_counts += np.sum(soup_delivery_counts)

                  # print(self.onion_pickup)


                  def normalize(arr, factor=100.0):
                      return np.array(arr, dtype=np.float32) / factor

                  # Normalize each stat (adjust factor as appropriate per stat)
                  u_onion_pickups = normalize(useful_onion_pickups)
                  u_onion_drops = normalize(useful_onion_drops)
                  u_dish_pickups = normalize(useful_dish_pickups)
                  opt_onion_pottings = normalize(optimal_onion_pottings)
                  soup_delivs = normalize(soup_delivery_counts) * 20
                  onion_pickups = normalize(onion_pickup)
                  soup_picks = normalize(soup_pickup)   
                  viable_onion_pottings = normalize(viable_onion_potting)
                  soup_drops = normalize(soup_drop) 


                  # Rearrange based on agent_idx (to match shaped reward convention)
                  if hasattr(self.env, 'agent_idx') and self.env.agent_idx:
                      u_onion_pickups = u_onion_pickups[[1, 0]]
                      u_onion_drops = u_onion_drops[[1, 0]]
                      u_dish_pickups = u_dish_pickups[[1, 0]]
                      opt_onion_pottings = opt_onion_pottings[[1, 0]]
                      soup_delivs = soup_delivs[[1, 0]]
                      onion_pickups = onion_pickups[[1, 0]]
                      soup_picks = soup_picks[[1, 0]]
                      viable_onion_pottings = viable_onion_pottings[[1, 0]]
                      soup_drops = soup_drops[[1, 0]]


                  # Stack all stats together into a final tensor per agent
                  # This gives a shape of (num_agents, num_features)
                  all_stats_np = np.sum([
                      u_onion_pickups,
                    #   u_onion_drops,
                      u_dish_pickups,
                      opt_onion_pottings,
                      soup_delivs,
                      onion_pickups,
                      soup_picks
                    #   viable_onion_pottings,
                    #   soup_drops
                  ], axis=0)  # shape = (2 agents, )

                  

                  # Convert to tensor
                  all_stats_tensor = torch.tensor(all_stats_np, dtype=torch.float32, device=self.device)
                  # print(all_stats_tensor)




                  # Convert shaped rewards properly considering env.agent_idx logic
                  # r_shaped = info.get("shaped_r_by_agent", [0]*self.num_agents)
                  # if any(r != 0 for r in r_shaped):
                  #     print("Shaped rewards by agent:", r_shaped)


                  if hasattr(self.env, 'agent_idx') and self.env.agent_idx:
                      r_shaped_0, r_shaped_1 = r_shaped[1], r_shaped[0]
                  else:
                      r_shaped_0, r_shaped_1 = r_shaped[0], r_shaped[1]




                  if isinstance(R, (int, float)):
                      sparse_rewards = [R] * self.num_agents
                  else:
                      sparse_rewards = R

                  num_soups_made += int(R / 20)



                  # Convert shaped rewards to tensor
                  rewards_tensor = torch.tensor([r_shaped_0, r_shaped_1], dtype=torch.float32, device=self.device)

                  self.soup_tracker =  num_soups_made



                  combined_rewards = [s + d + a for s, d, a in zip(sparse_rewards, rewards_tensor, all_stats_tensor)]
                  combined_rewards_tensor = torch.tensor(combined_rewards, dtype=torch.float32, device=self.device)

                  # print(f"Combined rewards: {combined_rewards_tensor}")
                  # print(f"Sparse rewards: {sparse_rewards}")
                  # print(f"Dense rewards: {dense_rewards}")


                  dones_tensor = torch.tensor([done]*self.num_agents, dtype=torch.float32, device=self.device)



                  self.buffer.store(obs_tensor, actions, log_probs, combined_rewards_tensor, dones_tensor, values)
                  # print(f"Buffer pointer after storing: {self.buffer.ptr}")

                  obs = next_obs
                  ep_reward += combined_rewards_tensor.cpu().numpy()




                samples_collected += 1  # counting timesteps


                episode += 1

                self.placement_in_pot_log.append(self.current_placement_count)
                self.current_placement_count = 0
                # print(self.placement_in_pot_log)
                self.onion_pickup_log.append(self.onion_pickup)
                self.dish_pickup_log.append(self.dish_pickup)
                self.soup_delivery_log.append(self.soup_delivery_counts)

                self.onion_pickup = 0
                self.dish_pickup = 0
                self.soup_delivery_counts = 0

                # print(self.onion_pickup_log)


                total_ep_reward += ep_reward
                total_soups += num_soups_made

                self.episode_rewards.append(np.mean(ep_reward))
                self.total_soups.append(num_soups_made)





            # Once we've collected enough samples (or episodes), do update:

            # print(f"Ptr: {self.buffer.ptr}")
            with torch.no_grad():
                last_obs_tensor = torch.tensor(obs["both_agent_obs"], dtype=torch.float32, device=self.device)
                joint_obs = last_obs_tensor.view(-1).unsqueeze(0)
                next_value = self.agent.get_value(joint_obs)


            # print("DEBUGGER")
            self.buffer.compute_advantages(next_value, self.gamma, self.lam)
            self.agent.update(self.buffer)

            # # Optionally log episode stats
            # self.episode_rewards.append(np.mean(ep_reward))
            # self.total_soups.append(num_soups_made)

            avg_onions = np.mean(self.placement_in_pot_log[-samples_collected])
            avg_ep_reward = np.mean(self.episode_rewards[-samples_collected:])
            avg_soups = np.mean(self.total_soups[-samples_collected:])


            print(f"Update {update_episode_count+1}, Episode {episode}:, Avg Episode Reward = {avg_ep_reward:.2f}, Avg Soups Made = {avg_soups:.2f}")
            # print(f"Reward for game stats by each agent, respectively: {all_stats_tensor}")
            # print(f"Cumulated rewards: {combined_rewards_tensor}")


            update_episode_count += 1

            # Reset counters
            samples_collected = 0
            self.buffer.reset()

            # print(self.buffer.ptr)


            # Early stop if soups threshold reached
            if np.mean(self.total_soups[-50:]) >= 8:
                break


            episode += 1

        print("Training complete.")
        self.agent.save("policy_layout3.pt")
        print("Policy saved to policy_layout3.pt")




    def evaluate(self, num_episodes=100):

        episode_rewards = []
        total_soups_per_episode = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            obs = torch.tensor(obs["both_agent_obs"], dtype=torch.float32, device=self.device)

            done = False
            ep_reward = np.zeros(self.num_agents)
            num_soups_made = 0  # Reset soups count for this episode

            while not done:
                with torch.no_grad():
                    actions, _, _ = self.agent.act(obs, deterministic = False)  # act without learning

                next_obs, rewards, dones, info = self.env.step(actions)
                next_obs = torch.tensor(next_obs["both_agent_obs"], dtype=torch.float32, device=self.device)

                obs = next_obs
                ep_reward += rewards

                if dones:
                    done = True

                num_soups_made += int(np.sum(rewards) / 20)  # sum rewards if rewards is an array

            episode_rewards.append(np.mean(ep_reward))
            total_soups_per_episode.append(num_soups_made)
            print(f"Eval Episode {episode + 1}: Avg Reward per agent = {np.mean(ep_reward):.2f}, Soups made = {num_soups_made}")

        avg_reward = np.mean(episode_rewards)
        avg_soups = np.mean(total_soups_per_episode)
        print(f"Evaluation over {num_episodes} episodes, average reward: {avg_reward:.2f}")
        print(f"Average soups made per episode: {avg_soups:.2f}")

        if len(episode_rewards) < 50:
          print(f"Not enough data points ({len(episode_rewards)}) for a rolling average with window size {50}. Plotting raw data instead.")
          rolling_avg = avg_reward
          x_values = range(len(episode_rewards))
        else:
          rolling_avg = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
          # The x-axis for rolling average needs to be adjusted.
          # It typically corresponds to the end of the window.
          x_values = range(50 - 1, len(episode_rewards))

        plt.figure(figsize=(12, 7))
        plt.plot(x_values, rolling_avg, label=f'Rolling Average (Window {50})')
        plt.xlabel('Episodes')
        plt.ylabel('Average Rewards')
        plt.title(f'Average Rewards during Evaluation(Rolling Average)')
        plt.legend()
        plt.grid(True)
        plt.show()


        if len(total_soups_per_episode) < 50:
          print(f"Not enough data points ({len(total_soups_per_episode)}) for a rolling average with window size {50}. Plotting raw data instead.")
          rolling_avg = avg_soups
          x_values = range(len(total_soups_per_episode))
        else:
          rolling_avg = np.convolve(total_soups_per_episode, np.ones(50)/50, mode='valid')
          # The x-axis for rolling average needs to be adjusted.
          # It typically corresponds to the end of the window.
          x_values = range(50 - 1, len(total_soups_per_episode))

        plt.figure(figsize=(12, 7))
        plt.plot(x_values, rolling_avg, label=f'Rolling Average (Window {50})')
        plt.xlabel('Episodes')
        plt.ylabel('Number of Soups')
        plt.title(f'Number of Soups Made during Evaluation (Rolling Average)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_rolling_average(self,data, title, ylabel, window_size=50):
      """
      Plots the rolling average of a list of data.

      Args:
          data (list or np.ndarray): The raw data (e.g., episode rewards, soups made).
          title (str): The title for the plot.
          ylabel (str): The label for the Y-axis.
          window_size (int): The size of the rolling window for the average.
      """
      if not data:
          print(f"No data available for plotting '{title}'. The list is empty.")
          return

      data_np = np.array(data)

      if len(data_np) < window_size:
          print(f"Not enough data points ({len(data_np)}) for a rolling average with window size {window_size}. Plotting raw data instead.")
          rolling_avg = data_np
          x_values = range(len(data_np))
      else:
          rolling_avg = np.convolve(data_np, np.ones(window_size)/window_size, mode='valid')
          # The x-axis for rolling average needs to be adjusted.
          # It typically corresponds to the end of the window.
          x_values = range(window_size - 1, len(data_np))


      plt.figure(figsize=(12, 7))
      plt.plot(x_values, rolling_avg, label=f'Rolling Average (Window {window_size})')
      plt.xlabel('Episodes')
      plt.ylabel(ylabel)
      plt.title(f'{title} (Rolling Average)')
      plt.legend()
      plt.grid(True)
      plt.show()


    def plot_placement_in_pot_per_episode(self, data, title, ylabel):
        if not data:
            print("No placement data to plot.")
            return

        plt.figure(figsize=(10,6))
        plt.plot(data, label=title)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title("Number of Placements in Pot per Episode")
        plt.legend()
        plt.grid(True)
        plt.show()

