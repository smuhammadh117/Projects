
import os
from PIL import Image
from IPython.display import display, Image as IPImage
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import gym
import numpy as np
import torch
from PIL import Image
import os
from IPython.display import display, Image as IPImage
from MAPPOAgent import MAPPOAgent
from MAPPOTrainer import MAPPOTrainer
from RolloutBuffer import RolloutBuffer

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Setup environment, agent, buffer


    ### Environment setup ###

    ## Swap between the 3 layouts here:
    # layout = "cramped_room"
    # layout = "coordination_ring"
    layout = "counter_circuit_o_1order"

    ## Reward shaping is disabled by default; i.e., only the sparse rewards are
    ## included in the reward returned by the enviornment).  If you'd like to do
    ## reward shaping (recommended to make the task much easier to solve), this
    ## data structure provides access to a built-in reward-shaping mechanism within
    ## the Overcooked environment.  You can, of course, do your own reward shaping
    ## in lieu of, or in addition to, using this structure. The shaped rewards
    ## provided by this structure will appear in a different place (see below)
    reward_shaping = {
    "PLACEMENT_IN_POT_REW": 8,
    "DISH_PICKUP_REWARD": 5,
    "SOUP_PICKUP_REWARD": 5,
    # "DISH_DISP_DISTANCE_REW": 0.1,
    # "POT_DISTANCE_REW": 0.1,
    # "SOUP_DISTANCE_REW": 0.1,
    }

    # Length of Episodes.  Do not modify for your submission!
    # Modification will result in a grading penalty!
    horizon = 400

    # Build the environment.  Do not modify!
    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    # FIX: Access num_agents from the mdp object
    num_agents = mdp.num_players
    print(f"Number of agents: {num_agents}")

    obs_dim = env.observation_space.shape[0]
    print(f"Observation space: {obs_dim}")

    action_dim = env.action_space.n # action_dim for a single agent
    print(f"Action space: {action_dim}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the MAPPO agent
    hidden_dim = (32, 32, 32, 32)
    mini_batch_size = 700
    episodes = 8000

    agent = MAPPOAgent(obs_dim, action_dim,hidden_dim, num_agents,mini_batch_size=mini_batch_size, clip_range = 0.13, entropy_coef=0.1, total_episodes = episodes, update_epochs = 10, device=device)


    # Initialize the rollout buffer
    buffer = RolloutBuffer(buffer_size=8000, num_agents=num_agents, obs_dim=obs_dim, device=device)



    # Create trainer and start training
    trainer = MAPPOTrainer(env=env, agent=agent, buffer=buffer, num_agents=num_agents,
                        obs_dim=obs_dim, action_dim=action_dim, device=device,
                        rollout_length=2048, log_interval=10, max_episodes=episodes, min_samples_per_update = 5)



    trainer.run()


    # Plot Rolling Average Episode Rewards
    trainer.plot_rolling_average(trainer.episode_rewards, 'Episode Reward Over Time', 'Average Reward per Agent', window_size=50)

    # Plot Rolling Average Soups Made
    trainer.plot_rolling_average(trainer.total_soups, 'Soups Made Over Time', 'Number of Soups Made', window_size=50)


    # Plotting info metrics
    trainer.plot_placement_in_pot_per_episode(trainer.placement_in_pot_log, 'Placement in Pot per Episode', 'Onion Count')
    trainer.plot_placement_in_pot_per_episode(trainer.onion_pickup_log, 'Number of Onions Picked up per Episode', 'Onion Count')
    trainer.plot_placement_in_pot_per_episode(trainer.dish_pickup_log, 'Number of Dishes Picked up per Episode', 'Dish Count')
    trainer.plot_placement_in_pot_per_episode(trainer.soup_delivery_log, 'Soup Deliveries per Episode (correct and incorrect)', 'Deliveries Count')


    # Load the saved policy
    # agent.load("policy_layout1.pt")




    # Evaluate

    trainer.evaluate(num_episodes=100)


                                
