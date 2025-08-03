import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math


# Actor network

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=[32, 32,32 ,32]):
        super(Actor, self).__init__()
        layers = []
        input_dim = obs_dim

        for h_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, action_dim))  # Final layer for logits

        self.model = nn.Sequential(*layers)
        print(hidden_dim)

    def forward(self, x):
        return self.model(x)

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, logits.detach()


    def compute_KL(self, obs, old_logits):
        new_logits = self.forward(obs)
        new_log_probs = Categorical(logits=new_logits)

        old_probs = Categorical(logits=old_logits)

        kl = torch.distributions.kl_divergence(old_probs, new_log_probs).mean()
        return kl



    def evaluate_actions(self, obs, actions):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


# Centralized Critic network
class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim_per_agent, n_agents, hidden_dim=[32, 32, 32, 32]):
        super(CentralizedCritic, self).__init__()
        input_dim = obs_dim_per_agent * n_agents  # Total observation dimension for all agents

        # print(f"DEBUG: CentralizedCritic INIT - Instance ID: {id(self)}")
        # print(f"DEBUG: CentralizedCritic INIT - obs_dim_per_agent={obs_dim_per_agent}, n_agents={n_agents}, calculated input_dim={input_dim}")

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dim:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, 1))  # Output: scalar value

        self.model = nn.Sequential(*layers)

        # print("DEBUG: CentralizedCritic INIT - All parameter shapes:")
        # for name, param in self.named_parameters():
        #     print(f"  {name}: {param.shape}")

    def forward(self, joint_obs):
        # print(f"DEBUG: Critic FORWARD CALLED - Instance ID: {id(self)}")
        # print(f"DEBUG: Critic forward input shape: {joint_obs.shape}")
        # print(f"DEBUG: Critic fc1.weight shape during forward: {self.fc1.weight.shape}")

        # # Double-check before feeding to fc1
        # if joint_obs.shape[-1] != self.fc1.in_features:
            # print(f"ERROR: Mismatched input size! Expected {self.fc1.in_features}, but got {joint_obs.shape[-1]}")

        return self.model(joint_obs).squeeze(-1)  # Return scalar values

        # print(f"DEBUG: Critic output value shape: {value.shape}")


