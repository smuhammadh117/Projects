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


# class DDPGAgent():
#     def __init__(self, state_dim, action_bounds, hidden_dims=(32,32), activation_fc=F.relu, out_activation_fc=torch.tanh,
#                  gamma=0.99, tau=0.001, value_lr=1e-4, policy_lr=1e-4,
#                  value_max_grad_norm=0.4, policy_max_grad_norm=0.5):
        

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.gamma = gamma
#         self.tau = tau
#         self.value_max_grad_norm = value_max_grad_norm
#         self.policy_max_grad_norm = policy_max_grad_norm
        
#         # Initialize actor and critic networks 
#         self.online_value_model = FCQV(state_dim, len(action_bounds[0]), hidden_dims, activation_fc).to(self.device)
#         self.target_value_model = FCQV(state_dim, len(action_bounds[0]), hidden_dims, activation_fc).to(self.device)
        
#         self.online_policy_model = FCDP(state_dim, action_bounds, hidden_dims, activation_fc, out_activation_fc).to(self.device)
#         self.target_policy_model = FCDP(state_dim, action_bounds, hidden_dims, activation_fc, out_activation_fc).to(self.device)
        
#         # Initialize target networks with the same weights as the online networks
#         self.target_value_model.load_state_dict(self.online_value_model.state_dict())
#         self.target_policy_model.load_state_dict(self.online_policy_model.state_dict())
        
#         # Optimizers
#         self.value_optimizer = optim.Adam(self.online_value_model.parameters(), lr=value_lr)
#         self.policy_optimizer = optim.Adam(self.online_policy_model.parameters(), lr=policy_lr)


#     def soft_update(self, target_net, source_net):
#         for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
#             target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data) # Slow updating following strategies for exponential smoothing and in techniques used in similar implementations such as https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py?utm_source=chatgpt.com


#     def optimize_model(self, experiences):  # experiences is a tuple of (states, actions, rewards, next_states, dones)
#         states, actions, rewards, next_states, dones = experiences

#         # Compute target Q-values using the target networks
#         with torch.no_grad():
#             argmax_a_q_sp = self.target_policy_model(next_states) # Get the action that maximizes Q for the next state
#             max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp) # Compute Q-value for the next state using the target value model
#             # Fixed target Q formula:
#             target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - dones) # Bellman equation for expected Q-value

#         # Compute current Q-values using the online value model from the replay buffer
#         q_sa = self.online_value_model(states, actions) 

#         # Compute critic loss (mean squared TD error)
#         td_error = target_q_sa.detach() - q_sa
#         value_loss = td_error.pow(2).mean()

#         self.value_optimizer.zero_grad()
#         value_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.value_max_grad_norm)
#         self.value_optimizer.step()

#         # Compute actor loss (maximize Q-value)
#         argmax_a_q_s = self.online_policy_model(states)
#         max_a_q_s = self.online_value_model(states, argmax_a_q_s)
#         policy_loss = -max_a_q_s.mean() # since minimizing this will maximize the q value

#         self.policy_optimizer.zero_grad()
#         policy_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), self.policy_max_grad_norm)
#         self.policy_optimizer.step()


#         self.soft_update(self.target_value_model, self.online_value_model)
#         self.soft_update(self.target_policy_model, self.online_policy_model)




class DDPGAgent():
    def __init__(self, state_dim, action_bounds, hidden_dims=(300,300,200,100,32,32), activation_fc=F.relu, out_activation_fc=torch.tanh,
                 gamma=0.99, tau=0.005, value_lr=0.001, policy_lr=0.001,
                 value_max_grad_norm=1, policy_max_grad_norm=1):

        self.gamma = gamma
        self.tau = tau
        self.value_max_grad_norm = value_max_grad_norm
        self.policy_max_grad_norm = policy_max_grad_norm

        # Initialize actor and critic networks
        self.online_value_model = FCQV(state_dim, len(action_bounds[0]), hidden_dims, activation_fc)
        self.target_value_model = FCQV(state_dim, len(action_bounds[0]), hidden_dims, activation_fc)

        self.online_policy_model = FCDP(state_dim, action_bounds, hidden_dims, activation_fc, out_activation_fc)
        self.target_policy_model = FCDP(state_dim, action_bounds, hidden_dims, activation_fc, out_activation_fc)

        # Initialize target networks with the same weights as the online networks
        self.target_value_model.load_state_dict(self.online_value_model.state_dict())
        self.target_policy_model.load_state_dict(self.online_policy_model.state_dict())

        # Optimizers
        self.value_optimizer = optim.Adam(self.online_value_model.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.online_policy_model.parameters(), lr=policy_lr)


    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data) # Slow updating following strategies for exponential smoothing and in techniques used in similar implementations such as https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py?utm_source=chatgpt.com


    def optimize_model(self, experiences):  # experiences is a tuple of (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = experiences

        # Compute target Q-values using the target networks
        with torch.no_grad():
            argmax_a_q_sp = self.target_policy_model(next_states) # Get the action that maximizes Q for the next state
            max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp) # Compute Q-value for the next state using the target value model
            # Fixed target Q formula:
            target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - dones) # Bellman equation for expected Q-value

        # Compute current Q-values using the online value model from the replay buffer
        q_sa = self.online_value_model(states, actions)

        # Compute critic loss (mean squared TD error)
        td_error = target_q_sa.detach() - q_sa
        value_loss = td_error.pow(2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        # Compute actor loss (maximize Q-value)
        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean() # since minimizing this will maximize the q value

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()


        self.soft_update(self.target_value_model, self.online_value_model)
        self.soft_update(self.target_policy_model, self.online_policy_model)

        return value_loss.detach()
