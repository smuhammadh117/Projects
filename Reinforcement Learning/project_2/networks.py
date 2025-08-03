import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy as sp
import gymnasium as gym
import pygame
import matplotlib.pyplot as plt
import seaborn as sns  
import Box2D



# DDPG Network for Actor-Critic Method


# Critic Network

class FCQV(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = (300,300,200,100,32,32), activation_fc = F.relu):

        # input_dim: dimensions of state s
        # output_dim: dimensions of action a
        # hidden_dims: 2 layers of size 32 each as default


        super(FCQV,self).__init__()

        self.activation_fc = activation_fc # this is to save the activation function to be used in the forward pass

        self.input_layer = nn.Linear(input_dim, hidden_dims[0]) # from input to the first hidden layer
        self.hidden_layers = nn.ModuleList() # list of hidden layers for easy access

        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i] # size of the previous hidden layer

            if i == 0:
                in_dim += output_dim # concatenate the action to the input of the first hidden layer

            hidden_layer = nn.Linear(in_dim, hidden_dims[i+1]) # create a hidden layer with the specified dimensions
            self.hidden_layers.append(hidden_layer) # add the hidden layer to the list helps to keep track of the layers

        self.output_layer = nn.Linear(hidden_dims[-1],1) # output layer to get the Q-value



    def forward(self, state,action):
        x,u = self._format(state,action) # format the input to be tensors and add batch dimension if needed
        x = self.activation_fc(self.input_layer(x)) # pass the input through the first layer

        for i, hidden_layer in enumerate(self.hidden_layers):

            if i == 0:
                x = torch.cat((x,u), dim = 1)

            x = self.activation_fc(hidden_layer(x))

        return self.output_layer(x)  # output the Q-value



    def _format(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        return state, action



# Actor Network

class FCDP(nn.Module):
    def __init__(self, input_dim, action_bounds, hidden_dims = (300,300,200,100,32,32), activation_fc = F.relu, out_activation_fc = F.tanh):

        super(FCDP,self).__init__()



        self.activation_fc = activation_fc # this is to save the activation function to be used in the forward pass
        self.out_activation_fc = out_activation_fc # this is to save the output activation function to be used in the forward pass
        self.env_min = torch.tensor(action_bounds[0], dtype=torch.float32) # Convert to tensor
        self.env_max = torch.tensor(action_bounds[1], dtype=torch.float32) # Convert to tensor


        self.input_layers = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)


        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max))



    def forward(self, state):
        x = self._format(state)  # format the input to be tensors and add batch dimension if needed
        x = self.activation_fc(self.input_layers(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        x = self.out_activation_fc(self.output_layer(x))
        action = x * (self.env_max - self.env_min) + self.env_min # Scale to the action bounds
        return action



    def _format(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension

        return state

