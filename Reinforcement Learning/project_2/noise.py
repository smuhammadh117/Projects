import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy as sp




# class NormalNoiseDecayStrategy():
#     def __init__(self, low, high, initial_noise_ratio = 1.0, min_noise_ratio = 0.1, decay_rate = 0.9):
#         self.low = low
#         self.high = high
#         self.noise_ratio = initial_noise_ratio
#         self.min_noise_ratio = min_noise_ratio
#         self.decay_rate = decay_rate
 

#     def noise_ratio_update(self):
#         self.noise_ratio = max(self.min_noise_ratio, self.noise_ratio * self.decay_rate)
#         return self.noise_ratio
    

#     def select_action(self, model, state, max_exploration = False):
#         if max_exploration:
#             noise_scale = self.high
#         else:
#             noise_scale = self.noise_ratio * self.high

#         with torch.no_grad():
#             greedy_action = model(state).cpu().numpy().squeeze()

#         noise = np.random.normal(loc = 0.0, scale = noise_scale, size = len(self.high))

#         noisy_action = greedy_action + noise
#         action = np.clip(noisy_action, self.low, self.high)

#         self.noise_ratio_update()
#         return action
    
#     # def decay(self):
#     #         self.noise_ratio = max(self.noise_ratio * self.decay_rate, self.min_noise_ratio)

class SimpleGaussianNoiseStrategy:
    def __init__(self, action_low, action_high, noise_std=0.1):
        self.low = action_low
        self.high = action_high
        self.noise_std = noise_std

    def select_action(self, model, state):
        with torch.no_grad():
            action = model(state).cpu().numpy().squeeze()
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        noisy_action = action + noise
        clipped_action = np.clip(noisy_action, self.low, self.high)
        return clipped_action