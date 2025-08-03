# creating the RolloutBuffer class for MAPPO


import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, num_agents, obs_dim, device ='cpu'):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device
        self.num_agents = num_agents

        # Initialize the buffer
        self.obs = torch.zeros((buffer_size, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, num_agents), dtype=torch.int64, device=device)
        self.log_probs = torch.zeros((buffer_size, num_agents), device=device)
        self.rewards = torch.zeros((buffer_size, num_agents), device=device)
        self.dones = torch.zeros((buffer_size, num_agents), device=device)
        self.values = torch.zeros((buffer_size, ), device=device)

        self.advantages = torch.zeros((buffer_size, num_agents), device=device)
        self.returns = torch.zeros((buffer_size, num_agents ), device=device)

        self.ptr = 0  # Pointer to the current position in the buffer
        self.max_size = buffer_size  # Maximum size of the


        self.storecount = 0

    def store(self, obs, action, log_prob, reward, done, value):
        """Store a single transition in the buffer."""
        self.storecount += 1
        # print(f"self.store: {self.storecount}")


        # print(self.ptr)
        if self.ptr >= self.max_size:
            self.ptr = 0

        # Convert to tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.int64, device=self.device)
        if isinstance(log_prob, np.ndarray):
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
        if isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if isinstance(done, np.ndarray):
            done = torch.tensor(done, dtype=torch.float32, device=self.device)
        if isinstance(value, np.ndarray):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)

        # If input is missing batch/agent dimension, expand to match buffer shape
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # [obs_dim] -> [1, obs_dim]
        if isinstance(value, torch.Tensor) and value.numel() > 1:
            value = value.mean()
        if isinstance(value, torch.Tensor):
            value = value.item()

        # print(self.obs[self.ptr].shape)
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value

        self.ptr += 1

        # print(f"OBSERVATION TENSOR SHAPE: {self.obs.shape}")


    def compute_advantages(self, next_value, gamma=0.99, lam=0.95):
            """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
            if self.ptr == 0:
                self.ptr = 0
                # raise ValueError("Buffer is empty. Cannot compute advantages.")

            next_value = next_value.squeeze() if next_value.ndim > 0 else next_value

            # Initialize the last advantage and return
            gae = 0 # For the last time step the advantage is 0. Recursively computed for previous time steps

            # print(f"next_value shape before squeeze: {next_value.shape}")


            # next_value_for_calc = next_value.squeeze(-1)

            # print(f"next_value_for_calc shape after squeeze: {next_value_for_calc.shape}")
            # print(f"Expected shape: ({self.num_agents},)")
            # # Ensure shape is (num_agents,)
            # assert next_value_for_calc.shape == (self.num_agents,), "next_value shape mismatch"

            next_value_for_calc = next_value
            next_return_target = next_value_for_calc

            # print("DEBUGGER 2")
            # Compute advantages in reverse order
            for t in reversed(range(self.ptr)):
                if t == self.ptr - 1:
                    # For the last time step, we use the next value directly
                    value_t1 = next_value_for_calc *(1-self.dones[t].any().float())

                else:
                    value_t1 = self.values[t + 1] * (1 - self.dones[t].any().float())

                avg_reward = self.rewards[t].mean()
                # delta = avg_reward + gamma * value_t1 - self.values[t] # Temporal difference error
                # gae = delta + gamma * lam * (1 - self.dones[t].any().float()) * gae # Generalized Advantage Estimation
                # self.advantages[t] = gae
                next_return_target = avg_reward + gamma * next_return_target * (1-self.dones[t].any().float()) # Return target for the current time step
                self.returns[t] = next_return_target


            for agent_id in range(self.num_agents):
                gae = 0
                next_return_target = next_value
                for t in reversed(range(self.ptr)):
                    done = self.dones[t, agent_id].float()
                    reward = self.rewards[t, agent_id]
                    if t == self.ptr - 1:
                        value_t1 = next_value * (1 - done)
                    else:
                        value_t1 = self.values[t + 1] * (1 - done)

                    delta = reward + gamma * value_t1 - self.values[t]
                    gae = delta + gamma * lam * (1 - done) * gae
                    self.advantages[t, agent_id] = gae
                    # self.returns[t, agent_id] = reward + gamma * next_return_target * (1 - done)



            return self.advantages[:self.ptr], self.returns[:self.ptr] # Return advantages and returns for the stored transitions


    # # Separating advantage per agent

    # def compute_advantages(self, next_value, gamma=0.99, lam=0.95):
    #         if self.ptr == 0:
    #             raise ValueError("Buffer is empty. Cannot compute advantages.")

    #         next_value = next_value.item() if isinstance(next_value, torch.Tensor) else next_value

    #         for agent_id in range(self.num_agents):
    #             gae = 0
    #             next_return_target = next_value
    #             for t in reversed(range(self.ptr)):
    #                 done = self.dones[t, agent_id].float()
    #                 reward = self.rewards[t, agent_id]
    #                 if t == self.ptr - 1:
    #                     value_t1 = next_value * (1 - done)
    #                 else:
    #                     value_t1 = self.values[t + 1] * (1 - done)

    #                 delta = reward + gamma * value_t1 - self.values[t]
    #                 gae = delta + gamma * lam * (1 - done) * gae
    #                 self.advantages[t, agent_id] = gae
    #                 self.returns[t, agent_id] = reward + gamma * next_return_target * (1 - done)


    #         return self.advantages[:self.ptr], self.returns[:self.ptr] # Return advantages and returns for the stored transitions


    def get_minibatches(self, batch_size, shuffle = True):
      """Get minibatches of the stored transitions."""

      total_samples = self.ptr # We only want to sample to the ptr size
      print(f"Total samples: {total_samples}")
      print(f"OBSERVATION TENSOR SHAPE in minibatch: {self.obs.shape}")

      if shuffle:
        indices = torch.randperm(total_samples)
      else:
        indices = torch.arange(total_samples)

      for i in range(0, total_samples, batch_size):
        end = min(i + batch_size, total_samples)
        batch_index = indices[i:end]
        yield (
            self.obs[batch_index],
            self.actions[batch_index],
            self.log_probs[batch_index],
            self.rewards[batch_index],
            self.dones[batch_index],
            self.values[batch_index],
            self.advantages[batch_index],
            self.returns[batch_index]
        )




    def reset(self):
            """Reset the buffer pointer to start storing from the beginning."""
            self.ptr = 0