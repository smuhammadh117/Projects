import numpy as np
import torch

# class ReplayBuffer:
#     def __init__(self, max_size = 50000, batch_size = 64):
#         self.ss_mem = np.empty(shape=(max_size), dtype=object)
#         self.as_mem = np.empty(shape=(max_size), dtype=object)
#         self.rs_mem = np.empty(shape=(max_size), dtype=object)
#         self.ps_mem = np.empty(shape=(max_size), dtype=object)
#         self.ds_mem = np.empty(shape=(max_size), dtype=object)

#         self.max_size = max_size
#         self.batch_size = batch_size
#         self._idx = 0
#         self.size = 0


#     def store(self, sample):
#         s, a, r, p, d = sample
#         self.ss_mem[self._idx] = s
#         self.as_mem[self._idx] = a
#         self.rs_mem[self._idx] = r
#         self.ps_mem[self._idx] = p
#         self.ds_mem[self._idx] = d

#         self._idx += 1
#         self._idx = self._idx % self.max_size

#         self.size += 1
#         self.size = min(self.size, self.max_size)


#     def sample(self, batch_size = None):
        

#         if batch_size == None:
#             batch_size = self.batch_size

#         ind = np.random.choice(self.size, batch_size)
#         experiences = np.vstack(self.ss_mem[ind]),\
#                       np.vstack(self.as_mem[ind]),\
#                       np.vstack(self.rs_mem[ind]),\
#                       np.vstack(self.ps_mem[ind]),\
#                       np.vstack(self.ds_mem[ind])
        
#         return experiences
    
#     def __len__(self):
#         return self.size


# Using this implementation for the replay buffer worked better with the Lander environment, even though both implementations are very similar, this one uses torch tensors directly instead of NumPy arrays

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0


        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32)  # float for done mask (0.0 or 1.0)

    def add(self, state, action, reward, next_state, done):
        # Convert inputs from NumPy to torch and store
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.float32)
        self.rewards[self.ptr] = torch.tensor([reward], dtype=torch.float32)
        self.next_states[self.ptr] = torch.tensor(next_state, dtype=torch.float32)
        self.dones[self.ptr] = torch.tensor([done], dtype=torch.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return self.size

    def sample(self, batch_size=64):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        return self.size

    