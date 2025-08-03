from Networks import Actor, CentralizedCritic
import torch



class MAPPOAgent:



    def __init__(self, obs_dim, action_dim, hidden_dim , num_agents, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, lam=0.95, clip_range = 0.2, value_coef = 0.5, entropy_coef = 0.01, update_epochs=10, mini_batch_size = 512, total_episodes = 10000, device='cpu'):

        # Initialize the MAPPO agent with the given parameters
        self.num_agents = num_agents
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device

        # Initialize actor and centralized critic networks
        self.actor = Actor(obs_dim, action_dim,hidden_dim).to(device)
        self.critic = CentralizedCritic(obs_dim, num_agents,hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Decay setup
        self.clip_range_init = clip_range
        self.clip_range_min = 0.03
        self.entropy_coef_init = entropy_coef
        self.entropy_coef_min = 0.001
        self.total_episodes = total_episodes

        # Precompute decay rates
        self.clip_decay = (self.clip_range_min / self.clip_range_init) ** (1 / (0.75 * self.total_episodes))
        self.entropy_decay = (self.entropy_coef_min / self.entropy_coef_init) ** (1 / (0.75 * self.total_episodes))

        # Print decay rates
        print(f"Clip range decay rate: {self.clip_decay}")
        print(f"Entropy decay rate: {self.entropy_decay}")


    def act(self, obs, deterministic=False): # This function is used to get the actions for the given observations using the actor network and the centralized critic network
        """obs: Tensor of shape (num_agents, obs_dim)"""
        # obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions, log_probs, _ = self.actor.get_action(obs, deterministic=deterministic)
            joint_obs = obs.reshape(-1).unsqueeze(0)  # Reshape to (1, num_agents * obs_dim) for centralized critic
            value = self.critic(joint_obs).squeeze(-1)  # Centralized joint observations with shape (1,) # this calls the forward method of the centralized critic

        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.cpu().numpy()



    def update(self, buffer): # This function is used to update the actor and critic networks using the rollout buffer
        """Update the actor and critic networks using the rollout buffer."""
        # Compute advantages and returns

        # T is the number of time steps in the buffer
        # num_agents is the number of agents

        # buffer.ptr is the current pointer in the buffer, indicating how many transitions have been stored while T is the total number of transitions stored in the buffer.


        obs = buffer.obs[:buffer.ptr].to(self.device)  # Shape: (T, num_agents, obs_dim)

        actions = buffer.actions[:buffer.ptr]  # Shape: (T, num_agents)
        # actions = actions.long()  # Ensure actions are of type long for compatibility with the actor's action space

        old_log_probs = buffer.log_probs[:buffer.ptr]  # Shape: (T, num_agents)

        returns = buffer.returns[:buffer.ptr]  # Shape: (T, num_agents)
        returns = returns.detach()  # Detach returns to avoid gradient flow through the returns computation


        advantages = buffer.advantages[:buffer.ptr]  # Shape: (T, num_agents)
        advantages = advantages.detach()  # Detach advantages to avoid gradient flow through the advantages computation

        # Flatten the tensors for batch processing
        # This is necessary for the MAPPO update step where we need to process all agents' data together
        # Reshape to (T * num_agents, obs_dim) for observations
        # Reshape to (T * num_agents,) for actions, log_probs, returns, and advantages

        T,N = obs.shape[0], self.num_agents
        obs = obs.reshape(-1, obs.shape[-1])  # Shape: (T * num_agents, obs_dim)
        # print(f"obs shape: {obs.shape}")  # Debugging line to check the shape of observations
        actions = actions.reshape(-1) # Shape: (T * num_agents,)
        # print(f"actions shape: {actions.shape}")  # Debugging line to check the shape of actions
        old_log_probs = old_log_probs.reshape(-1)  # Shape: (T * num_agents,)
        # print(f"old_log_probs shape: {old_log_probs.shape}")
        returns = returns.reshape(-1)  # Shape: (T * num_agents,)
        # print(f"returns shape: {returns.shape}")  # Debugging line to check the shape of returns

        advantages = advantages.reshape(-1)  # Shape: (T * num_agents,)


        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = obs.shape[0]
        #print(batch_size)
        mini_batch_size = self.mini_batch_size
        #print(mini_batch_size)


        for _ in range(self.update_epochs): # Update the actor and critic networks for a number of epochs. The number of epochs is specified by the udpate_epochs parameter and is needed to ensure convergence of the policy and value functions.

          indices = torch.randperm(batch_size)
          for start in range(0, batch_size, mini_batch_size):
              end = start + mini_batch_size
              mb_idx = indices[start:end]

              mb_obs = obs[mb_idx]
              mb_actions = actions[mb_idx]
              mb_old_log_probs = old_log_probs[mb_idx]
              mb_returns = returns[mb_idx]
              mb_advantages = advantages[mb_idx]

              # Actor update
              new_log_probs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)
              ratio = torch.exp(new_log_probs - mb_old_log_probs)

              surr1 = ratio * mb_advantages
              surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
              actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

              self.actor_optimizer.zero_grad()
              actor_loss.backward()
              self.actor_optimizer.step()

              # Critic update — done once per full trajectory for centralized critic
              # You can optionally mini-batch this too if needed:
              joint_obs = buffer.obs[:buffer.ptr].reshape(T, -1).to(self.device)
              values = self.critic(joint_obs)
              returns_per_timestep = returns.view(T, N).mean(dim=1)
              critic_loss = (returns_per_timestep - values).pow(2).mean()

              self.critic_optimizer.zero_grad()
              critic_loss.backward()
              self.critic_optimizer.step()

          # Decay clip range and entropy
          self.clip_range = max(self.clip_range*self.clip_decay, self.clip_range_min)
          self.entropy_coef = max(self.entropy_coef*self.entropy_decay, self.entropy_coef_min)



    def save(self, path): # This function is used to save the actor and critic networks
        """Save the actor and critic networks to the specified path."""
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)


    def load(self, path): # This function is used to load the actor and critic networks from the specified path
        """Load the actor and critic networks from the specified path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


    def get_action_probs(self, obs_batch):
        """
        obs_batch: tensor shape [num_agents, obs_dim]
        Return: tensor shape [num_agents, action_dim] of action probabilities
        """
        # Running the policy network forward to get logits
        logits = self.actor(obs_batch)  # adjust to your actor call signature
        probs = torch.softmax(logits, dim=-1)
        return probs

    def get_value(self, joint_obs_batch):
        """
        joint_obs_batch: tensor shape [batch_size, joint_obs_dim]
        Return: tensor shape [batch_size, 1] of value estimates
        """
        values = self.critic(joint_obs_batch)
        return values.squeeze(-1)  # make sure output is [batch_size]
