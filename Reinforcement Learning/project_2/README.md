
# Project 2: Deep Deterministic Policy Gradient (DDPG) for LunarLanderContinuous-v3

This project applies the Deep Deterministic Policy Gradient (DDPG) algorithm to the continuous control task of LunarLanderContinuous-v3. The goal is to train an agent that learns a policy to land the lunar module smoothly by optimizing continuous thrust actions, while maximizing cumulative rewards in the environment.

---

## Author

**Syed Muhammad Husainie (shusainie3)**  
GT OMSCS - Reinforcement Learning Summer 2025  
GitHub repo: [GT GitHub Repository](https://github.gatech.edu/gt-omscs-rldm/7642RLDMSummer2025shusainie3)

---

## Project Structure

```
├── ddpg_agent.py          # Implementation of the DDPG agent connecting the actor, critic, noise, and replay buffer
├── networks.py            # The actor and critic networks implementation
├── noise.py               # Simple Gaussian noise implementation
├── replay_buffer.py       # Replay buffer implementation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Setup Instructions

### Environment

- Python 3.8 or higher

---

### Installation

```bash
pip install -r requirements.txt
```

Make sure to have the required Gymnasium environment dependencies:

```bash
pip install gymnasium
pip install 'gymnasium[box2d]'
pip install torch torchvision torchaudio
pip install numpy matplotlib
```

Note: This project uses the LunarLanderContinuous-v3 environment from gymnasium[box2d], which requires the swig system tool to build box2d-py.
If running locally, make sure to install SWIG

---

### Running the Code

Run the training script from the command line:

```bash
python lander_implementation.py
```

The script initializes the environment and agent, then trains the DDPG agent over multiple episodes. Hyperparameters and architecture can be modified inside `ddpg_agent.py`.

---

### Notes

- The agent uses Gaussian noise for exploration without any decay.
- Soft target updates ensure training stability.
- Replay buffer samples are used for batch updates.
- GPU acceleration is leveraged if available.

---

