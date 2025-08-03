
# Project 3: Multi-Agent Proximal Policy Optimization (MAPPO) for Overcooked-AI

This project implements the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm to train cooperative agents in the Overcooked-AI environment. The agents learn coordinated policies to complete complex cooking tasks by optimizing joint actions, leveraging centralized critics and reward shaping to improve teamwork and sample efficiency.

---

## Author

**Syed Muhammad Husainie (shusainie3)**  
GT OMSCS - Reinforcement Learning Summer 2025  
GitHub repo: [GT GitHub Repository](https://github.gatech.edu/gt-omscs-rldm/7642RLDMSummer2025shusainie3)

---

## Project Structure

```
├── mappo_agent.py # Implementation of MAPPO agent with actor-critic networks
├── rollout_buffer.py # Mini-batch rollout buffer for storing experiences
├── overcooked_env.py # Wrapper for the Overcooked-AI environment
├── trainer.py # MAPPOTrainer class managing training loop and updates
├── implementation.py # Main script to set up and run training and evaluation
├── requirements.txt # Python dependencies
└── README.md # This file
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
pip install torch==2.2.0 torchvision torchaudio
pip install numpy==1.26.4
pip install gymnasium
pip install 'gymnasium[box2d]'
pip install matplotlib==3.8.4 Pillow==10.3.0 ipython==8.24.0 ipykernel==6.29.4
pip install git+https://github.com/HumanCompatibleAI/overcooked_ai.git@0a6ab67
```
Optional (for Google Colab / Google Drive support):

```bash
pip install google-colab
pip install gdown
```
---

### Running the Code

Run the training script from the command line:

```bash
python implementation.py
```

This script initializes the environment and agent, then trains the MAPPO agent over multiple episodes.
To run saved policies, edit implementation.py by commenting out trainer.run() and uncommenting the desired layout policy evaluation section.


---

### Notes

- Implements Multi-Agent PPO with centralized critics for effective coordination.
- Uses detailed reward shaping with event-based signals for better credit assignment.
- Mini-batch rollout buffer with random sampling improves training stability by breaking temporal correlations.
- Applies entropy regularization and clipped surrogate loss to balance exploration and stable policy updates.
- Supports GPU acceleration when available.
- Policies can be loaded for evaluation by adjusting the main training script.
- Requires SWIG installed locally to build Box2D dependencies for the Overcooked environment.


---

