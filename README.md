# 🐾 EC518-Bark9: Reinforcement Learning for a Custom 2-DOF Robot in MuJoCo

This project implements and compares **Soft Actor-Critic (SAC)** and **Deep Q-Learning (DQN)** algorithms on a **custom 2-degree-of-freedom robot model** simulated in **MuJoCo**.  
Developed as part of **EC518: Robotics and Embedded Systems**, the repository explores how different reinforcement learning algorithms perform on continuous vs. discrete control tasks in low-dimensional robotic systems.

---

## 🚀 Overview

The **Bark9** platform is a lightweight simulation environment featuring a minimalist robot model with two controllable joints.  
The goal is to train control policies for dynamic movement (balancing, reaching, or locomotion) while comparing algorithmic stability, convergence, and efficiency.

### Key Features
- 🧠 **Two Reinforcement Learning Agents**
  - **Soft Actor-Critic (SAC):** Continuous control via entropy-regularized policy optimization.  
  - **Deep Q-Learning (DQN):** Discrete control baseline using experience replay and target networks.  
- ⚙️ **Custom 2-DOF MuJoCo Model**
  - Fully defined XML structure (geometry, joints, actuators, and sensors).  
  - Designed for easy parameterization and visualization.  
- 📊 **Performance Analysis Tools**
  - Reward tracking, policy visualization, and training curve logging.  
  - Configurable hyperparameters and replay buffer parameters.  

---

## 🧩 Project Structure
```bash
EC518-Bark9/
│
├── mujoco_models/           # Custom 2-DOF robot XML files
├── agents/
│   ├── sac_agent.py         # Soft Actor-Critic implementation
│   ├── dqn_agent.py         # Deep Q-Learning implementation
│
├── envs/
│   ├── bark9_env.py         # Custom MuJoCo environment wrapper
│
├── training/
│   ├── train_sac.py         # SAC training loop
│   ├── train_dqn.py         # DQN training loop
│
├── utils/
│   ├── replay_buffer.py     # Experience replay buffer implementation
│   ├── plot_utils.py        # Visualization helpers
│
├── results/                 # Saved models, logs, and reward plots
└── README.md
```


---

## 🧠 Algorithms

### **Soft Actor-Critic (SAC)**
- Uses stochastic policy gradients for continuous control.  
- Encourages exploration via entropy regularization.  
- Achieves smoother convergence and better stability in continuous action spaces.

### **Deep Q-Learning (DQN)**
- Discrete-action agent trained with target networks and experience replay.  
- Provides a strong baseline for evaluating SAC’s advantages in continuous control.

---

## ⚒️ Installation & Setup

### Prerequisites
- Python 3.8+
- MuJoCo (>= 2.3)
- Gymnasium or OpenAI Gym (depending on environment wrapper)
- PyTorch
- NumPy, Matplotlib

### Installation
```bash
git clone https://github.com/abingeorge07/EC518-Bark9.git
cd EC518-Bark9
pip install -r requirements.txt

