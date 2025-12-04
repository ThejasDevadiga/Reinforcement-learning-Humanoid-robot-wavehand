# HandWavers: Humanoid Hand-Waving Behavior Learning with Isaac Lab

This repository implements a reinforcement learning environment for teaching a humanoid robot to perform hand-waving gestures using NVIDIA Isaac Lab and PPO algorithm.

## Overview

HandWavers is an external Isaac Lab project that trains a humanoid robot to perform rhythmic hand-waving motions using Proximal Policy Optimization (PPO). The environment uses manager-based RL workflows with modular components for observations, actions, rewards, and terminations.

## Prerequisites

- **OS**: Ubuntu 20.04/22.04
- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 3080)
- **CUDA**: 11.8 or 12.x
- **Python**: 3.11 (required for Isaac Sim 5.0+)
- **Memory**: 32GB+ RAM recommended
- **Storage**: 100GB+ free space

## Installation

### Step 1: Install Isaac Sim from Source

```
# Clone Isaac Sim repository
cd ~/Documents/projects/RL
git clone https://github.com/isaac-sim/IsaacSim.git isaacsim
cd isaacsim

# Build Isaac Sim (this will take 1-2 hours)
./build.sh

# Set up environment
source setup_conda_env.sh
# Or manually set PYTHONPATH if not using conda
```

### Step 2: Install Isaac Lab

```
# Clone Isaac Lab (this repository assumes it's in same parent directory)
cd ~/Documents/projects/RL
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab

# Install Isaac Lab in development mode
cd IsaacLab
pip install -e .

# Install RL libraries
pip install -e .[skrl]  # For skrl backend
# Or: pip install -e .[all]  # For all supported RL libraries
```

### Step 3: Set Up This Project

```
# Clone this repository
cd ~/Documents/projects/RL/customTask
git clone <your-repo-url> hand_wavers
cd hand_wavers

# Install the project
pip install -e source/hand_wavers

# Verify installation
python scripts/list_envs.py
# Should show: Template-Hand-Wavers-v0
```

### Step 4: System Configuration

**Increase inotify watch limit** (required for Isaac Sim):
```
# Temporary
sudo sysctl -w fs.inotify.max_user_watches=524288

# Permanent
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Project Structure

```
hand_wavers/
├── source/hand_wavers/
│   ├── __init__.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── manager_based/
│   │       ├── __init__.py
│   │       ├── hand_wavers_env.py      # Main environment logic
│   │       └── hand_wavers_env_cfg.py  # Environment configuration
├── scripts/
│   ├── skrl/
│   │   ├── train.py         # Training script
│   │   └── zero_agent.py    # Test with zero actions
│   └── list_envs.py
├── .vscode/
│   ├── settings.json        # VSCode configuration
│   └── tools/
│       └── setup_vscode.py  # IDE setup script
└── README.md
```

## Training

### Quick Start

```
# Activate your conda environment
conda activate rltrain311

# Navigate to project directory
cd ~/Documents/projects/RL/customTask/hand_wavers

# Train with PPO (default: 4096 parallel environments)
python scripts/skrl/train.py --task=Template-Hand-Wavers-v0 --headless

# Train with custom parameters
python scripts/skrl/train.py --task=Template-Hand-Wavers-v0 \
    --num_envs=1024 \
    --max_iterations=1000 \
    --headless
```

### Training Parameters

The PPO algorithm is configured in `scripts/skrl/train.py` with these key hyperparameters:

```
# Agent configuration
cfg.ppo = {
    "rollouts": 4096,              # Steps per environment per iteration
    "learning_epochs": 5,          # PPO epochs per iteration
    "mini_batches": 4,             # Number of mini-batches
    "discount_factor": 0.99,       # Gamma
    "lambda": 0.95,                # GAE lambda
    "learning_rate": 3e-4,         # Adam learning rate
    "learning_rate_scheduler": KLAdaptiveRL,
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
    "state_preprocessor": RunningStandardScaler,
    "state_preprocessor_kwargs": {"size": env.observation_space},
    "value_preprocessor": RunningStandardScaler,
    "value_preprocessor_kwargs": {"size": 1},
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 1.0,
    "ratio_clip": 0.2,             # PPO clip parameter
    "value_clip": 0.2,
    "clip_predicted_values": True,
    "entropy_loss_scale": 0.01,    # Entropy coefficient
    "value_loss_scale": 1.0,       # Value loss coefficient
    "kl_threshold": 0.0,           # KL divergence threshold
    "rewards_shaper": lambda rewards, timestep, timesteps: rewards * 0.01,
    "time_limit_bootstrap": True,
}
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Template-Hand-Wavers-v0` | Environment name |
| `--num_envs` | 4096 | Number of parallel environments |
| `--max_iterations` | 3000 | Maximum training iterations |
| `--headless` | False | Run without GUI (faster training) |
| `--device` | `cuda:0` | Training device |
| `--seed` | None | Random seed |

### Monitoring Training

Training logs are automatically saved to `logs/skrl/<timestamp>/`:

```
# View TensorBoard logs
tensorboard --logdir logs/skrl/

## Logs include:
#### - Episode_Reward (total reward per episode)
#### - Episode_Reward/<term_name> (per reward term)
#### - Loss/value_loss
#### - Loss/policy_loss
#### - Policy/entropy
#### - Policy/kl_divergence
```

 Evaluation

```
# Run trained policy
python scripts/skrl/train.py --task=Template-Hand-Wavers-v0 \
    --checkpoint=logs/skrl/<timestamp>/checkpoint_1000.pt \
    --num_envs=1 \
    --headless False

# Test with zero actions (standing pose)
python scripts/skrl/zero_agent.py --task=Template-Hand-Wavers-v0 --num_envs=1
```

## Algorithm Details

### PPO (Proximal Policy Optimization) with skrl

We use **skrl's PPO implementation** (based on PyTorch) with these key features:

- **GAE**: Generalized Advantage Estimation for variance reduction (λ=0.95)
- **Value Clipping**: Stabilizes value function learning (clip ratio: 0.2)
- **Entropy Bonus**: Encourages exploration during learning (coefficient: 0.01)
- **Adaptive Learning Rate**: KL-adaptive scheduler adjusts LR based on policy divergence (threshold: 0.008)
- **Running Standardization**: Normalizes observations and value targets using running mean/variance
- **Mini-batch Training**: 5 epochs per iteration with 4 mini-batches for stable gradient updates
- **Gradient Clipping**: Prevents exploding gradients (norm clip: 1.0)
- **Reward Scaling**: Automatically shapes rewards during training

**Key Hyperparameters:**
- Learning rate: 3×10⁻⁴ (with KL-adaptive scheduler)
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- PPO clip parameter (ε): 0.2
- Rollout length: 4096 steps per environment
- Value loss scale: 1.0
- Entropy loss scale: 0.01
- Batch size: Scales with `num_envs` and `rollouts`

**skrl-Specific Advantages:**
- Vectorized environment support (up to 8096 parallel environments)
- Multi-agent capability (for future extensions)
- JAX/PyTorch backend flexibility
- Built-in integration with Isaac Lab managers


### Environment Design

**Manager-Based Workflow**: The environment is decomposed into modular managers:

- **Action Manager**: Converts policy outputs to joint position commands
- **Observation Manager**: Computes 87-dim state vector (base pose, joint angles, velocities, contacts)
- **Reward Manager**: Combines 7 weighted reward terms
- **Termination Manager**: Ends episodes on timeout or fall detection
- **Event Manager**: Randomizes initial states for robustness

### Reward Function

The reward is a weighted sum of:
```
\[
r_t = 1.0 \cdot \text{progress} + 2.0 \cdot \text{alive} + 0.1 \cdot \text{upright} + 0.5 \cdot \text{move\_to\_target} - 0.01 \cdot \text{action\_l2} - 0.005 \cdot \text{energy} - 0.25 \cdot \text{joint\_pos\_limits}
\]
```
Each term is scaled by `dt = 0.0167` (simulation step size).

## Troubleshooting

### Common Issues

**1. "No space left on device" error**
```
# Increase inotify limit
sudo sysctl -w fs.inotify.max_user_watches=524288
```

**2. CUDA out of memory**
```
# Reduce number of environments
python scripts/skrl/train.py --task=Template-Hand-Wavers-v0 --num_envs=512
```

### Performance Optimization

For faster training:
- Use `--headless` flag
- Increase `num_envs` (up to 8096 on RTX 4090)
- Enable `use_fabric=True` (default)
- Set `sim.render_interval = decimation` (already configured)

## Customization

To adapt this for hand-waving behavior:

1. **Modify observations**: Add hand position/velocity terms
2. **Change rewards**: Replace locomotion rewards with hand-tracking rewards
3. **Adjust actions**: Use position control for arms, zero actions for legs
4. **Update terminations**: Define waving success criteria

See `source/hand_wavers/tasks/manager_based/hand_wavers_env_cfg.py` for configuration details.

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)
- [skrl Documentation](https://skrl.readthedocs.io)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## License

This project follows the BSD-3-Clause license as specified in the Isaac Lab Project.

