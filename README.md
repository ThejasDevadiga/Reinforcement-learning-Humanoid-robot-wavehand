# 🤖✨ **HandWavers: Humanoid Hand-Waving Behavior Learning with Isaac Lab**

Teach a humanoid robot to wave *rhythmically* using reinforcement learning!
HandWavers is an external Isaac Lab project built around **PPO (Proximal Policy Optimization)** and **Isaac Lab’s manager-based RL workflow**.

---
<p align="left"> <img src="https://img.shields.io/github/stars/ThejasDevadiga/Reinforcement-learning-Humanoid-robot-wavehand?style=flat-square&logo=github" /> <img src="https://img.shields.io/badge/License-BSD--3--Clause-blue?style=flat-square" /> <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" /> <img src="https://img.shields.io/badge/Isaac%20Lab-0.5+-orange?style=flat-square&logo=nvidia" /> <img src="https://img.shields.io/badge/Framework-skrl-green?style=flat-square" /> </p>

## 🌟 **Overview**

HandWavers trains a humanoid robot to perform expressive hand-waving motions using:

* 🧠 **PPO** (skrl implementation)
* 🏗️ **Isaac Lab manager-based environment design**
* ⚙️ **Modular actions, observations, rewards, and terminations**
* ⚡ **Massively parallel vectorized simulation**

```
██╗  ██╗ █████╗ ███╗   ██╗█████╗     ██╗    ██╗ █████╗ ██╗   ██╗███████╗██████╗ 
██║  ██║██╔══██╗████╗  ██║██   ██╗   ██║    ██║██╔══██╗██║   ██║██╔════╝██╔══██╗
███████║███████║██╔██╗ ██║██   ██║   ██║ █╗ ██║███████║██║   ██║█████╗  ██████╔╝
██╔══██║██╔══██║██║╚██╗██║██   ██║   ██║███╗██║██╔══██║╚██╗ ██╔╝██╔══╝  ██╔══██╗
██║  ██║██║  ██║██║ ╚████║█████ ║    ╚███╔███╔╝██║  ██║ ╚████╔╝ ███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚════╝      ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝

           Humanoid Hand-Waving using Isaac Lab + PPO (skrl)
```

## 🧩 **Prerequisites**

| Component     | Requirement                                |
| ------------- | ------------------------------------------ |
| 🖥️ **OS**    | Ubuntu 20.04 / 22.04                       |
| 🎮 **GPU**    | NVIDIA GPU (8GB+ VRAM, tested on RTX 5070) |
| 🧪 **CUDA**   | 12.x                               |
| 🐍 **Python** | 3.11 (required for Isaac Sim 5.0+)         |
| 🧠 **RAM**    | 32GB+ recommended                          |
| 💾 **Disk**   | 100GB+ free                                |

---

## ⚙️ **Installation**

### **📦 Step 1: Install Isaac Sim (from source)**

```bash
cd ~/Documents/projects/RL
git clone https://github.com/isaac-sim/IsaacSim.git isaacsim
cd isaacsim

./build.sh       # ⏳ Takes 1–2 hours

source setup_conda_env.sh
```

---

### **🔧 Step 2: Install Isaac Lab**

```bash
cd ~/Documents/projects/RL
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab

cd IsaacLab
pip install -e .

pip install -e .[skrl]    # RL backend
# or: pip install -e .[all]
```

---

### **🤝 Step 3: Install HandWavers**

```bash
cd ~/Documents/projects/RL/customTask
git clone <your-repo-url> hand_wavers
cd hand_wavers

pip install -e source/hand_wavers
```

**Verify installation:**

```bash
python scripts/list_envs.py
# Should show: Template-Hand-Wavers-v0 🎉
```

---

### **🛠️ Step 4: System Configuration (important!)**

Increase inotify limits (required for Isaac Sim):

```bash
sudo sysctl -w fs.inotify.max_user_watches=524288
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## 📁 **Project Structure**

```
hand_wavers/
├── source/hand_wavers/
│   ├── tasks/
│   │   └── manager_based/
│   │       ├── hand_wavers_env.py      # 🌟 Main environment
│   │       └── hand_wavers_env_cfg.py  # ⚙️ Config
├── scripts/
│   ├── skrl/
│   │   ├── train.py         # 🚀 Training
│   │   └── zero_agent.py    # 🧪 Baseline (zero actions)
│   └── list_envs.py
└── README.md
```


# 🧩 Architecture Diagram
```
HandWavers System Architecture
                           +------------------------------+
                           |         PPO Agent            |
                           |   (Policy + Value networks)  |
                           +---------------+--------------+
                                           |
                                           | actions
                                           v
+------------------------------+    +-------+--------+    +------------------------+
|      Observation Manager     | <--| Action Manager |--> |   Reward Manager       |
|  (joint states, base pose,   |    | (arm joints,   |    |  (tracking, upright,   |
|   velocities, contacts, etc) |    |  wave control) |    |   energy, limits...)   |
+--------------+---------------+    +-------+--------+    +-----------+------------+
               ^                                    |                    |
               |                                    |                    v
      observations                         joint commands         reward sum
               |                                    |                    |
               +---------------------------------------------------------+
                                           Environment
                                          (Isaac Lab)
```


# 🏋️‍♂️ **Training**

### 🚀 **Quick Start**

```bash
conda activate rltrain311
cd ~/Documents/projects/RL/customTask/hand_wavers

python scripts/skrl/train.py --task=Template-Hand-Wavers-v0 --headless
```

### 🎛️ Custom Settings

```bash
python scripts/skrl/train.py \
  --task=Template-Hand-Wavers-v0 \
  --num_envs=1024 \
  --max_iterations=1000 \
  --headless
```

---

## 📊 **Training Parameters (PPO)**

Key PPO hyperparameters:

```python
"learning_rate": 3e-4,
"ratio_clip": 0.2,
"discount_factor": 0.99,
"lambda": 0.95,
"entropy_loss_scale": 0.01,
"value_loss_scale": 1.0,
"rollouts": 4096,
"mini_batches": 4,
"learning_epochs": 5,
"grad_norm_clip": 1.0,
```

✔ KL-adaptive learning rate
✔ Running state/value normalization
✔ Value clipping
✔ Reward scaling

---

## 📈 **Monitoring Training**

Start TensorBoard:

```bash
tensorboard --logdir logs/skrl/
```

Includes:

* 📈 Episode Reward
* 🔍 Per-term reward breakdown
* 📉 Value & policy loss
* ♻️ KL divergence
* 🔥 Entropy

---

## 🎬 **Evaluation & Testing**

```bash
# Run trained policy
python scripts/skrl/train.py \
  --task=Template-Hand-Wavers-v0 \
  --checkpoint=logs/skrl/<timestamp>/checkpoint_1000.pt \
  --num_envs=1
```

Test standing (zero actions):

```bash
python scripts/skrl/zero_agent.py --task=Template-Hand-Wavers-v0 --num_envs=1
```

---

# 🧠 **Algorithm Details (PPO)**

HandWavers uses skrl’s PPO implementation with:

* 🌊 **GAE (λ = 0.95)**
* ✂️ **Clipped surrogate objective**
* 🧊 **Value function clipping**
* ♻️ **KL-adaptive LR**
* 🧮 **RunningStandardScaler**
* 🧱 **Mini-batch updates**
* 🛡️ **Gradient clipping**

---

## 🏗️ **Environment Design**

Modular Isaac Lab managers:

* 🎮 **Action Manager** – joint commands
* 👁️ **Observation Manager** – ~87-dim state
* 🎯 **Reward Manager** – shaping & tracking
* 🛑 **Termination Manager** – timeout / fall
* 🎲 **Event Manager** – randomization

---

## 🏆 **Reward Function**

Main reward components:

```
r = 1.0 * progress
  + 2.0 * alive
  + 0.1 * upright
  + 0.5 * move_to_target
  - 0.01 * action_l2
  - 0.005 * energy
  - 0.25 * joint_pos_limits
```

⏱️ Scaled by dt = 0.0167

---

# 🛠️ Troubleshooting

### ❌ Out of GPU Memory?

```bash
python scripts/skrl/train.py --num_envs=512
```

### ❌ "No space left on device"

Increase inotify (see above).

### ⚡ Performance Tips

* Use `--headless`
* Increase `num_envs` (if VRAM allows)
* Enable Fabric (default)
* Reduce rendering frequency

---

# 🧩 Customization

To adapt for **hand-waving**:

* ➕ Add hand pose/velocity observations
* 🎯 Replace locomotion rewards with gesture targets
* 🦾 Modify action space to focus on arm joints
* 🛑 Add custom termination / success criteria

Edit:
`source/hand_wavers/tasks/manager_based/hand_wavers_env_cfg.py`

---

# 📚 References

* 📘 Isaac Lab Docs
* 📙 skrl Documentation
* 📄 PPO Paper (Schulman et al. 2017)

---

# 📜 License

BSD-3-Clause (inherits Isaac Lab’s license)

---
