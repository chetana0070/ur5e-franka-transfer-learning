# UR5e-Franka Cross Embodiment Learning

Sim-to-sim cross-embodiment robot imitation learning using UR5e and Franka/Panda robots with DTW alignment, MixUp interpolation, and diffusion/Transformer-based policy learning.

---

# Overview

This project studies how one robot can learn manipulation behavior from another robot with different kinematics and control dynamics.

The system transfers manipulation skills from:

```text
UR5e  →  Franka Panda
```

inside simulation.

The pipeline is inspired by the ImMimic framework and adapts cross-domain imitation learning into a fully reproducible robot-to-robot simulation environment. :contentReference[oaicite:0]{index=0}

The project focuses on:

- Cross-embodiment imitation learning
- Trajectory alignment
- Demonstration interpolation
- Shared robot representations
- Policy learning from mixed demonstrations
- Sim-to-sim transfer learning

---

# Core Problem

Robots have:

- Different joint structures
- Different kinematics
- Different action spaces
- Different gripper mechanics
- Different controller dynamics

A trajectory that works for one robot cannot directly work for another.

This project solves that problem using:

- Shared state representations
- Dynamic Time Warping (DTW)
- MixUp interpolation
- Sequence modeling policies

---

# Core Concepts

---

## 1. Cross-Embodiment Imitation Learning

Cross-embodiment learning means:

```text
Learning a task from a different robot embodiment
```

Example:

```text
UR5e demonstrations
        ↓
Franka learns task behavior
```

The challenge is that the robots are physically different.

---

## 2. Shared Representation

Both UR5e and Franka trajectories are converted into the same observation space.

Shared 14D observation vector:

| Feature | Dimensions |
|---|---|
| End-effector position | 3 |
| End-effector quaternion | 4 |
| Gripper scalar | 1 |
| Object position | 3 |
| Goal XY | 2 |
| Task phase scalar | 1 |

Total:

```text
14 Dimensions
```

This allows trajectories from different robots to become comparable. :contentReference[oaicite:1]{index=1}

---

## 3. Dynamic Time Warping (DTW)

DTW aligns trajectories of different speeds and lengths.

Purpose:

- Match semantically similar robot actions
- Create correspondence between UR5e and Franka timesteps

Example:

```text
UR5e timestep 20 → Franka timestep 17
```

Without DTW:

- Demonstrations remain unaligned
- Cross-training becomes noisy

DTW creates meaningful temporal alignment between robots. :contentReference[oaicite:2]{index=2}

---

## 4. MixUp Interpolation

After DTW alignment, synthetic demonstrations are generated using interpolation.

Formula:

:contentReference[oaicite:3]{index=3}

Used alpha values:

- 0.25
- 0.50
- 0.75

Purpose:

- Bridge embodiment gap
- Smooth transition between source and target domains
- Improve policy generalization

This creates intermediate robot behaviors between UR5e and Franka. :contentReference[oaicite:4]{index=4}

---

## 5. Diffusion / Transformer Policy Learning

The policy learns:

```text
Observation History → Franka Action
```

Input:

```text
16 × 14 observation history
```

Output:

```text
7D Franka action
```

Action space:

- 3D end-effector displacement
- 3D end-effector rotation
- 1 gripper scalar

The sequence model learns temporal manipulation behavior such as:

- Grasping
- Lifting
- Transport
- Placement

:contentReference[oaicite:5]{index=5}

---

# Complete Pipeline

```text
UR5e Demonstrations
        ↓
Franka Demonstrations
        ↓
Shared Representation Conversion
        ↓
DTW Alignment
        ↓
Best Alignment Path Selection
        ↓
Aligned Trajectory Pair Construction
        ↓
MixUp Interpolation
        ↓
Merged Shared Dataset
        ↓
Shared Policy Training
        ↓
Franka Evaluation
```

---

# Repository Structure

```bash
.
├── data/
│   ├── raw/
│   ├── shared/
│   ├── aligned/
│   ├── interpolated/
│   └── merged/
│
├── src/
│   ├── preprocessing/
│   ├── alignment/
│   ├── interpolation/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── checkpoints/
├── logs/
├── results/
├── videos/
├── reports/
└── README.md
```

---

# Environment Setup

---

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ur5e-franka-cross-embodiment-learning.git

cd ur5e-franka-cross-embodiment-learning
```

---

## Create Conda Environment

```bash
conda create -n sim2sim python=3.10 -y

conda activate sim2sim
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Main Dependencies

```text
torch
numpy
mujoco
robosuite
gymnasium
scipy
matplotlib
h5py
transformers
diffusers
```

---

# Reproducing the Project

Run everything in this exact order.

---

## 1. Convert Demonstrations into Shared Representation

```bash
python src/preprocessing/convert_pick_place_to_shared.py
```

Output:

```text
data/shared/
```

---

## 2. Compute DTW Alignments

```bash
python src/alignment/dtw_all_pairs.py
```

Purpose:

- Compute alignment cost between UR5e and Franka trajectories

---

## 3. Save Best DTW Paths

```bash
python src/alignment/save_best_dtw_paths.py
```

Purpose:

- Store best temporal correspondence mappings

---

## 4. Build Aligned Trajectory Pairs

```bash
python src/alignment/build_aligned_pairs.py
```

Output:

```text
data/aligned/
```

---

## 5. Generate MixUp Interpolated Demonstrations

```bash
python src/interpolation/interpolate_ur5e_franka_mixup.py
```

Output:

```text
data/interpolated/
```

---

## 6. Merge Shared Training Dataset

```bash
python src/dataset/merge_shared_training_data.py
```

Output:

```text
data/merged/
```

---

## 7. Train Shared Diffusion Policy

```bash
bash src/training/train_diffusion_shared.sh
```

Outputs:

```text
checkpoints/
logs/
```

---

## 8. Evaluate on Franka

```bash
python src/evaluation/eval_diffusion_shared_on_franka.py
```

Outputs:

```text
results/
videos/
```

---

# Results

The final system was evaluated inside robosuite using Franka Panda. :contentReference[oaicite:6]{index=6}

---

## Quantitative Results

| Metric | Value |
|---|---|
| Successful Rollouts | 19 / 20 |
| Success Rate | 95% |
| Evaluation Horizon | 900 timesteps |
| Control Frequency | 20 Hz |

:contentReference[oaicite:7]{index=7}

---

## Demonstration Collection Efficiency

| Method | Average Time |
|---|---|
| Human Demonstration | 2.66 min |
| UR5e Sim Demonstration | 0.90 min |
| Real Franka Demonstration | 8.33 min |
| Franka Sim Demonstration | 0.90 min |

:contentReference[oaicite:8]{index=8}

---

# Key Contributions

This project adapts:

```text
Human-to-Robot Learning
```

into:

```text
Robot-to-Robot Sim-to-Sim Learning
```

Main contributions:

- Fully reproducible simulation pipeline
- Cross-embodiment trajectory alignment
- Shared observation representation
- DTW-based temporal correspondence
- MixUp-based trajectory interpolation
- Stable sequence policy learning
- High rollout success rate

:contentReference[oaicite:9]{index=9}

---

# Limitations

Current limitations:

- Single task environment
- Limited robot embodiments
- No real-world deployment
- Sensitive to alignment quality
- Contact mismatch between robots

Possible failure sources:

- Imperfect DTW mapping
- Gripper mismatch
- Insufficient target coverage
- Simulation dynamics differences

:contentReference[oaicite:10]{index=10}

---

# Future Improvements

Potential future work:

- Real-to-sim transfer
- Sim-to-real deployment
- Multi-task learning
- Multi-object manipulation
- Vision-language conditioning
- Reinforcement learning fine-tuning
- Multi-robot transfer learning
- Diffusion-policy extensions
- Larger-scale demonstration datasets

---

# References

1. Liu et al. — ImMimic: Cross-Domain Imitation from Human Videos via Mapping and Interpolation
2. Chi et al. — Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
3. Wang et al. — MimicPlay
4. Kareer et al. — EgoMimic
5. Shaw et al. — VideoDex
6. Heppert et al. — DITTO
7. Xu et al. — XSkill
8. Memmel et al. — STRAP
9. Zhang et al. — MixUp
10. Mueller — Dynamic Time Warping
11. Gong et al. — DLOW
12. Qin et al. — AnyTeleop

:contentReference[oaicite:11]{index=11}

---

# License

MIT License

---

# Acknowledgements

Built using:

- MuJoCo
- robosuite
- PyTorch
- DTW
- Diffusion models
- Transformer architectures
- Cross-domain imitation learning research
```
