# Results

## Project
Adapted ImMimic-style sim-to-sim pick-and-place transfer from UR5e (source) to Franka (target).

## Pipeline
1. Collect UR5e and Franka pick-place demonstrations
2. Convert both to a shared representation
3. Align trajectories using DTW
4. Generate interpolated trajectories with MixUp
5. Merge shared and interpolated data into a co-training dataset
6. Train a shared diffusion policy
7. Evaluate on Franka

## Final Data Files
- Raw Franka demos: `data/raw/franka_pick_place_random_goal.hdf5`
- Raw UR5e demos: `data/raw/ur5e_pick_place_random_goal_50eps_v2.hdf5`
- Shared Franka demos: `data/shared/franka_pick_place_shared_v1.hdf5`
- Shared UR5e demos: `data/shared/ur5e_pick_place_shared_v1.hdf5`
- Aligned pairs: `data/aligned/ur5e_franka_pick_place_aligned_pairs_v1.hdf5`
- Interpolated data: `data/interpolated/ur5e_franka_pick_place_interpolated_v1.hdf5`
- Final training dataset: `data/final/pick_place_shared_cotrain_v1.hdf5`

## Model
- Backend: DiffusionPolicyUNet
- Observation key: `eef_pose_w_gripper`
- Observation dimension: 8
- Action dimension: 7

## Training Summary
- Train demos: 19
- Validation demos: 1
- Train sequences: 9370
- Validation sequences: 470
- Best validation loss: 0.9854340382984706
- Best checkpoint: `checkpoints/best_model.pth`

## Notes
This project follows an adapted ImMimic-style sim-to-sim workflow in which UR5e demonstrations replace the original paper’s human-video source trajectories, while preserving shared representation learning, DTW alignment, interpolation, and co-training.
