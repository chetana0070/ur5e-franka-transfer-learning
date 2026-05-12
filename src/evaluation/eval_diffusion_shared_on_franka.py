import os
import json
import numpy as np
import torch

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from robomimic.config import config_factory
from robomimic.algo import algo_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


CONFIG_PATH = "src/training/diffusion_shared_cotrain.json"
MODEL_PATH = "checkpoints/best_model.pth"

NUM_EPISODES = 5
HORIZON = 300
RENDER = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


PHASE_TO_ID = {
    "approach_pick": 0,
    "align_pick": 1,
    "descend_pick": 2,
    "close_gripper": 3,
    "lift": 4,
    "move_mid": 5,
    "move_to_place": 6,
    "descend_place": 7,
    "open_gripper": 8,
    "retreat": 9,
}
PHASE_DENOM = float(max(PHASE_TO_ID.values()))


print("CONFIG PATH:", CONFIG_PATH)
print("CONFIG EXISTS:", os.path.exists(CONFIG_PATH))
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

with open(CONFIG_PATH, "r") as f:
    ext_cfg = json.load(f)

config = config_factory(ext_cfg["algo_name"])
with config.values_unlocked():
    config.update(ext_cfg)

ObsUtils.initialize_obs_utils_with_config(config)
print("Config loaded")

dataset_path = config.train.data
ds_format = config.train.data_format if hasattr(config.train, "data_format") else "robomimic"

print("DATASET PATH:", dataset_path)
print("DATASET EXISTS:", os.path.exists(dataset_path))
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

shape_meta = FileUtils.get_shape_metadata_from_dataset(
    dataset_path=dataset_path,
    action_keys=config.train.action_keys,
    all_obs_keys=config.all_obs_keys,
    ds_format=ds_format,
    verbose=True,
)

obs_key_shapes = shape_meta["all_shapes"]
ac_dim = shape_meta["ac_dim"]

print("Metadata loaded")
print("obs shapes:", obs_key_shapes)
print("action dim from dataset:", ac_dim)

model = algo_factory(
    algo_name=config.algo_name,
    config=config,
    obs_key_shapes=obs_key_shapes,
    ac_dim=ac_dim,
    device=device,
)

ckpt = torch.load(MODEL_PATH, map_location=device)

if isinstance(ckpt, dict) and "model" in ckpt:
    model.deserialize(ckpt["model"])
else:
    model.deserialize(ckpt)

if hasattr(model, "set_eval"):
    model.set_eval()
elif hasattr(model, "nets"):
    model.nets.eval()

print("Model loaded successfully")


def make_env():
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera="frontview",
        ignore_done=True,
        control_freq=20,
        horizon=500,
    )
    return env


def get_eef_site_id(env):
    site_ref = env.robots[0].eef_site_id
    if isinstance(site_ref, dict):
        if "right" in site_ref:
            site_ref = site_ref["right"]
        else:
            site_ref = next(iter(site_ref.values()))
    if isinstance(site_ref, str):
        return env.sim.model.site_name2id(site_ref)
    return int(site_ref)


def get_object_body_id(env):
    for candidate in ["cube_main", "cube", "Cube", "object", "Milk0", "Bread0", "Cereal0", "Can0"]:
        if candidate in list(env.sim.model.body_names):
            return env.sim.model.body_name2id(candidate)
    raise ValueError("Object body not found")


def build_policy_obs(obs, env, phase_name="approach_pick", place_target_xy=None):
    needed = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    missing = [k for k in needed if k not in obs]
    if missing:
        raise KeyError(f"Missing keys needed to build policy obs: {missing}. Available keys: {list(obs.keys())}")

    obj_body_id = get_object_body_id(env)
    object_pos = env.sim.data.body_xpos[obj_body_id].copy().astype(np.float32)

    if place_target_xy is None:
        # fallback heuristic if no explicit goal tracker is used in eval
        place_target_xy = np.array([0.14, 0.14], dtype=np.float32)

    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    gripper_scalar = np.array([float(np.mean(gripper_qpos))], dtype=np.float32)

    phase_scalar = np.array([PHASE_TO_ID.get(phase_name, 0) / PHASE_DENOM], dtype=np.float32)

    vec = np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),   # 3
        np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),  # 4
        gripper_scalar,                                                     # 1
        object_pos.reshape(-1),                                             # 3
        place_target_xy.reshape(-1),                                        # 2
        phase_scalar,                                                       # 1
    ]).astype(np.float32)

    if vec.shape[0] != 14:
        raise ValueError(f"Expected 14-dim eef_pose_obj_goal_phase, got shape {vec.shape}")

    return {
        "eef_pose_obj_goal_phase": torch.tensor(
            vec, dtype=torch.float32, device=device
        ).unsqueeze(0)
    }


def extract_success(reward, info):
    if isinstance(info, dict):
        for key in ["success", "task_success", "is_success"]:
            if key in info:
                val = info[key]
                if isinstance(val, dict):
                    if "task" in val:
                        return bool(val["task"])
                    return any(bool(v) for v in val.values())
                return bool(val)
    return reward > 0


success_count = 0

for ep in range(NUM_EPISODES):
    env = make_env()
    obs = env.reset()

    if hasattr(model, "reset"):
        model.reset()

    print(f"\n--- Episode {ep} ---")
    ep_success = False

    place_target_xy = np.array([0.14, 0.14], dtype=np.float32)

    try:
        for t in range(HORIZON):
            policy_obs = build_policy_obs(
                obs=obs,
                env=env,
                phase_name="approach_pick",
                place_target_xy=place_target_xy,
            )

            with torch.no_grad():
                action_seq = model.get_action(policy_obs)

            action = action_seq[0].detach().cpu().numpy().astype(np.float32)
            obs, reward, done, info = env.step(action)

            if RENDER:
                env.render()

            if extract_success(reward, info):
                print(f"SUCCESS at step {t}")
                success_count += 1
                ep_success = True
                break

            if done:
                break

        if not ep_success:
            print(f"FAILED after {t + 1} steps")

    finally:
        env.close()

print("\n===================================")
print(f"Successes: {success_count} / {NUM_EPISODES}")
print(f"Success rate: {success_count / NUM_EPISODES:.3f}")
print("===================================")
