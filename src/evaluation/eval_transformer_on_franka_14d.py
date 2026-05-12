import sys
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.transformer.transformer_policy import TransformerPolicy


DEFAULT_CKPT = "checkpoints/transformer_single_step_14d/best_transformer_single_step_14d.pt"
DEFAULT_DATASET = "data/final/pick_place_shared_cotrain_v1.hdf5"


def load_action_limits(dataset_path):
    mins, maxs = [], []
    with h5py.File(dataset_path, "r") as f:
        for demo_key in f["data"].keys():
            a = f["data"][demo_key]["actions"][:].astype(np.float32)
            mins.append(a.min(axis=0))
            maxs.append(a.max(axis=0))
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def make_env(render=True, horizon=900):
    controller_config = load_composite_controller_config(controller="BASIC")
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera="frontview",
        ignore_done=True,
        control_freq=20,
        horizon=horizon,
    )


def build_obs_vec_14d(obs, goal_xy, phase):
    eef_pos = obs["robot0_eef_pos"].astype(np.float32)          # 3
    eef_quat = obs["robot0_eef_quat"].astype(np.float32)        # 4
    grip = np.array([obs["robot0_gripper_qpos"][0]], dtype=np.float32)  # 1
    cube_pos = obs["cube_pos"].astype(np.float32)               # 3
    goal_xy = np.array(goal_xy, dtype=np.float32)               # 2
    phase = np.array([phase], dtype=np.float32)                 # 1

    vec = np.concatenate(
        [eef_pos, eef_quat, grip, cube_pos, goal_xy, phase],
        axis=0,
    ).astype(np.float32)

    if vec.shape != (14,):
        raise ValueError(f"Expected obs vec shape (14,), got {vec.shape}")

    return vec


def check_success(env):
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=900)
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument("--end-wait-sec", type=float, default=10.0)
    parser.add_argument("--no-render", action="store_true")

    parser.add_argument("--goal-x", type=float, default=0.15)
    parser.add_argument("--goal-y", type=float, default=0.15)

    parser.add_argument("--no-binary-gripper", action="store_true")
    parser.add_argument("--no-dataset-clip", action="store_true")
    parser.add_argument("--stop-on-lift-success", action="store_true")
    parser.add_argument("--hold-after-first-close", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    render = not args.no_render
    binary_gripper = not args.no_binary_gripper
    dataset_clip = not args.no_dataset_clip
    goal_xy = np.array([args.goal_x, args.goal_y], dtype=np.float32)

    print("=" * 80)
    print("DEVICE:", device)
    print("CKPT:", args.ckpt)
    print("DATASET:", args.dataset)
    print("GOAL_XY:", goal_xy)
    print("hold_after_first_close:", args.hold_after_first_close)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]

    print("best epoch:", ckpt["epoch"])
    print("best valid_loss:", ckpt["valid_loss"])
    print("model cfg:", cfg)

    assert cfg["obs_dim"] == 14, f"Expected 14D checkpoint, got obs_dim={cfg['obs_dim']}"

    model = TransformerPolicy(
        obs_dim=cfg["obs_dim"],
        action_dim=cfg["action_dim"],
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    obs_mean = ckpt["obs_mean"].astype(np.float32)
    obs_std = ckpt["obs_std"].astype(np.float32)
    action_mean = ckpt["action_mean"].astype(np.float32)
    action_std = ckpt["action_std"].astype(np.float32)

    action_min, action_max = load_action_limits(args.dataset)
    print("action_min:", action_min)
    print("action_max:", action_max)

    successes = 0

    for ep in range(args.episodes):
        print("=" * 80)
        print(f"EPISODE {ep}")

        env = make_env(render=render, horizon=args.horizon)
        obs = env.reset()

        first_phase = 0.0
        first_vec = build_obs_vec_14d(obs, goal_xy, first_phase)
        obs_buffer = [first_vec.copy() for _ in range(cfg["seq_len"])]

        lift_success = False
        first_close_step = None

        for t in range(args.horizon):
            phase = np.clip(t / max(args.horizon - 1, 1), 0.0, 1.0).astype(np.float32)

            current_vec = build_obs_vec_14d(obs, goal_xy, phase)
            obs_buffer.append(current_vec)

            obs_seq = np.stack(obs_buffer[-cfg["seq_len"]:], axis=0).astype(np.float32)
            obs_seq_norm = (obs_seq - obs_mean) / obs_std

            x = torch.tensor(obs_seq_norm, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(x).squeeze(0).cpu().numpy().astype(np.float32)

            action = pred_norm * action_std + action_mean
            action = action.astype(np.float32)

            action[3:6] = 0.0

            if binary_gripper:
                action[6] = 1.0 if action[6] >= 0.0 else -1.0

            if action[6] > 0 and first_close_step is None:
                first_close_step = t
                print(f"FIRST_GRIPPER_CLOSE at step {t}")

            if args.hold_after_first_close and first_close_step is not None:
                action[6] = 1.0

            if dataset_clip:
                action = np.clip(action, action_min, action_max)

            low, high = env.action_spec
            action = np.clip(action, low, high).astype(np.float32)

            if t % args.debug_every == 0:
                print(
                    f"t={t:03d} | "
                    f"phase={phase:.3f} | "
                    f"eef={np.round(obs['robot0_eef_pos'], 4)} | "
                    f"cube={np.round(obs['cube_pos'], 4)} | "
                    f"goal={np.round(goal_xy, 4)} | "
                    f"action={np.round(action, 4)}"
                )

            obs, reward, done, info = env.step(action)

            if render:
                env.render()
                time.sleep(0.03)

            if check_success(env):
                if not lift_success:
                    print(f"LIFT_SUCCESS at step {t}")
                    lift_success = True
                    successes += 1

                if args.stop_on_lift_success:
                    break

        if not lift_success:
            print("NO_LIFT_SUCCESS")
        else:
            print("EPISODE_CONTINUED_AFTER_LIFT")

        if render and args.end_wait_sec > 0:
            print(f"Waiting {args.end_wait_sec} sec before closing window...")
            time.sleep(args.end_wait_sec)

        env.close()

    print("=" * 80)
    print("RESULT")
    print(f"Success: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
