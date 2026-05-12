import argparse
from pathlib import Path
import json

import h5py
import numpy as np


OBS_KEY = "eef_pose_obj_goal_phase"


def copy_dataset(src_group, dst_group, key):
    if key in src_group:
        src_group.copy(key, dst_group, name=key)


def make_next_obs(obs: np.ndarray) -> np.ndarray:
    next_obs = np.zeros_like(obs, dtype=np.float32)
    if len(obs) == 0:
        return next_obs
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]
    return next_obs.astype(np.float32)


def create_next_obs_from_obs(src_group, dst_group, obs_key=OBS_KEY):
    if "obs" not in src_group:
        raise KeyError("Source group is missing 'obs' group, cannot create next_obs")

    if obs_key not in src_group["obs"]:
        raise KeyError(
            f"Source obs is missing key '{obs_key}', cannot create next_obs. "
            f"Available keys: {list(src_group['obs'].keys())}"
        )

    obs = src_group["obs"][obs_key][:]
    next_obs = make_next_obs(obs)

    next_grp = dst_group.create_group("next_obs")
    next_grp.create_dataset(obs_key, data=next_obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--franka",
        default="data/shared/franka_pick_place_shared_v1.hdf5",
        help="Original Franka shared dataset",
    )
    parser.add_argument(
        "--interp",
        default="data/interpolated/ur5e_franka_pick_place_interpolated_v1.hdf5",
        help="Interpolated shared dataset",
    )
    parser.add_argument(
        "--output",
        default="data/final/pick_place_shared_cotrain_v1.hdf5",
        help="Merged output dataset",
    )
    parser.add_argument(
        "--obs-key",
        default=OBS_KEY,
        help="Shared low-dimensional observation key",
    )
    parser.add_argument(
        "--franka-train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for Franka demos",
    )
    parser.add_argument(
        "--interp-train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio for interpolated demos",
    )
    args = parser.parse_args()

    franka_path = Path(args.franka)
    interp_path = Path(args.interp)
    out_path = Path(args.output)

    if not franka_path.exists():
        raise FileNotFoundError(f"Franka shared dataset not found: {franka_path}")
    if not interp_path.exists():
        raise FileNotFoundError(f"Interpolated dataset not found: {interp_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    demo_counter = 0
    train_keys = []
    valid_keys = []

    env_args = {
        "env_name": "Lift",
        "type": 1,
        "env_kwargs": {
            "robots": "Panda",
            "controller_configs": {
                "control_freq": 20
            },
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "use_object_obs": True,
            "render_camera": "frontview",
            "ignore_done": False,
            "control_freq": 20,
            "horizon": 500
        }
    }

    with h5py.File(franka_path, "r") as f_franka, \
         h5py.File(interp_path, "r") as f_interp, \
         h5py.File(out_path, "w") as f_out:

        if "data" not in f_franka:
            raise KeyError("Franka dataset missing root group 'data'")
        if "data" not in f_interp:
            raise KeyError("Interpolated dataset missing root group 'data'")

        data_out = f_out.create_group("data")

        # --------------------------------------------------
        # FRANKA ORIGINAL DEMOS
        # --------------------------------------------------
        franka_data = f_franka["data"]
        franka_keys = sorted(franka_data.keys())

        n_franka = len(franka_keys)
        n_franka_train = int(args.franka_train_ratio * n_franka)
        if n_franka > 1:
            n_franka_train = max(1, min(n_franka_train, n_franka - 1))
        franka_train = set(franka_keys[:n_franka_train])
        franka_valid = set(franka_keys[n_franka_train:])

        for src_demo in franka_keys:
            src = franka_data[src_demo]
            demo_name = f"demo_{demo_counter}"
            dst = data_out.create_group(demo_name)

            copy_dataset(src, dst, "obs")

            if "next_obs" in src:
                copy_dataset(src, dst, "next_obs")
            else:
                create_next_obs_from_obs(src, dst, obs_key=args.obs_key)

            copy_dataset(src, dst, "actions")
            copy_dataset(src, dst, "rewards")
            copy_dataset(src, dst, "dones")

            copy_dataset(src, dst, "phases")
            copy_dataset(src, dst, "initial_cube_pos")
            copy_dataset(src, dst, "place_target_xy")

            for k, v in src.attrs.items():
                dst.attrs[k] = v

            if "actions" not in dst:
                raise KeyError(f"{src_demo} missing actions after copy")

            T = int(dst["actions"].shape[0])
            dst.attrs["num_samples"] = T
            dst.attrs["data_source"] = "franka_original"
            dst.attrs["task"] = "pick_place"
            dst.attrs["source_name"] = src_demo

            total_samples += T

            if src_demo in franka_train:
                train_keys.append(demo_name)
            elif src_demo in franka_valid:
                valid_keys.append(demo_name)
            else:
                raise RuntimeError(f"Unexpected Franka split state for {src_demo}")

            demo_counter += 1

        # --------------------------------------------------
        # INTERPOLATED DEMOS
        # --------------------------------------------------
        interp_data = f_interp["data"]
        interp_keys = sorted(interp_data.keys())

        n_interp = len(interp_keys)
        n_interp_train = int(args.interp_train_ratio * n_interp)
        if n_interp > 1:
            n_interp_train = max(1, min(n_interp_train, n_interp - 1))
        interp_train = set(interp_keys[:n_interp_train])
        interp_valid = set(interp_keys[n_interp_train:])

        for src_demo in interp_keys:
            src = interp_data[src_demo]
            demo_name = f"demo_{demo_counter}"
            dst = data_out.create_group(demo_name)

            copy_dataset(src, dst, "obs")

            if "next_obs" in src:
                copy_dataset(src, dst, "next_obs")
            else:
                create_next_obs_from_obs(src, dst, obs_key=args.obs_key)

            copy_dataset(src, dst, "actions")
            copy_dataset(src, dst, "rewards")
            copy_dataset(src, dst, "dones")

            copy_dataset(src, dst, "ur5e_phases")
            copy_dataset(src, dst, "franka_phases")

            for k, v in src.attrs.items():
                dst.attrs[k] = v

            if "actions" not in dst:
                raise KeyError(f"{src_demo} missing actions after copy")

            T = int(dst["actions"].shape[0])
            dst.attrs["num_samples"] = T
            dst.attrs["data_source"] = "interpolated_mixup"
            dst.attrs["task"] = "pick_place"
            dst.attrs["source_name"] = src_demo

            total_samples += T

            if src_demo in interp_train:
                train_keys.append(demo_name)
            elif src_demo in interp_valid:
                valid_keys.append(demo_name)
            else:
                raise RuntimeError(f"Unexpected interpolated split state for {src_demo}")

            demo_counter += 1

        data_out.attrs["total"] = int(total_samples)
        data_out.attrs["env_args"] = json.dumps(env_args)

        mask = f_out.create_group("mask")
        mask.create_dataset("train", data=np.array(train_keys, dtype="S"))
        mask.create_dataset("valid", data=np.array(valid_keys, dtype="S"))

    print(f"Saved: {out_path}")
    print(f"Total demos: {demo_counter}")
    print(f"Total samples: {total_samples}")
    print(f"Train: {len(train_keys)}")
    print(f"Valid: {len(valid_keys)}")


if __name__ == "__main__":
    main()
