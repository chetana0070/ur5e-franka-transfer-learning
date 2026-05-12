import argparse
from pathlib import Path

import h5py
import numpy as np


def make_next_obs(obs: np.ndarray) -> np.ndarray:
    next_obs = np.zeros_like(obs, dtype=np.float32)
    if len(obs) == 0:
        return next_obs
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]
    return next_obs


def ensure_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def interpolate_pair(
    ur5e_obs: np.ndarray,
    franka_obs: np.ndarray,
    ur5e_actions: np.ndarray,
    franka_actions: np.ndarray,
    alpha: float,
):
    T = min(len(ur5e_obs), len(franka_obs), len(ur5e_actions), len(franka_actions))
    ur5e_obs = ensure_float32(ur5e_obs[:T])
    franka_obs = ensure_float32(franka_obs[:T])
    ur5e_actions = ensure_float32(ur5e_actions[:T])
    franka_actions = ensure_float32(franka_actions[:T])

    mixed_obs = (1.0 - alpha) * ur5e_obs + alpha * franka_obs
    mixed_actions = (1.0 - alpha) * ur5e_actions + alpha * franka_actions

    return mixed_obs, mixed_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/aligned/ur5e_franka_pick_place_aligned_pairs_v1.hdf5",
    )
    parser.add_argument(
        "--output",
        default="data/interpolated/ur5e_franka_pick_place_interpolated_v1.hdf5",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "--copy-endpoints",
        action="store_true",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Aligned-pairs file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = list(args.alphas)
    if args.copy_endpoints:
        alphas = [0.0] + alphas + [1.0]

    with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        if "data" not in f_in:
            raise KeyError("Expected root group 'data' in aligned pairs file")

        data_in = f_in["data"]
        data_out = f_out.create_group("data")

        demo_count = 0
        total_samples = 0

        for pair_key in sorted(data_in.keys()):
            pair_grp = data_in[pair_key]

            required = ["ur5e_obs", "franka_obs", "ur5e_actions", "franka_actions"]
            for k in required:
                if k not in pair_grp:
                    raise KeyError(f"Missing key '{k}' in pair group '{pair_key}'")

            ur5e_obs = pair_grp["ur5e_obs"][:]
            franka_obs = pair_grp["franka_obs"][:]
            ur5e_actions = pair_grp["ur5e_actions"][:]
            franka_actions = pair_grp["franka_actions"][:]

            for alpha in alphas:
                mixed_obs, mixed_actions = interpolate_pair(
                    ur5e_obs=ur5e_obs,
                    franka_obs=franka_obs,
                    ur5e_actions=ur5e_actions,
                    franka_actions=franka_actions,
                    alpha=alpha,
                )

                T = len(mixed_obs)
                if T == 0:
                    continue

                demo_name = f"demo_{demo_count}"
                demo_grp = data_out.create_group(demo_name)

                obs_grp = demo_grp.create_group("obs")
                next_obs_grp = demo_grp.create_group("next_obs")

                obs_grp.create_dataset("eef_pose_w_gripper", data=ensure_float32(mixed_obs))
                next_obs_grp.create_dataset(
                    "eef_pose_w_gripper",
                    data=ensure_float32(make_next_obs(mixed_obs)),
                )

                demo_grp.create_dataset("actions", data=ensure_float32(mixed_actions))

                rewards = np.zeros((T,), dtype=np.float32)
                rewards[-1] = 1.0
                demo_grp.create_dataset("rewards", data=rewards)

                dones = np.zeros((T,), dtype=np.bool_)
                dones[-1] = True
                demo_grp.create_dataset("dones", data=dones)

                if "ur5e_phases" in pair_grp:
                    demo_grp.create_dataset("ur5e_phases", data=pair_grp["ur5e_phases"][:T])
                if "franka_phases" in pair_grp:
                    demo_grp.create_dataset("franka_phases", data=pair_grp["franka_phases"][:T])

                demo_grp.attrs["num_samples"] = int(T)
                demo_grp.attrs["source_pair"] = pair_key
                demo_grp.attrs["alpha"] = float(alpha)
                demo_grp.attrs["interpolation_type"] = "linear_mixup"
                demo_grp.attrs["task"] = "pick_place"

                if "source_demo" in pair_grp.attrs:
                    demo_grp.attrs["source_demo"] = pair_grp.attrs["source_demo"]
                if "target_demo" in pair_grp.attrs:
                    demo_grp.attrs["target_demo"] = pair_grp.attrs["target_demo"]

                demo_count += 1
                total_samples += T

        data_out.attrs["total"] = int(total_samples)

    print(f"Saved: {out_path}")
    print(f"Interpolated demos: {demo_count}")
    print(f"Total samples: {total_samples}")
    print(f"Alphas used: {alphas}")


if __name__ == "__main__":
    main()
