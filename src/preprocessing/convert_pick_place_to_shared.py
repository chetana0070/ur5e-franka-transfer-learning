import h5py
import numpy as np


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


def decode_phase_array(phases):
    out = []
    for p in phases:
        if isinstance(p, bytes):
            p = p.decode("utf-8")
        out.append(PHASE_TO_ID.get(p, 0))
    return np.asarray(out, dtype=np.float32).reshape(-1, 1) / PHASE_DENOM


def convert(raw_path, out_path):
    with h5py.File(raw_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        if "data" not in f_in:
            raise KeyError(f"Expected root group 'data' in raw dataset: {raw_path}")

        data_in = f_in["data"]
        data_out = f_out.create_group("data")

        total = 0
        kept = 0
        skipped = 0

        for demo_key in sorted(data_in.keys()):
            demo_in = data_in[demo_key]

            if "success" in demo_in.attrs and not bool(demo_in.attrs["success"]):
                print(f"Skipping failed demo: {demo_key}")
                skipped += 1
                continue

            obs_in = demo_in["obs"]

            required_obs = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object_pos"]
            missing = [k for k in required_obs if k not in obs_in]
            if missing:
                raise KeyError(f"{demo_key} missing required obs keys: {missing}")

            eef_pos = obs_in["robot0_eef_pos"][:]            # (T, 3)
            eef_quat = obs_in["robot0_eef_quat"][:]          # (T, 4)
            gripper_qpos = obs_in["robot0_gripper_qpos"][:]  # (T, G)
            object_pos = obs_in["object_pos"][:]             # (T, 3)

            if "place_target_xy" not in demo_in:
                raise KeyError(f"{demo_key} missing place_target_xy")
            place_target_xy = demo_in["place_target_xy"][:]  # (T, 2)

            if "phases" not in demo_in:
                raise KeyError(f"{demo_key} missing phases")
            phase_scalar = decode_phase_array(demo_in["phases"][:])  # (T, 1)

            gripper_scalar = np.mean(gripper_qpos, axis=1, keepdims=True)  # (T, 1)

            obs_shared = np.concatenate(
                [
                    eef_pos,           # 3
                    eef_quat,          # 4
                    gripper_scalar,    # 1
                    object_pos,        # 3
                    place_target_xy,   # 2
                    phase_scalar,      # 1
                ],
                axis=1,
            ).astype(np.float32)       # total = 14

            demo_out = data_out.create_group(demo_key)
            obs_group = demo_out.create_group("obs")
            obs_group.create_dataset("eef_pose_obj_goal_phase", data=obs_shared)

            demo_out.create_dataset("actions", data=demo_in["actions"][:])

            if "rewards" in demo_in:
                demo_out.create_dataset("rewards", data=demo_in["rewards"][:])
            else:
                rewards = np.zeros((obs_shared.shape[0],), dtype=np.float32)
                rewards[-1] = 1.0
                demo_out.create_dataset("rewards", data=rewards)

            if "dones" in demo_in:
                demo_out.create_dataset("dones", data=demo_in["dones"][:])
            else:
                dones = np.zeros((obs_shared.shape[0],), dtype=np.bool_)
                dones[-1] = True
                demo_out.create_dataset("dones", data=dones)

            demo_out.create_dataset("phases", data=demo_in["phases"][:])
            demo_out.create_dataset("initial_cube_pos", data=demo_in["initial_cube_pos"][:])
            demo_out.create_dataset("place_target_xy", data=demo_in["place_target_xy"][:])

            for attr_key, attr_val in demo_in.attrs.items():
                demo_out.attrs[attr_key] = attr_val

            demo_out.attrs["num_samples"] = obs_shared.shape[0]
            demo_out.attrs["task"] = "pick_place"

            total += obs_shared.shape[0]
            kept += 1

            print(
                f"Converted {demo_key}: "
                f"obs={obs_shared.shape}, "
                f"actions={demo_in['actions'][:].shape}"
            )

        data_out.attrs["total"] = total
        data_out.attrs["num_demos_kept"] = kept
        data_out.attrs["num_demos_skipped"] = skipped

    print(f"\nSaved: {out_path}")
    print(f"Total samples: {total}")
    print(f"Demos kept: {kept}")
    print(f"Demos skipped: {skipped}")


def main():
    convert(
        "data/raw/franka_pick_place_random_goal.hdf5",
        "data/shared/franka_pick_place_shared_v1.hdf5",
    )
    convert(
        "data/raw/ur5e_pick_place_random_goal.hdf5",
        "data/shared/ur5e_pick_place_shared_v1.hdf5",
    )


if __name__ == "__main__":
    main()
