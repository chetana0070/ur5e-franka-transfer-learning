import json
import h5py
import numpy as np

UR5E_PATH = "data/shared/ur5e_pick_place_shared_v1.hdf5"
FRANKA_PATH = "data/shared/franka_pick_place_shared_v1.hdf5"
DTW_PATH = "data/aligned/ur5e_to_franka_pick_place_dtw_paths_v1.json"
OUT_PATH = "data/aligned/ur5e_franka_pick_place_aligned_pairs_v1.hdf5"

OBS_KEY = "eef_pose_obj_goal_phase"


def load_demo_data(path, demo_key):
    with h5py.File(path, "r") as f:
        demo = f["data"][demo_key]

        if "obs" not in demo:
            raise KeyError(f"{demo_key} missing obs group in {path}")
        if OBS_KEY not in demo["obs"]:
            raise KeyError(
                f"{demo_key} missing obs key '{OBS_KEY}' in {path}. "
                f"Available: {list(demo['obs'].keys())}"
            )

        obs = demo["obs"][OBS_KEY][:]
        actions = demo["actions"][:]
        rewards = demo["rewards"][:] if "rewards" in demo else np.zeros((len(actions),), dtype=np.float32)
        phases = demo["phases"][:] if "phases" in demo else None

    return obs, actions, rewards, phases


def main():
    with open(DTW_PATH, "r") as f:
        dtw_data = json.load(f)

    path_dict = dtw_data["paths"]

    with h5py.File(OUT_PATH, "w") as f_out:
        data_grp = f_out.create_group("data")

        total_pairs = 0

        for src_demo, info in path_dict.items():
            tgt_demo = info["target_demo"]
            path = info["path"]

            u_obs, u_act, u_rew, u_phase = load_demo_data(UR5E_PATH, src_demo)
            f_obs, f_act, f_rew, f_phase = load_demo_data(FRANKA_PATH, tgt_demo)

            grp = data_grp.create_group(src_demo)

            ur5e_obs_aligned = []
            ur5e_act_aligned = []
            ur5e_rew_aligned = []

            franka_obs_aligned = []
            franka_act_aligned = []
            franka_rew_aligned = []

            ur5e_phase_aligned = []
            franka_phase_aligned = []

            for u_idx, f_idx in path:
                ur5e_obs_aligned.append(u_obs[u_idx])
                ur5e_act_aligned.append(u_act[u_idx])
                ur5e_rew_aligned.append(u_rew[u_idx])

                franka_obs_aligned.append(f_obs[f_idx])
                franka_act_aligned.append(f_act[f_idx])
                franka_rew_aligned.append(f_rew[f_idx])

                if u_phase is not None:
                    ur5e_phase_aligned.append(u_phase[u_idx])
                if f_phase is not None:
                    franka_phase_aligned.append(f_phase[f_idx])

            grp.create_dataset("ur5e_obs", data=np.asarray(ur5e_obs_aligned, dtype=np.float32))
            grp.create_dataset("ur5e_actions", data=np.asarray(ur5e_act_aligned, dtype=np.float32))
            grp.create_dataset("ur5e_rewards", data=np.asarray(ur5e_rew_aligned, dtype=np.float32))

            grp.create_dataset("franka_obs", data=np.asarray(franka_obs_aligned, dtype=np.float32))
            grp.create_dataset("franka_actions", data=np.asarray(franka_act_aligned, dtype=np.float32))
            grp.create_dataset("franka_rewards", data=np.asarray(franka_rew_aligned, dtype=np.float32))

            if len(ur5e_phase_aligned) > 0:
                grp.create_dataset("ur5e_phases", data=np.asarray(ur5e_phase_aligned, dtype="S32"))
            if len(franka_phase_aligned) > 0:
                grp.create_dataset("franka_phases", data=np.asarray(franka_phase_aligned, dtype="S32"))

            grp.attrs["source_demo"] = src_demo
            grp.attrs["target_demo"] = tgt_demo
            grp.attrs["num_pairs"] = len(path)
            grp.attrs["task"] = "pick_place"
            grp.attrs["obs_key"] = OBS_KEY

            total_pairs += len(path)
            print(f"{src_demo} -> {tgt_demo}: saved {len(path)} aligned pairs")

        data_grp.attrs["total_pairs"] = total_pairs
        data_grp.attrs["task"] = "pick_place"
        data_grp.attrs["obs_key"] = OBS_KEY

    print(f"\nSaved aligned pair dataset to: {OUT_PATH}")
    print(f"Total aligned pairs: {total_pairs}")


if __name__ == "__main__":
    main()
