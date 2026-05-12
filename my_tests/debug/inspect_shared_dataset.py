import h5py
import numpy as np

path = "data/final/pick_place_shared_cotrain_v1.hdf5"

print("=" * 80)
print("DATASET:", path)

with h5py.File(path, "r") as f:
    print("ROOT KEYS:", list(f.keys()))
    print("ROOT ATTRS:")
    for k, v in f.attrs.items():
        print(" ", k, "=", v)

    data = f["data"]
    demos = sorted(list(data.keys()))
    print("=" * 80)
    print("NUM DEMOS:", len(demos))
    print("FIRST 10 DEMOS:", demos[:10])

    all_actions = []
    lengths = []

    for demo_name in demos[:5]:
        demo = data[demo_name]
        print("=" * 80)
        print("DEMO:", demo_name)
        print("KEYS:", list(demo.keys()))

        if "obs" in demo:
            print("OBS KEYS:", list(demo["obs"].keys()))
            for k in demo["obs"].keys():
                arr = demo["obs"][k]
                print(f"obs/{k}: shape={arr.shape}, dtype={arr.dtype}")

        if "actions" in demo:
            a = demo["actions"][:]
            print("actions shape:", a.shape)
            print("actions first 3:")
            print(np.round(a[:3], 4))
            print("actions min:", np.round(a.min(axis=0), 4))
            print("actions max:", np.round(a.max(axis=0), 4))
            print("actions mean:", np.round(a.mean(axis=0), 4))
            print("actions std:", np.round(a.std(axis=0), 4))

    for demo_name in demos:
        demo = data[demo_name]
        if "actions" not in demo:
            continue
        a = demo["actions"][:]
        all_actions.append(a)
        lengths.append(len(a))

    A = np.concatenate(all_actions, axis=0)

    print("=" * 80)
    print("GLOBAL")
    print("num trajectories:", len(lengths))
    print("total samples:", A.shape[0])
    print("action dim:", A.shape[1])
    print("traj length min/mean/max:", min(lengths), np.mean(lengths), max(lengths))
    print("global action min:", np.round(A.min(axis=0), 4))
    print("global action max:", np.round(A.max(axis=0), 4))
    print("global action mean:", np.round(A.mean(axis=0), 4))
    print("global action std:", np.round(A.std(axis=0), 4))

    print("=" * 80)
    print("CHECKS")
    print("action_dim_ok:", A.shape[1] == 7)
    print("has_obs:", "obs" in data[demos[0]])
    print("done")
