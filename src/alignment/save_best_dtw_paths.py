import json
import h5py
import numpy as np

FRANKA_PATH = "data/shared/franka_pick_place_shared_v1.hdf5"
UR5E_PATH = "data/shared/ur5e_pick_place_shared_v1.hdf5"
MAPPING_PATH = "data/aligned/ur5e_to_franka_pick_place_dtw_mapping_v1.json"
OUT_PATH = "data/aligned/ur5e_to_franka_pick_place_dtw_paths_v1.json"


def load_actions(path, demo_key):
    with h5py.File(path, "r") as f:
        return f["data"][demo_key]["actions"][:]


def pairwise_l2_distance_matrix(a, b):
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True).T
    dist_sq = a_sq + b_sq - 2 * (a @ b.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def dtw(cost):
    t1, t2 = cost.shape
    dp = np.full((t1, t2), np.inf, dtype=np.float64)
    dp[0, 0] = cost[0, 0]

    for i in range(t1):
        for j in range(t2):
            if i == 0 and j == 0:
                continue

            candidates = []
            if i > 0:
                candidates.append(dp[i - 1, j])
            if j > 0:
                candidates.append(dp[i, j - 1])
            if i > 0 and j > 0:
                candidates.append(dp[i - 1, j - 1])

            dp[i, j] = cost[i, j] + min(candidates)

    i, j = t1 - 1, t2 - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        moves = []
        if i > 0 and j > 0:
            moves.append((dp[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            moves.append((dp[i - 1, j], i - 1, j))
        if j > 0:
            moves.append((dp[i, j - 1], i, j - 1))

        _, i, j = min(moves, key=lambda x: x[0])
        path.append((i, j))

    path.reverse()
    return path


def main():
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    results = {
        "task": "pick_place",
        "source_robot": "ur5e",
        "target_robot": "franka",
        "paths": {},
    }

    for u_demo, info in mapping["matches"].items():
        f_demo = info["best_target_demo"]

        u_actions = load_actions(UR5E_PATH, u_demo)
        f_actions = load_actions(FRANKA_PATH, f_demo)

        cost = pairwise_l2_distance_matrix(u_actions, f_actions)
        path = dtw(cost)

        results["paths"][u_demo] = {
            "target_demo": f_demo,
            "path": path,
        }

        print(f"{u_demo} -> {f_demo}, path_len={len(path)}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f)

    print(f"\nSaved DTW paths to: {OUT_PATH}")


if __name__ == "__main__":
    main()
