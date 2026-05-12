import json
import h5py
import numpy as np

FRANKA_PATH = "data/shared/franka_pick_place_shared_v1.hdf5"
UR5E_PATH = "data/shared/ur5e_pick_place_shared_v1.hdf5"
OUT_PATH = "data/aligned/ur5e_to_franka_pick_place_dtw_mapping_v1.json"


def load_demo_actions(path):
    demos = {}
    with h5py.File(path, "r") as f:
        data = f["data"]
        for demo_key in data.keys():
            demos[demo_key] = data[demo_key]["actions"][:]
    return demos


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
    return dp, path


def compute_match(source_actions, target_actions):
    cost = pairwise_l2_distance_matrix(source_actions, target_actions)
    dp, path = dtw(cost)
    total_cost = float(dp[-1, -1])
    normalized_cost = total_cost / len(path)
    return {
        "total_cost": total_cost,
        "normalized_cost": normalized_cost,
        "path_length": len(path),
        "source_len": int(source_actions.shape[0]),
        "target_len": int(target_actions.shape[0]),
    }


def main():
    franka_demos = load_demo_actions(FRANKA_PATH)
    ur5e_demos = load_demo_actions(UR5E_PATH)

    results = {
        "source_robot": "ur5e",
        "target_robot": "franka",
        "source_dataset": UR5E_PATH,
        "target_dataset": FRANKA_PATH,
        "task": "pick_place",
        "matches": {},
    }

    print("\nAll-pairs DTW costs:\n")

    for u_key, u_actions in ur5e_demos.items():
        pair_results = {}

        for f_key, f_actions in franka_demos.items():
            match = compute_match(u_actions, f_actions)
            pair_results[f_key] = match

            print(
                f"{u_key} -> {f_key} | "
                f"total={match['total_cost']:.6f}, "
                f"norm={match['normalized_cost']:.6f}, "
                f"path_len={match['path_length']}"
            )

        best_franka = min(
            pair_results.keys(),
            key=lambda k: pair_results[k]["normalized_cost"]
        )

        results["matches"][u_key] = {
            "best_target_demo": best_franka,
            "best_match_stats": pair_results[best_franka],
            "all_target_costs": pair_results,
        }

        print(
            f"BEST for {u_key}: {best_franka} "
            f"(norm={pair_results[best_franka]['normalized_cost']:.6f})\n"
        )

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved DTW mapping to: {OUT_PATH}")


if __name__ == "__main__":
    main()
