import os
import h5py
import time
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


SAVE_DIR = "data/raw"
SAVE_PATH = os.path.join(SAVE_DIR, "franka_pick_place_random_goal.hdf5")

NUM_EPISODES = 20
HORIZON = 1000
CONTROL_FREQ = 20

RENDER = False
PRINT_EVERY = 50
FINAL_VIEW_TIME = 0.0

ROBOT = "Panda"
COMPOSITE_CONTROLLER = "BASIC"
ARM_CONTROLLER = "OSC_POSE"

OPEN_ACTION = -1.0
CLOSE_ACTION = 1.0

BASE_KP_POS = 32.0
BASE_MAX_DPOS = 0.12

XY_TOL_OBJ = 0.018
XY_TOL_PLACE = 0.020
Z_TOL_DESCEND = 0.018

PLACE_X_RANGE = (0.10, 0.18)
PLACE_Y_RANGE = (0.10, 0.18)
MIN_PICK_PLACE_DIST = 0.10

OBJECT_BODY_CANDIDATES = [
    "cube_main",
    "cube",
    "Cube",
    "object",
    "Milk0",
    "Bread0",
    "Cereal0",
    "Can0",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clip_norm(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    if norm > max_norm:
        return vec / norm * max_norm
    return vec


def sample_place_target_xy(obj_xy, x_range, y_range, min_dist=0.10, max_tries=100):
    obj_xy = np.asarray(obj_xy, dtype=np.float32)

    for _ in range(max_tries):
        target = np.array(
            [
                np.random.uniform(*x_range),
                np.random.uniform(*y_range),
            ],
            dtype=np.float32,
        )
        if np.linalg.norm(target - obj_xy) >= min_dist:
            return target

    return np.array([np.mean(x_range), np.mean(y_range)], dtype=np.float32)


def get_eef_site_id(env):
    site_ref = env.robots[0].eef_site_id

    if isinstance(site_ref, dict):
        if "right" in site_ref:
            site_ref = site_ref["right"]
        else:
            site_ref = next(iter(site_ref.values()))

    if isinstance(site_ref, str):
        return env.sim.model.site_name2id(site_ref)

    if isinstance(site_ref, (int, np.integer)):
        return int(site_ref)

    raise TypeError(f"Unsupported eef_site_id type: {type(site_ref)} | value: {site_ref}")


def get_eef_pos(env):
    site_id = get_eef_site_id(env)
    return env.sim.data.site_xpos[site_id].copy()


def get_object_body_id(env):
    body_names = list(env.sim.model.body_names)

    for candidate in OBJECT_BODY_CANDIDATES:
        if candidate in body_names:
            return env.sim.model.body_name2id(candidate), candidate

    print("\n[DEBUG] Available body names:")
    for name in body_names:
        print(" ", name)

    raise ValueError("Could not find object body name.")


def get_object_pos(env, obj_body_id):
    return env.sim.data.body_xpos[obj_body_id].copy()


def reached_xyz(current, target, xy_tol, z_tol):
    xy_ok = np.linalg.norm(current[:2] - target[:2]) < xy_tol
    z_ok = abs(current[2] - target[2]) < z_tol
    return xy_ok and z_ok


def build_action_towards_target(env, target_xyz, gripper_cmd, kp_pos, max_dpos):
    eef_pos = get_eef_pos(env)
    dpos = kp_pos * (target_xyz - eef_pos)
    dpos = clip_norm(dpos, max_dpos)

    action = np.zeros(7, dtype=np.float32)
    action[:3] = dpos
    action[3:6] = 0.0
    action[6] = gripper_cmd
    return action


def make_controller_config():
    controller_config = load_composite_controller_config(
        controller=COMPOSITE_CONTROLLER,
        robot=ROBOT,
    )

    if "body_parts" in controller_config and "arms" in controller_config["body_parts"]:
        if "right" in controller_config["body_parts"]["arms"]:
            controller_config["body_parts"]["arms"]["right"]["type"] = ARM_CONTROLLER

    return controller_config


def settle_scene(env, steps=20):
    for _ in range(steps):
        zero_action = np.zeros(7, dtype=np.float32)
        env.step(zero_action)
        if RENDER:
            env.render()


def sample_episode_style():
    style = np.random.choice(["direct", "side_arc", "high_arc", "fast", "slow"])

    base = {
        "style_name": style,
        "approach_above_z": np.random.uniform(0.09, 0.14),
        "grasp_z_offset": np.random.uniform(0.002, 0.008),
        "lift_height": np.random.uniform(0.12, 0.18),
        "place_above_z": np.random.uniform(0.10, 0.15),
        "place_z_offset": np.random.uniform(0.003, 0.010),
        "retreat_z": np.random.uniform(0.16, 0.22),
        "speed_scale": 1.0,
    }

    if style == "direct":
        base["side_offset"] = np.array([0.0, 0.0], dtype=np.float32)
        base["extra_arc_z"] = 0.0
    elif style == "side_arc":
        base["side_offset"] = np.array(
            [np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)],
            dtype=np.float32,
        )
        base["extra_arc_z"] = 0.02
    elif style == "high_arc":
        base["side_offset"] = np.array(
            [np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03, 0.03)],
            dtype=np.float32,
        )
        base["extra_arc_z"] = 0.06
    elif style == "fast":
        base["side_offset"] = np.array([0.0, 0.0], dtype=np.float32)
        base["extra_arc_z"] = 0.0
        base["speed_scale"] = 1.5
    elif style == "slow":
        base["side_offset"] = np.array([0.0, 0.0], dtype=np.float32)
        base["extra_arc_z"] = 0.0
        base["speed_scale"] = 0.65

    return base


def extract_obs_fields(obs):
    required = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    missing = [k for k in required if k not in obs]
    if missing:
        raise KeyError(f"Missing required obs keys: {missing}. Available: {list(obs.keys())}")

    return {
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
    }


def save_dataset_hdf5(save_path, episodes):
    with h5py.File(save_path, "w") as f:
        data_grp = f.create_group("data")

        total = 0
        for ep_idx, ep in enumerate(episodes):
            demo_name = f"demo_{ep_idx}"
            demo_grp = data_grp.create_group(demo_name)
            obs_grp = demo_grp.create_group("obs")

            obs_grp.create_dataset("robot0_eef_pos", data=np.asarray(ep["robot0_eef_pos"], dtype=np.float32))
            obs_grp.create_dataset("robot0_eef_quat", data=np.asarray(ep["robot0_eef_quat"], dtype=np.float32))
            obs_grp.create_dataset("robot0_gripper_qpos", data=np.asarray(ep["robot0_gripper_qpos"], dtype=np.float32))
            obs_grp.create_dataset("object_pos", data=np.asarray(ep["object_pos"], dtype=np.float32))

            demo_grp.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float32))
            demo_grp.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
            demo_grp.create_dataset("dones", data=np.asarray(ep["dones"], dtype=np.bool_))
            demo_grp.create_dataset("phases", data=np.asarray(ep["phases"], dtype="S32"))
            demo_grp.create_dataset("initial_cube_pos", data=np.asarray(ep["initial_cube_pos"], dtype=np.float32))
            demo_grp.create_dataset("place_target_xy", data=np.asarray(ep["place_target_xy"], dtype=np.float32))

            demo_grp.attrs["success"] = bool(ep["success"])
            demo_grp.attrs["num_samples"] = int(len(ep["actions"]))
            demo_grp.attrs["robot"] = ROBOT
            demo_grp.attrs["task"] = "pick_place"

            total += len(ep["actions"])

        data_grp.attrs["total"] = int(total)

    print(f"\n[INFO] Saved dataset to: {save_path}")


def main():
    ensure_dir(SAVE_DIR)

    controller_config = make_controller_config()

    env = suite.make(
        env_name="Lift",
        robots=ROBOT,
        controller_configs=controller_config,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        reward_shaping=False,
        ignore_done=True,
    )

    env.robots[0].print_action_info()

    obj_body_id, obj_body_name = get_object_body_id(env)
    print(f"[INFO] Using object body: {obj_body_name}")

    episodes = []

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        time.sleep(0.1)
        settle_scene(env, steps=20)

        obj_pos = get_object_pos(env, obj_body_id)
        obj_xy = obj_pos[:2].copy()
        table_z = float(obj_pos[2])

        place_target_xy = sample_place_target_xy(
            obj_xy, PLACE_X_RANGE, PLACE_Y_RANGE, min_dist=MIN_PICK_PLACE_DIST
        )

        style = sample_episode_style()

        approach_above_z = style["approach_above_z"]
        grasp_z_offset = style["grasp_z_offset"]
        lift_height = style["lift_height"]
        place_above_z = style["place_above_z"]
        retreat_z = style["retreat_z"]
        side_offset = style["side_offset"]
        extra_arc_z = style["extra_arc_z"]
        speed_scale = style.get("speed_scale", 1.0)

        kp_pos = BASE_KP_POS * speed_scale
        max_dpos = BASE_MAX_DPOS * speed_scale

        obj_pos = get_object_pos(env, obj_body_id)

        pick_above = np.array([obj_pos[0] + side_offset[0], obj_pos[1] + side_offset[1], obj_pos[2] + approach_above_z + extra_arc_z], dtype=np.float32)
        pick_align = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + approach_above_z], dtype=np.float32)
        pick_grasp = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + grasp_z_offset], dtype=np.float32)
        lift_pose = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + lift_height + extra_arc_z], dtype=np.float32)

        move_mid = np.array([
            0.5 * (obj_pos[0] + place_target_xy[0]) + side_offset[0],
            0.5 * (obj_pos[1] + place_target_xy[1]) + side_offset[1],
            obj_pos[2] + lift_height + extra_arc_z,
        ], dtype=np.float32)

        place_above = np.array([place_target_xy[0], place_target_xy[1], obj_pos[2] + place_above_z + extra_arc_z], dtype=np.float32)
        place_drop = np.array([place_target_xy[0], place_target_xy[1], table_z + np.random.uniform(0.002, 0.006)], dtype=np.float32)
        retreat_pose = np.array([place_target_xy[0], place_target_xy[1], obj_pos[2] + retreat_z], dtype=np.float32)

        stage = "approach_pick"
        close_count = 0
        open_count = 0
        current_gripper_cmd = OPEN_ACTION

        ep_robot0_eef_pos = []
        ep_robot0_eef_quat = []
        ep_robot0_gripper_qpos = []
        ep_object_pos = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        ep_phases = []

        episode_success = False
        initial_cube_pos = obj_pos.astype(np.float32)

        for t in range(HORIZON):
            eef_pos = get_eef_pos(env)
            current_obj_pos = get_object_pos(env, obj_body_id)

            if stage == "approach_pick":
                target = pick_above
                current_gripper_cmd = OPEN_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if reached_xyz(eef_pos, pick_above, XY_TOL_OBJ, 0.035):
                    stage = "align_pick"

            elif stage == "align_pick":
                target = pick_align
                current_gripper_cmd = OPEN_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if reached_xyz(eef_pos, pick_align, XY_TOL_OBJ, 0.03):
                    stage = "descend_pick"

            elif stage == "descend_pick":
                target = pick_grasp
                current_gripper_cmd = OPEN_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if reached_xyz(eef_pos, pick_grasp, XY_TOL_OBJ, Z_TOL_DESCEND):
                    stage = "close_gripper"

            elif stage == "close_gripper":
                target = pick_grasp
                current_gripper_cmd = CLOSE_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                close_count += 1
                if close_count >= 25:
                    stage = "lift"

            elif stage == "lift":
                target = lift_pose
                current_gripper_cmd = CLOSE_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if abs(eef_pos[2] - lift_pose[2]) < 0.04:
                    stage = "move_mid"

            elif stage == "move_mid":
                target = move_mid
                current_gripper_cmd = CLOSE_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if reached_xyz(eef_pos, move_mid, 0.03, 0.04):
                    stage = "move_to_place"

            elif stage == "move_to_place":
                target = place_above
                current_gripper_cmd = CLOSE_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if reached_xyz(eef_pos, place_above, XY_TOL_PLACE, 0.04):
                    stage = "descend_place"

            elif stage == "descend_place":
                target = place_drop
                current_gripper_cmd = CLOSE_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if reached_xyz(eef_pos, place_drop, XY_TOL_PLACE, Z_TOL_DESCEND):
                    stage = "open_gripper"

            elif stage == "open_gripper":
                target = place_drop
                current_gripper_cmd = OPEN_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                open_count += 1
                obj_now = get_object_pos(env, obj_body_id)
                if open_count >= 30 and abs(obj_now[2] - table_z) < 0.01:
                    stage = "retreat"

            elif stage == "retreat":
                target = retreat_pose
                current_gripper_cmd = OPEN_ACTION
                action = build_action_towards_target(env, target, current_gripper_cmd, kp_pos, max_dpos)
                if abs(eef_pos[2] - retreat_pose[2]) < 0.04:
                    episode_success = True
                    break

            else:
                raise ValueError(f"Unknown stage: {stage}")

            curr_fields = extract_obs_fields(obs)
            ep_robot0_eef_pos.append(curr_fields["robot0_eef_pos"])
            ep_robot0_eef_quat.append(curr_fields["robot0_eef_quat"])
            ep_robot0_gripper_qpos.append(curr_fields["robot0_gripper_qpos"])
            ep_object_pos.append(current_obj_pos.astype(np.float32))
            ep_actions.append(action.copy())
            ep_rewards.append(0.0)
            ep_dones.append(False)
            ep_phases.append(stage)

            obs, reward, done, info = env.step(action)

            if RENDER and (t % PRINT_EVERY == 0):
                print(
                    f"step={t:03d} | stage={stage:15s} | "
                    f"eef={np.round(eef_pos, 4)} | obj={np.round(current_obj_pos, 4)}"
                )
                env.render()

        if len(ep_actions) == 0:
            continue

        ep_rewards[-1] = 1.0 if episode_success else 0.0
        ep_dones[-1] = True

        episodes.append({
            "robot0_eef_pos": np.asarray(ep_robot0_eef_pos, dtype=np.float32),
            "robot0_eef_quat": np.asarray(ep_robot0_eef_quat, dtype=np.float32),
            "robot0_gripper_qpos": np.asarray(ep_robot0_gripper_qpos, dtype=np.float32),
            "object_pos": np.asarray(ep_object_pos, dtype=np.float32),
            "actions": np.asarray(ep_actions, dtype=np.float32),
            "rewards": np.asarray(ep_rewards, dtype=np.float32),
            "dones": np.asarray(ep_dones, dtype=np.bool_),
            "phases": np.asarray(ep_phases, dtype="S32"),
            "initial_cube_pos": np.tile(initial_cube_pos[None, :], (len(ep_actions), 1)),
            "place_target_xy": np.tile(place_target_xy[None, :], (len(ep_actions), 1)),
            "success": episode_success,
        })

    env.close()
    save_dataset_hdf5(SAVE_PATH, episodes)


if __name__ == "__main__":
    main()
