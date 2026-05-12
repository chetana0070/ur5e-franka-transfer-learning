import h5py
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

DATASET = "data/final/pick_place_shared_cotrain_v1.hdf5"

print("=" * 80)
print("DATASET 14D OBS CHECK")

with h5py.File(DATASET, "r") as f:
    demos = sorted(list(f["data"].keys()))
    for demo_key in demos[:5]:
        demo = f["data"][demo_key]
        obs_key = list(demo["obs"].keys())[0]
        obs = demo["obs"][obs_key][:]
        actions = demo["actions"][:]
        print("-" * 80)
        print("demo:", demo_key)
        print("obs_key:", obs_key)
        print("obs shape:", obs.shape)
        print("first obs:", np.round(obs[0], 4))
        print("middle obs:", np.round(obs[len(obs)//2], 4))
        print("last obs:", np.round(obs[-1], 4))
        print("first action:", np.round(actions[0], 4))
        print("middle action:", np.round(actions[len(actions)//2], 4))
        print("last action:", np.round(actions[-1], 4))

        if "place_target_xy" in demo:
            print("place_target_xy:", np.round(demo["place_target_xy"][:], 4))
        if "initial_cube_pos" in demo:
            print("initial_cube_pos:", np.round(demo["initial_cube_pos"][:], 4))
        if "phases" in demo:
            phases = demo["phases"][:]
            print("phases shape:", phases.shape)
            print("phase unique first 20:", np.unique(phases[:20]))
            print("phase unique all:", np.unique(phases)[:20])

print("=" * 80)
print("ROBOSUITE RESET OBS KEYS")

controller_config = load_composite_controller_config(controller="BASIC")
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    ignore_done=True,
    control_freq=20,
    horizon=500,
)

obs = env.reset()
print("obs keys:")
for k in sorted(obs.keys()):
    v = obs[k]
    if hasattr(v, "shape"):
        print(k, v.shape, np.round(v, 4))
    else:
        print(k, type(v), v)

print("=" * 80)
print("ENV OBJECTS / ATTRS")
for name in [
    "cube",
    "cube_body_id",
    "cube_body",
    "cube_body_name",
    "table_offset",
    "reward_scale",
]:
    if hasattr(env, name):
        print(name, "=", getattr(env, name))

env.close()
