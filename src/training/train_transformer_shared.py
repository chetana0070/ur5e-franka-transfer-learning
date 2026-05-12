import os
import sys
import json
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.transformer.transformer_policy import TransformerPolicy


DATASET_PATH = "data/final/pick_place_shared_cotrain_v1.hdf5"
OUTPUT_DIR = "checkpoints/transformer_single_step_14d"

OBS_HORIZON = 16
OBS_DIM = 14
ACTION_DIM = 7

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_obs_key(demo):
    keys = list(demo["obs"].keys())
    if "eef_pose_w_gripper" in keys:
        return "eef_pose_w_gripper"
    if "eef_pose_obj_goal_phase" in keys:
        return "eef_pose_obj_goal_phase"
    raise KeyError(f"No supported obs key found. Available obs keys: {keys}")


class HDF5SingleStepDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        obs_horizon=16,
        obs_mean=None,
        obs_std=None,
        action_mean=None,
        action_std=None,
    ):
        self.path = path
        self.split = split
        self.obs_horizon = obs_horizon

        self.obs_demos = []
        self.action_demos = []
        self.samples = []

        with h5py.File(path, "r") as f:
            if "mask" in f and split in f["mask"]:
                demo_keys = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in f["mask"][split][:]
                ]
            else:
                all_keys = sorted(list(f["data"].keys()))
                n = len(all_keys)
                cut = int(0.9 * n)
                demo_keys = all_keys[:cut] if split == "train" else all_keys[cut:]

            for demo_key in demo_keys:
                demo = f["data"][demo_key]
                obs_key = get_obs_key(demo)

                obs14 = demo["obs"][obs_key][:].astype(np.float32)
                obs8 = obs14[:, :OBS_DIM].astype(np.float32)
                actions = demo["actions"][:].astype(np.float32)

                T = actions.shape[0]
                max_start = T - obs_horizon - 1
                if max_start < 0:
                    continue

                demo_idx = len(self.obs_demos)
                self.obs_demos.append(obs8)
                self.action_demos.append(actions)

                for start in range(max_start + 1):
                    self.samples.append((demo_idx, start))

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

        print(f"{split} loaded demos:", len(self.obs_demos))
        print(f"{split} loaded samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        demo_idx, start = self.samples[idx]

        obs8 = self.obs_demos[demo_idx][start : start + self.obs_horizon].copy()
        action_idx = start + self.obs_horizon
        action = self.action_demos[demo_idx][action_idx].copy()

        if self.obs_mean is not None and self.obs_std is not None:
            obs8 = (obs8 - self.obs_mean) / self.obs_std

        if self.action_mean is not None and self.action_std is not None:
            action = (action - self.action_mean) / self.action_std

        return {
            "obs": torch.tensor(obs8, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
        }


def compute_stats(path, split="train", obs_horizon=16):
    obs_all = []
    action_all = []

    with h5py.File(path, "r") as f:
        if "mask" in f and split in f["mask"]:
            demo_keys = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["mask"][split][:]
            ]
        else:
            all_keys = sorted(list(f["data"].keys()))
            n = len(all_keys)
            cut = int(0.9 * n)
            demo_keys = all_keys[:cut] if split == "train" else all_keys[cut:]

        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            obs_key = get_obs_key(demo)

            obs14 = demo["obs"][obs_key][:].astype(np.float32)
            obs8 = obs14[:, :OBS_DIM]
            actions = demo["actions"][:].astype(np.float32)

            obs_all.append(obs8)
            action_all.append(actions)

    obs_all = np.concatenate(obs_all, axis=0)
    action_all = np.concatenate(action_all, axis=0)

    obs_mean = obs_all.mean(axis=0).astype(np.float32)
    obs_std = obs_all.std(axis=0).astype(np.float32)
    obs_std = np.maximum(obs_std, 1e-6)

    action_mean = action_all.mean(axis=0).astype(np.float32)
    action_std = action_all.std(axis=0).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)

    return obs_mean, obs_std, action_mean, action_std


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train(train)

    loss_fn = nn.MSELoss()
    losses = []

    for batch in loader:
        obs = batch["obs"].to(DEVICE)          # (B, 16, 8)
        target = batch["action"].to(DEVICE)   # (B, 7)

        pred = model(obs)                     # (B, 7)
        loss = loss_fn(pred, target)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def main():
    set_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("DEVICE:", DEVICE)
    print("DATASET:", DATASET_PATH)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    obs_mean, obs_std, action_mean, action_std = compute_stats(
        DATASET_PATH,
        split="train",
        obs_horizon=OBS_HORIZON,
    )

    print("obs_mean shape:", obs_mean.shape)
    print("obs_std shape:", obs_std.shape)
    print("action_mean shape:", action_mean.shape)
    print("action_std shape:", action_std.shape)

    print("obs_mean:", np.round(obs_mean, 4))
    print("obs_std:", np.round(obs_std, 4))
    print("action_mean:", np.round(action_mean, 4))
    print("action_std:", np.round(action_std, 4))

    train_ds = HDF5SingleStepDataset(
        DATASET_PATH,
        split="train",
        obs_horizon=OBS_HORIZON,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    valid_ds = HDF5SingleStepDataset(
        DATASET_PATH,
        split="valid",
        obs_horizon=OBS_HORIZON,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    print("train samples:", len(train_ds))
    print("valid samples:", len(valid_ds))

    sample = train_ds[0]
    print("sample obs shape:", sample["obs"].shape)
    print("sample action shape:", sample["action"].shape)

    assert sample["obs"].shape == (OBS_HORIZON, OBS_DIM)
    assert sample["action"].shape == (ACTION_DIM,)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = TransformerPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        seq_len=OBS_HORIZON,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ).to(DEVICE)

    test_x = torch.randn(2, OBS_HORIZON, OBS_DIM).to(DEVICE)
    test_y = model(test_x)
    print("model test input:", test_x.shape)
    print("model test output:", test_y.shape)
    assert test_y.shape == (2, ACTION_DIM)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "dataset_path": DATASET_PATH,
        "obs_horizon": OBS_HORIZON,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "model": {
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "seq_len": OBS_HORIZON,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    best_valid = float("inf")
    best_path = os.path.join(OUTPUT_DIR, "best_transformer_single_step_14d.pt")
    last_path = os.path.join(OUTPUT_DIR, "last_transformer_single_step_14d.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        valid_loss = run_epoch(model, valid_loader, optimizer=None)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"valid loss: {valid_loss:.6f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "config": config,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "epoch": epoch,
            "valid_loss": valid_loss,
        }

        torch.save(ckpt, last_path)

        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(ckpt, best_path)
            print("saved best:", best_path)

    print("finished training")
    print("best valid:", best_valid)
    print("best path:", best_path)


if __name__ == "__main__":
    main()
