"""
Diffusion Policy - debugged bring-up version for robot-only low-dim training.
EMA disabled for inference compatibility.
"""

from typing import Callable
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    return DiffusionPolicyUNet, {}


class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        obs_encoder = replace_bn_with_gn(obs_encoder)
        obs_encoder = obs_encoder.float().to(self.device)

        # Detect the true post-encoder observation dimension
        dummy_obs = OrderedDict()
        for k, shape in self.obs_shapes.items():
            dummy_obs[k] = torch.zeros(1, *shape, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dummy_out = obs_encoder(obs=dummy_obs)

        if dummy_out.ndim != 2:
            dummy_out = dummy_out.reshape(dummy_out.shape[0], -1)

        self.obs_encoder_out_dim = int(dummy_out.shape[-1])
        self.global_cond_dim = self.obs_encoder_out_dim * self.algo_config.horizon.observation_horizon

        print(f"[FINAL] encoder dim = {self.obs_encoder_out_dim}")

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.global_cond_dim,
        )

        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net,
            })
        }).float().to(self.device)

        if self.algo_config.ddpm.enabled:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type,
            )
        elif self.algo_config.ddim.enabled:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type,
            )
        else:
            raise RuntimeError("Either DDPM or DDIM must be enabled.")

        # EMA is kept for training-time step() compatibility, but not used in inference.
        self.ema = EMAModel(model=nets, power=self.algo_config.ema.power) \
            if self.algo_config.ema.enabled else None

        self.nets = nets

        # Kept only for compatibility / debugging
        self._shadow_nets = copy.deepcopy(nets).eval()
        self._shadow_nets.requires_grad_(False)

        self.obs_queue = None
        self.alpha = 1.0

    def _sanitize_obs(self, obs_dict):
        """
        Ensure all low-dim observations are [B, T, D].
        """
        out = OrderedDict()
        for k in self.obs_shapes:
            if k not in obs_dict:
                raise KeyError(f"Missing observation key '{k}'")

            v = obs_dict[k]
            if v.ndim < 3:
                raise ValueError(f"Obs {k} must be [B, T, D], got {tuple(v.shape)}")

            v = v.reshape(v.shape[0], v.shape[1], -1)
            out[k] = v
        return out

    def _encode_obs(self, obs_dict, nets=None):
        if nets is None:
            nets = self.nets

        obs_dict = self._sanitize_obs(obs_dict)

        first = next(iter(obs_dict.values()))
        B, T = first.shape[:2]

        flat = OrderedDict()
        for k, v in obs_dict.items():
            flat[k] = v.reshape(B * T, -1).to(self.device).float()

        feat = nets["policy"]["obs_encoder"](obs=flat)

        if feat.ndim != 2:
            feat = feat.reshape(B * T, -1)

        feat = feat.to(self.device).float()
        return feat.reshape(B, T, -1)

    def process_batch_for_training(self, batch):
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        obs = {
            k: batch["obs"][k][:, :To, :].to(self.device).float()
            for k in batch["obs"]
        }
        actions = batch["actions"][:, :Tp, :].to(self.device).float()

        return {
            "obs": obs,
            "actions": actions,
        }

    def train_on_batch(self, batch, epoch, validate=False):
        To = self.algo_config.horizon.observation_horizon
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            obs_feat = self._encode_obs(batch["obs"])[:, :To]
            obs_cond = obs_feat.reshape(B, -1).to(self.device).float()

            expected = self.global_cond_dim
            if obs_cond.shape[1] != expected:
                raise ValueError(
                    f"obs_cond shape mismatch: got {tuple(obs_cond.shape)}, expected second dim {expected}"
                )

            actions = batch["actions"].to(self.device).float()

            noise = torch.randn_like(actions, device=self.device)
            t = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=self.device,
                dtype=torch.long,
            )

            noisy = self.noise_scheduler.add_noise(actions, noise, t).to(self.device).float()
            pred = self.nets["policy"]["noise_pred_net"](noisy, t, global_cond=obs_cond)

            loss = F.mse_loss(pred, noise)

        if not validate:
            TorchUtils.backprop_for_loss(self.nets, self.optimizers["policy"], loss)

            if self.ema is not None:
                # Training-time EMA update kept as-is
                self.ema.step(self.nets)

        return {"losses": {"l2_loss": loss.detach()}}

    def log_info(self, info):
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        return log

    def reset(self):
        To = self.algo_config.horizon.observation_horizon
        self.obs_queue = deque(maxlen=To)

    def get_action(self, obs_dict):
        To = self.algo_config.horizon.observation_horizon

        if self.obs_queue is None:
            self.reset()

        if len(self.obs_queue) == 0:
            for _ in range(To):
                self.obs_queue.append(obs_dict)
        else:
            self.obs_queue.append(obs_dict)

        obs_seq = {}
        for k in obs_dict.keys():
            elems = [o[k] for o in self.obs_queue]

            if elems[0].ndim == 1:
                # each obs: [D] -> [1, T, D]
                obs_seq[k] = torch.stack(elems, dim=0).unsqueeze(0)

            elif elems[0].ndim == 2:
                # support either [1, D] or [T, D]
                if elems[0].shape[0] == 1:
                    squeezed = [e.squeeze(0) for e in elems]
                    obs_seq[k] = torch.stack(squeezed, dim=0).unsqueeze(0)
                elif elems[0].shape[0] == To:
                    obs_seq[k] = elems[-1].unsqueeze(0) if elems[-1].ndim == 2 else elems[-1]
                else:
                    raise ValueError(
                        f"Unexpected rollout obs shape for key {k}: {tuple(elems[0].shape)}"
                    )

            elif elems[0].ndim == 3:
                obs_seq[k] = elems[-1]

            else:
                raise ValueError(
                    f"Unexpected rollout obs shape for key {k}: {tuple(elems[0].shape)}"
                )

        return self._sample_action(obs_seq)[0]

    def _sample_action(self, obs_dict):
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        # EMA disabled completely for inference compatibility
        nets = self.nets

        obs_seq = {}
        for k, v in obs_dict.items():
            if v.ndim == 2:
                v = v.unsqueeze(0)
            obs_seq[k] = v.to(self.device).float()

        obs_feat = self._encode_obs(obs_seq, nets=nets)[:, :To]
        obs_cond = obs_feat.reshape(obs_feat.shape[0], -1).to(self.device).float()

        x = torch.randn((obs_feat.shape[0], Tp, self.ac_dim), device=self.device)

        if self.algo_config.ddpm.enabled:
            self.noise_scheduler.set_timesteps(self.algo_config.ddpm.num_inference_timesteps)
        else:
            self.noise_scheduler.set_timesteps(self.algo_config.ddim.num_inference_timesteps)

        for t in self.noise_scheduler.timesteps:
            if not torch.is_tensor(t):
                t_input = torch.full(
                    (x.shape[0],),
                    int(t),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                if t.ndim == 0:
                    t_input = t.to(self.device).long().expand(x.shape[0])
                else:
                    t_input = t.to(self.device).long()

            pred = nets["policy"]["noise_pred_net"](x, t_input, global_cond=obs_cond)

            step_out = self.noise_scheduler.step(pred, t, x)

            if hasattr(step_out, "prev_sample"):
                x = step_out.prev_sample
            elif isinstance(step_out, tuple):
                x = step_out[0]
            else:
                x = step_out

        return x[:, To - 1: To - 1 + Ta]

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": None,
        }

    def deserialize(self, model_dict):
        self.nets.load_state_dict(model_dict["nets"])


def replace_submodules(root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(module):
    return replace_submodules(
        root_module=module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // 16),
            num_channels=x.num_features,
        ),
    )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.to(next(self.parameters(), torch.empty(0, device=x.device)).device if len(list(self.parameters())) else x.device)
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=x.device) * -math.log(10000) / (half - 1))
        emb = x[:, None] * emb[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(global_cond_dim + 256, 512),
            nn.Mish(),
            nn.Linear(512, input_dim),
        )
        self.time = SinusoidalPosEmb(256)

    def forward(self, sample, timestep, global_cond):
        device = sample.device

        sample = sample.to(device).float()
        global_cond = global_cond.to(device).float()

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        else:
            timestep = timestep.to(device)

        if timestep.ndim == 0:
            timestep = timestep[None]

        timestep = timestep.expand(sample.shape[0])

        t = self.time(timestep).to(device).float()
        cond = torch.cat([t, global_cond], dim=-1).to(device).float()

        out = self.fc(cond)
        return out.unsqueeze(1).repeat(1, sample.shape[1], 1)
