import os
import logging
from logging import Logger
from datetime import datetime
from pathlib import Path
from argparse import Namespace
from typing import Dict, List
import shutil
import random
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_dirs(args: Namespace):
    run_dirs = [
        Path(f"checkpoints/run_{args.run_id}"),
        Path(f"samples/run_{args.run_id}"),
    ]

    for run_dir in run_dirs:
        if not args.resume and run_dir.exists():
            shutil.rmtree(run_dir)

        run_dir.mkdir(parents=True, exist_ok=True)

def setup_logs(args: Namespace) -> Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (f"train_{args.run_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

    return logging.getLogger(__name__)

def setup_wnb(args: Namespace, wandb_run_id: str | None = None) -> str:
    if args.resume:
        run = wandb.init(
        project=f"Planet",
        entity=os.getenv("WANDB_ENTITY"),
        name=f"run__{args.domain_name}-{args.task_name}_{args.run_id}",
        id=wandb_run_id,
        config=vars(args),
        resume="must",
    )
    else:
        run = wandb.init(
        project=f"Planet",
        name=f"run__{args.domain_name}-{args.task_name}_{args.run_id}",
        config=vars(args),
    )
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    return run.id


def compute_loss(args: Namespace, models: Dict[str, nn.Module], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    encoder = models["encoder"]
    det_state_model = models["det_state_model"]
    stoch_state_model = models["stoch_state_model"]
    obs_model = models["obs_model"]
    reward_model = models["reward_model"]
    mse_loss = nn.MSELoss()

    batch_obs_loss = 0.0
    batch_reward_loss = 0.0
    batch_kl_loss = 0.0
    batch_loss = 0.0

    for k in range(args.train_seq_len):
        observation = batch["observation"][:, k, :, :, :]
        if k == 0:
            prev_action = torch.zeros((args.batch_size, args.action_dim), device="cuda")
            s = torch.zeros((args.batch_size, args.latent_dim), device="cuda")
            h = torch.zeros((args.batch_size, args.hidden_dim), device="cuda")
        else:
            prev_action = batch["action"][:, k-1, :]
            reward = batch["reward"][:, k-1]
        
        h = det_state_model(s, prev_action, h)
        mu_e, sigma_e = encoder(observation, h)
        encoder_pred_state = mu_e + torch.randn_like(sigma_e) * sigma_e
        mu_s, sigma_s = stoch_state_model(h)
        pred_reward = reward_model(h, encoder_pred_state)
        pred_obs = obs_model(h, encoder_pred_state)
        s = encoder_pred_state

        if k > 0:
            reward_loss = mse_loss(pred_reward.squeeze(1), reward)
        else:
            reward_loss = torch.tensor(0.0, device="cuda")
        obs_loss = mse_loss(pred_obs, observation)
        kl_per_dimension = torch.log(sigma_s) - torch.log(sigma_e) + 1/2 * (sigma_e**2 + (mu_e - mu_s)**2) / sigma_s**2 - 1/2
        kl = kl_per_dimension.sum(dim=-1)
        kl_loss = kl.clamp_min(3.0).mean()

        batch_obs_loss += obs_loss
        batch_reward_loss += reward_loss
        batch_kl_loss += kl_loss
        batch_loss += args.reward_scale*reward_loss + obs_loss + kl_loss

    return {
        "obs_loss": batch_obs_loss / args.train_seq_len,
        "reward_loss": batch_reward_loss / args.train_seq_len,
        "kl_loss": batch_kl_loss / args.train_seq_len,
        "loss": batch_loss / args.train_seq_len
        }


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    obs = np.ascontiguousarray(obs)
    obs = torch.from_numpy(obs).permute(2, 0, 1)
    obs = torch.nn.functional.interpolate(
        obs.unsqueeze(0).float(),
        size=(64, 64),
        mode="bilinear",
    ).squeeze(0)

    obs = torch.floor(obs / 8.0)
    obs = obs / 32.0
    obs = obs + torch.rand_like(obs) / 32.0
    obs = obs - 0.5

    return obs
