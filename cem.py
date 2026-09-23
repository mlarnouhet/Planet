from argparse import Namespace
from typing import Dict
import torch
import torch.nn as nn
from dm_env import TimeStep
from torch.distributions import Independent, Normal

def evaluate(args: Namespace, actions: torch.Tensor, models: Dict[str, nn.Module], s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    total_reward = torch.zeros((args.n_candidate_samples, 1), device="cuda")
    det_state_model = models["det_state_model"]
    stoch_state_model = models["stoch_state_model"]
    reward_model = models["reward_model"]

    h = h.expand(args.n_candidate_samples, -1)
    s = s.expand(args.n_candidate_samples, -1)

    for i in range(args.horizon_len):
        h = det_state_model(s, actions[:, i*args.action_dim:(i+1)*args.action_dim] , h)
        mu_s, sigma_s = stoch_state_model(h)
        s = mu_s + torch.randn_like(sigma_s) * sigma_s
        total_reward += reward_model(h, s)

    return total_reward


def plan_action(args: Namespace, models: list[nn.Module], s: torch.Tensor, h: torch.Tensor):
    mu_q = torch.zeros(args.action_dim * args.horizon_len, device="cuda")
    sigma_q = torch.ones(args.action_dim * args.horizon_len, device="cuda")

    for _ in range(args.n_optimization_steps):
        exp_mu_q = mu_q.unsqueeze(0).expand(args.n_candidate_samples, -1)
        exp_sigma_q = sigma_q.unsqueeze(0).expand(args.n_candidate_samples, -1)
        actions = exp_mu_q + torch.randn_like(exp_sigma_q) * exp_sigma_q
        rewards = evaluate(args, actions, models, s, h)
        actions = torch.unbind(actions, dim=0)
        rewards = rewards.cpu().numpy()
        order = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
        actions = [actions[i] for i in order]
        top_k_actions = torch.stack(actions[:100])
        mu_q = torch.mean(top_k_actions, dim=0) 
        sigma_q = torch.abs(top_k_actions - mu_q.unsqueeze(0).expand(100, -1)).sum(dim=0) / 99

    return mu_q[:args.action_dim]