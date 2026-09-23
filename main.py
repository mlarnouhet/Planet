import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
from argparse import Namespace
import math
import numpy as np
from tqdm import tqdm
import torch
from huggingface_hub import HfApi
import wandb
from dm_control import suite
from dm_control.suite.wrappers import pixels
from utils import setup_logs, setup_wnb, setup_dirs, compute_loss, preprocess_obs, set_seed
from cem import plan_action
from dataset import PlaNetDataset
from models import get_models

def main(args: Namespace):

    logger = setup_logs(args)
    dataset = PlaNetDataset(args)
    models = get_models(args)
    for model in models.values():
        model.cuda()
    parameters = [
    param
    for model in list(models.values())
    for param in model.parameters()
    ]
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    metrics_list = []

    env = suite.load(domain_name=args.domain_name, task_name=args.task_name, visualize_reward=True)
    env = pixels.Wrapper(env)
    spec = env.action_spec()

    if args.resume:
        checkpoint_dir = f"checkpoints/run_{args.run_id}/checkpoint_{args.run_id}_{args.domain_name}_{args.task_name}_step_{args.step_to_load}.pt"
        logger.info(f"Loading checkpoint {checkpoint_dir}")
        checkpoint = torch.load(checkpoint_dir)
        models["encoder"].load_state_dict(checkpoint["encoder"])
        models["det_state_model"].load_state_dict(checkpoint["det_state_model"])
        models["stoch_state_model"].load_state_dict(checkpoint["stoch_state_model"])  
        models["obs_model"].load_state_dict(checkpoint["obs_model"])
        models["reward_model"].load_state_dict(checkpoint["reward_model"]) 
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        dataset.trajectories = checkpoint["trajectories"]
        metrics_list = checkpoint["metrics_list"]
        start_step = checkpoint["step"]+1
        logger.info(f"Resuming run {args.run_id} on {args.domain_name}-{args.task_name} at step {start_step}")
        if args.setup_wandb:
            wandb_run_id = checkpoint["wandb_run_id"]
            _ = setup_wnb(args, wandb_run_id)
    else:
        if args.setup_wandb:
            wandb_run_id = setup_wnb(args)

        start_step = 0 
        if args.debug:
            import pickle
            with open("debug/dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
        else:
            for _ in range(args.n_random_seeds):
                trajectory = {
                    "observation": [],
                    "action": [],
                    "reward": []
                }
                time_step = env.reset()
                for _ in range(math.ceil(args.T/args.n_action_repeat)):
                    obs = preprocess_obs(time_step.observation["pixels"])
                    action = np.random.uniform(spec.minimum, spec.maximum, spec.shape).astype(np.float32)

                    reward = 0.0
                    for _ in range(args.n_action_repeat):
                        time_step = env.step(action)
                        reward += time_step.reward.astype(np.float32) if time_step.reward is not None else 0.0

                    trajectory["observation"].append(obs)
                    trajectory["action"].append(torch.tensor(action))
                    trajectory["reward"].append(torch.tensor(reward))

                dataset.add({k: torch.stack(v) for (k,v) in trajectory.items()})

    for model in models.values():
        model.train()

    for step in tqdm(range(start_step, args.n_steps), desc="Training"):

        metrics_dict = {
            "obs_loss": 0.0,
            "reward_loss": 0.0,
            "kl_loss": 0.0,
            "total_loss": 0.0,
            "encoder_grad_norm": 0.0,
            "det_state_model_grad_norm": 0.0,
            "stoch_state_model_grad_norm": 0.0,
            "obs_model_grad_norm": 0.0,
            "reward_model_grad_norm": 0.0,
        }

        for _ in tqdm(range(args.n_update_steps), desc="Running update steps"):
            batch = dataset.draw_batch()
            batch = {k: v.cuda() for (k,v) in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                temp_metrics_dict = compute_loss(args, models, batch)

            loss = temp_metrics_dict["total_loss"]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            for (k, v) in temp_metrics_dict.items():
                metrics_dict[k] += v
            for (model_name, model) in models.items():
                metrics_dict[model_name + "_grad_norm"] += torch.nn.utils.clip_grad_norm_([param for param in model.parameters()], max_norm=1000.0)
            
            scaler.step(optimizer)
            scaler.update()

        metrics_dict = {k: v.item() / args.n_update_steps for (k, v) in metrics_dict.items()}

        if step % args.log_every == 0:
            logger.info(f"Step {step} metrics:")
            reward = np.mean([dataset.trajectories[-i]['reward'].sum() for i in range(5)])
            for key, value in metrics_dict.items():
                logger.info(f"{key}: {value}")
            logger.info(f"Rewards: {reward}")
            if args.setup_wandb:
                wandb.log({"step": step, **metrics_dict})
                wandb.log({"step": step, "reward": reward})
        
        metrics_list.append(metrics_dict)

        with torch.no_grad():
            for model in models.values():
                model.eval()

            time_step = env.reset()
            obs = preprocess_obs(time_step.observation["pixels"])
            h = torch.zeros((1, args.hidden_dim)).cuda()
            trajectory = {
                "observation": [],
                "action": [],
                "reward": []
            }
            for _ in tqdm(range(math.ceil(args.T/args.n_action_repeat)), desc="Sampling"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mu_s, sigma_s = models["encoder"](obs.unsqueeze(0).cuda(), h)
                    s = mu_s + torch.randn_like(sigma_s) * sigma_s
                    action = plan_action(args, models, s, h)
                    action += 0.3 * torch.randn_like(action)

                reward = 0.0
                for _ in range(args.n_action_repeat):
                    time_step = env.step(action.cpu().numpy())
                    reward += (time_step.reward.astype(np.float32) if time_step.reward is not None else 0.0)

                trajectory["observation"].append(obs)
                trajectory["action"].append(action.cpu())
                trajectory["reward"].append(torch.tensor(reward))

                h = models["det_state_model"](s, action.unsqueeze(0), h)
                obs = preprocess_obs(time_step.observation["pixels"])

            dataset.add({k: torch.stack(v) for (k,v) in trajectory.items()})

        if (step % args.save_interval == 0) and (step > start_step):
            checkpoint_dir = f"checkpoints/run_{args.run_id}/checkpoint_{args.run_id}_{args.domain_name}_{args.task_name}_step_{step}.pt"
            checkpoint = {
                "encoder": models["encoder"].state_dict(),
                "det_state_model": models["det_state_model"].state_dict(),
                "stoch_state_model": models["stoch_state_model"].state_dict(),   
                "obs_model": models["obs_model"].state_dict(),
                "reward_model": models["reward_model"].state_dict(),       
                "optimizer_state_dict": optimizer.state_dict(),
                "trajectories": dataset.trajectories,
                "metrics_list": metrics_list,
                "step": step,
                "wandb_run_id": wandb_run_id if args.setup_wandb else "" 
            }
            torch.save(checkpoint, checkpoint_dir)

            api = HfApi()
            api.create_repo(repo_id=args.hf_repo_id, private=True, exist_ok=True)
            upload_future_model = api.upload_file(
                repo_id=args.hf_repo_id,
                path_or_fileobj=checkpoint_dir,
                path_in_repo=f"run_{args.run_id}_{args.domain_name}_{args.task_name}/step{step}/models.pt",
                commit_message=f"Checkpoint: run {args.domain_name}_{args.task_name}_{args.run_id}, step {step}",
                run_as_future=True,
            )
            upload_future_model.result()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain_name", type=str, default="cartpole")
    parser.add_argument("--task_name", type=str, default="swingup")
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--n_update_steps", type=int, default=100) #100
    parser.add_argument("--n_action_repeat", type=int, default=8)
    parser.add_argument("--train_seq_len", type=int, default=50) #50
    parser.add_argument("--T", type=int, default=1000) #1000
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_optimization_steps", type=int, default=10)
    parser.add_argument("--n_candidate_samples", type=int, default=1000)
    parser.add_argument("--action_dim", type=int, default=1)
    parser.add_argument("--horizon_len", type=int, default=12)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--n_random_seeds", type=int, default=5)
    parser.add_argument("--setup_wandb", type=bool, default=True)
    parser.add_argument("--run_id", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--debug", type=int, default=False)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--step_to_load", type=int, default=0)
    parser.add_argument("--hf_repo_id", type=str, default="Marcorico/planet")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_dirs(args)
    main(args)
