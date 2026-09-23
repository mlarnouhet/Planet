from argparse import Namespace
from typing import List, Dict
import math
import random
import torch

class PlaNetDataset:
    def __init__(self, args: Namespace):
        self.trajectories: List[Dict[str, torch.Tensor]] = []
        self.seq_len = args.train_seq_len
        self.T = args.T
        self.batch_size = args.batch_size
        self.action_dim = args.action_dim
        self.n_action_repeat = args.n_action_repeat

    def draw_batch(self) -> Dict[str, torch.Tensor]:
        batch = []
        for _ in range(self.batch_size):
            i = random.randint(0, len(self.trajectories)-1)
            j = random.randint(0, math.ceil((self.T/self.n_action_repeat)) - self.seq_len)
            batch.append({k: v[j:j+self.seq_len] for (k,v) in self.trajectories[i].items()})
        return {k: torch.stack([traj[k] for traj in batch]) for k in batch[0].keys()}

    def add(self, trajectory: Dict[str, torch.Tensor]):
        self.trajectories.append(trajectory)

            
                
