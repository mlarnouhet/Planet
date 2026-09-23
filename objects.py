from typing import List
from pydantic import BaseModel, ConfigDict
import torch


class TensorModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class Transition(TensorModel):
    observation: torch.Tensor
    action: torch.Tensor
    reward: float | None


class TransitionBatch(TensorModel):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor | None


Trajectory = List[Transition]