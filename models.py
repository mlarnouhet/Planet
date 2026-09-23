from argparse import Namespace
from typing import Dict
import torch
import torch.nn as nn

class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=1
            )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x) -> torch.Tensor:
        return nn.functional.relu(self.bn(self.conv(x))) 

class ConvModel(nn.Module):
    def __init__(self, latent_dim: int, n_channels: int = 32):
        super().__init__()
        self.conv1 = ConvUnit(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1)
        self.conv2 = ConvUnit(in_channels=n_channels, out_channels=n_channels*2, kernel_size=3, stride=2)
        self.conv3 = ConvUnit(in_channels=n_channels*2, out_channels=n_channels*4, kernel_size=3, stride=2)
        self.conv4 = ConvUnit(in_channels=n_channels*4, out_channels=n_channels*8, kernel_size=3, stride=2)
        self.lin_out = nn.Linear(8*8*256, out_features=latent_dim, bias=True)

    def forward(self, x) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.lin_out(torch.flatten(x4, start_dim=1))
        return out

class EncoderModel(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.conv_model = ConvModel(latent_dim)
        self.lin = nn.Linear(in_features=hidden_dim+latent_dim, out_features=hidden_dim+latent_dim, bias=True)
        self.mu_lin = nn.Linear(in_features=hidden_dim+latent_dim, out_features=latent_dim, bias=True)
        self.sigma_lin = nn.Linear(in_features=hidden_dim+latent_dim, out_features=latent_dim, bias=True)

    def forward(self, observation, hidden) -> tuple[torch.Tensor, torch.Tensor]:
        obs_latent = self.conv_model(observation)
        cat_latents = torch.cat([obs_latent, hidden], dim=-1)
        z = nn.functional.relu(self.lin(cat_latents))
        mu = self.mu_lin(z)
        sigma = nn.functional.softplus(self.sigma_lin(z)) + 0.01
        return mu, sigma

class DeterministicStateModel(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, action_dim: int):
        super().__init__()

        self.gru = nn.GRUCell(
            input_size=latent_dim+action_dim,
            hidden_size=hidden_dim
        )

    def forward(self, s, a, h):
        input = torch.cat([a, s], dim=-1)
        hidden = self.gru(input, h)
        return hidden

class StochasticStateModel(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.mu_lin = nn.Linear(in_features=hidden_dim, out_features=latent_dim, bias=True)
        self.sig_lin = nn.Linear(in_features=hidden_dim, out_features=latent_dim, bias=True)

    def forward(self, h) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = nn.functional.relu(self.lin1(h))
        z2 = nn.functional.relu(self.lin2(z1))
        mu = self.mu_lin(z2)
        sigma = nn.functional.softplus(self.sig_lin(z2)) + 0.01
        return mu, sigma

class DeconvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=1,
            output_padding=1 
            )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x) -> torch.Tensor:
        return nn.functional.relu(self.bn(self.deconv(x)))

class ObservationModel(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, n_channels: int = 32):
        super().__init__()
        self.lin_in = nn.Linear(in_features=hidden_dim+latent_dim, out_features=8*8*256, bias=True)
        self.deconv1 = DeconvUnit(in_channels=n_channels*8, out_channels=n_channels*4, kernel_size=3, stride=2)
        self.deconv2 = DeconvUnit(in_channels=n_channels*4, out_channels=n_channels*2, kernel_size=3, stride=2)
        self.deconv3 = DeconvUnit(in_channels=n_channels*2, out_channels=n_channels*1, kernel_size=3, stride=2)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, h, s) -> torch.Tensor:
        cat_latent = torch.cat([h, s], dim=-1)
        z = self.lin_in(cat_latent).view(-1, 256, 8, 8)
        z1 = self.deconv1(z)
        z2 = self.deconv2(z1)
        z3 = self.deconv3(z2)
        mu = self.conv_out(z3)
        return mu

class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.lin1=  nn.Linear(in_features=hidden_dim+latent_dim, out_features=hidden_dim+latent_dim, bias=True)
        self.lin2=  nn.Linear(in_features=hidden_dim+latent_dim, out_features=hidden_dim+latent_dim, bias=True)
        self.lin3 = nn.Linear(in_features=hidden_dim+latent_dim, out_features=1, bias=True)

    def forward(self, h, s) -> torch.Tensor:
        cat_latent = torch.cat([h, s], dim=-1)
        z1 = nn.functional.relu(self.lin1(cat_latent))
        z2 = nn.functional.relu(self.lin2(z1))
        mu = self.lin3(z2)
        return mu

def get_models(args: Namespace) -> Dict[str, nn.Module]:
    return {
        "encoder": EncoderModel(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim),
        "det_state_model": DeterministicStateModel(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, action_dim=args.action_dim),
        "stoch_state_model": StochasticStateModel(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim),
        "obs_model": ObservationModel(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim),
        "reward_model": RewardModel(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    }

if __name__ == "__main__":
    pass