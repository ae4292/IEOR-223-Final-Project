import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader, TensorDataset

import math
from typing import List


# import the normalized slices
slices = pd.read_csv('data/slices_normalized.csv', index_col=0)

# Convert the DataFrame to a numpy array, then to a PyTorch Tensor
slices_tensor = torch.tensor(slices.values)

# Convert your data to a TensorDataset
slices_dataset = TensorDataset(slices_tensor)

# DEFINES THE BATCH SIZE
batch_size = 32

# Create a DataLoader with shuffle=True for shuffling at each epoch
train_loader = DataLoader(slices_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

N = 100

def kerras_boundaries(sigma, eps, N, T):
    # This will be used to generate the boundaries for the time discretization

    return torch.tensor(
        [
            (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
            ** sigma
            for i in range(N)
        ]
    )

blk = lambda ic, oc: nn.Sequential(
    nn.GroupNorm(32, num_channels=ic),
    nn.SiLU(),
    nn.Conv2d(ic, oc, 3, padding=1),
    nn.GroupNorm(32, num_channels=oc),
    nn.SiLU(),
    nn.Conv2d(oc, oc, 3, padding=1),
)


class ConsistencyModel(nn.Module):
    """
    This is ridiculous Unet structure, hey but it works!
    """

    def __init__(self, n_channel: int, eps: float = 0.002, D: int = 128) -> None:
        super(ConsistencyModel, self).__init__()

        self.eps = eps

        self.freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=D, dtype=torch.float32) / D
        )

        self.down = nn.Sequential(
            *[
                nn.Conv2d(n_channel, D, 3, padding=1),
                blk(D, D),
                blk(D, 2 * D),
                blk(2 * D, 2 * D),
            ]
        )

        self.time_downs = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, 2 * D),
            nn.Linear(2 * D, 2 * D),
        )

        self.mid = blk(2 * D, 2 * D)

        self.up = nn.Sequential(
            *[
                blk(2 * D, 2 * D),
                blk(2 * 2 * D, D),
                blk(D, D),
                nn.Conv2d(2 * D, 2 * D, 3, padding=1),
            ]
        )
        self.last = nn.Conv2d(2 * D + n_channel, n_channel, 3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        if isinstance(t, float):
            t = (
                torch.tensor([t] * x.shape[0], dtype=torch.float32)
                .to(x.device)
                .unsqueeze(1)
            )
        # time embedding
        args = t.float() * self.freqs[None].to(t.device)
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(x.device)

        x_ori = x

        # perform F(x, t)
        hs = []
        for idx, layer in enumerate(self.down):
            if idx % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
                x = F.interpolate(x, scale_factor=0.5)
                hs.append(x)

            x = x + self.time_downs[idx](t_emb)[:, :, None, None]

        x = self.mid(x)

        for idx, layer in enumerate(self.up):
            if idx % 2 == 0:
                x = layer(x) + x
            else:
                x = torch.cat([x, hs.pop()], dim=1)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = layer(x)

        x = self.last(torch.cat([x, x_ori], dim=1))

        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).pow(0.5)

        return c_skip_t[:, :, None, None] * x_ori + c_out_t[:, :, None, None] * x

    def loss(self, x, z, t1, t2, ema_model):
        x2 = x + z * t2[:, :, None, None]
        x2 = self(x2, t2)

        with torch.no_grad():
            x1 = x + z * t1[:, :, None, None]
            x1 = ema_model(x1, t1)

        return F.mse_loss(x1, x2)

    @torch.no_grad()
    def sample(self, x, ts: List[float]):
        x = self(x, ts[0])

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t**2 - self.eps**2) * z
            x = self(x, t)

        return x
    

def train(n_epoch: int = 100, device="cuda:0", dataloader=train_loader, n_channels=1, name="ts"):
    model = ConsistencyModel(n_channels, D=256)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    ema_model = ConsistencyModel(n_channels, D=256)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())
    
    # Initialize the CSV file
    with open('losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Mu"])
    
    print("Starting Training ...")

    for epoch in range(1, n_epoch):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epoch) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x in pbar: 
            x = torch.tensor(x[0])
            optim.zero_grad()
            x = x.to(device)

            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            loss = model.loss(x, z, t_0, t_1, ema_model=ema_model)

            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")
            
            # Write the progress to the CSV file
            # if epoch % 10 == 0:
            with open('losses.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, loss_ema, mu])
                print(f"Loss: {loss_ema:.10f}, Mu: {mu:.10f}",flush=True)

train(dataloader=train_loader, n_channels=3, name="sp500")