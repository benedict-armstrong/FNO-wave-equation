import numpy as np
import torch
from torch.utils.data import Dataset


class PDEDataset(Dataset):
    def __init__(
        self,
        path: str,
    ):
        self.data = torch.from_numpy(np.load(path)).type(torch.float32)
        self.length = self.data.shape[0]

        self.sample_resolution = self.data.shape[-1]
        self.sample_timesteps = self.data.shape[-2]

        self.x_values = torch.tensor(
            np.linspace(0, 1, self.sample_resolution), dtype=torch.float32
        )
        self.t_values = torch.tensor(
            np.linspace(0, 1, self.sample_timesteps), dtype=torch.float32
        )

        # for each sample add the x and t values
        self.data = self.data.unsqueeze(-1)

        self.x_values = self.x_values.expand(
            self.length, self.sample_timesteps, -1
        ).unsqueeze(-1)

        self.data = torch.cat([self.data, self.x_values], dim=-1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]
