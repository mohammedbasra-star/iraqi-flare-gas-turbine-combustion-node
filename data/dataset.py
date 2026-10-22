"""
PyTorch Dataset and DataLoader for Sour APG Kinetics.
"""

from typing import Tuple, Dict
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class APGCombustionDataset(Dataset):
    """
    Dataset wrapping (conditions, trajectories) for Stiff Neural ODE training.
    Conditions: (N, 3) -> [Pressure, Phi, H2S_pct]
    Trajectories: (N, T_steps, 9) -> [T, CH4, O2, CO2, H2O, H2S, SO2, SO3, NO]
    """

    def __init__(self, data_dir: str = "data") -> None:
        traj_path = os.path.join(data_dir, "trajectories.pt")
        cond_path = os.path.join(data_dir, "conditions.pt")
        time_path = os.path.join(data_dir, "time.pt")

        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Run data/generate_apg_data.py first.")

        self.trajectories = torch.load(traj_path, weights_only=True)
        self.conditions = torch.load(cond_path, weights_only=True)
        self.time = torch.load(time_path, weights_only=True)

        # Scale factor for temperature ~ 2200 K
        self.T_scale = 2200.0

    def __len__(self) -> int:
        return self.trajectories.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conditions[idx], self.trajectories[idx]


def get_apg_loaders(
    data_dir: str = "data",
    batch_size: int = 8,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, APGCombustionDataset]:
    dataset = APGCombustionDataset(data_dir=data_dir)
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset
