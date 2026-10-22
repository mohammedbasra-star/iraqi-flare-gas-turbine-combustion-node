"""
Stiff Neural ODE Architecture for Sour Flare Gas Combustion Kinetics.

Parametrizes the stiff chemical derivative vector field:
    dy/dt = f_theta(y, c)
where y is the state vector [T, Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_H2S, Y_SO2, Y_SO3, Y_NO]
and c is the operating condition [Pressure, Phi, H2S_pct].
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombustionVectorField(nn.Module):
    """
    MLP approximating the chemical production rates and thermal release.
    """

    def __init__(self, state_dim: int = 9, cond_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim

        # Input projection combining normalized state and operating conditions
        self.net = nn.Sequential(
            nn.Linear(state_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        y: (batch, 9)
        cond: (batch, 3)
        returns dy/dt: (batch, 9)
        """
        inp = torch.cat([y, cond], dim=-1)
        dy_dt = self.net(inp)

        # Enforce total mass conservation constraint on species derivatives:
        # sum_{k=1}^8 (dY_k / dt) = 0
        species_derivs = dy_dt[:, 1:]
        mean_deriv = torch.mean(species_derivs, dim=-1, keepdim=True)
        conserved_species_derivs = species_derivs - mean_deriv

        return torch.cat([dy_dt[:, 0:1], conserved_species_derivs], dim=-1)


class StiffNeuralODE(nn.Module):
    """
    Continuous Neural ODE integrator for multi-step kinetics trajectories.
    """

    def __init__(self, state_dim: int = 9, cond_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.vf = CombustionVectorField(state_dim=state_dim, cond_dim=cond_dim, hidden_dim=hidden_dim)

    def rk4_step(
        self, y: torch.Tensor, cond: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Classic 4th-order Runge-Kutta integration step.
        """
        k1 = self.vf(y, cond)
        k2 = self.vf(y + 0.5 * dt * k1, cond)
        k3 = self.vf(y + 0.5 * dt * k2, cond)
        k4 = self.vf(y + dt * k3, cond)

        y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_next

    def forward(
        self, y0: torch.Tensor, cond: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrates from y0 along the time_steps grid.
        Returns trajectory tensor: (batch, num_steps, 9)
        """
        batch_size = y0.shape[0]
        num_steps = len(time_steps)
        traj = [y0]

        curr_y = y0
        for i in range(num_steps - 1):
            dt = float(time_steps[i + 1] - time_steps[i])
            curr_y = self.rk4_step(curr_y, cond, dt)
            traj.append(curr_y)

        return torch.stack(traj, dim=1)
