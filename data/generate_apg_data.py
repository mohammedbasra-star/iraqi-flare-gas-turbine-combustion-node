"""
Dataset Generator for Sour APG Combustion Chemical Kinetics.
"""

from typing import Dict, List, Tuple
import os
import argparse
import numpy as np
import torch
from physics.apg_combustion_kinetics import APGCombustionKinetics


def generate_apg_dataset(
    num_cases: int = 40,
    num_timesteps: int = 100,
    output_dir: str = "data",
    seed: int = 42,
) -> None:
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    h2s_vals = np.linspace(0.5, 3.5, num_cases)
    phi_vals = np.random.uniform(0.55, 0.80, num_cases)
    p_vals = np.random.uniform(12.0, 18.0, num_cases)

    trajectories = []
    conditions = []  # [p_bar, phi, h2s_pct]

    print(f"Generating {num_cases} combustion kinetics trajectories...")
    for i in range(num_cases):
        kinetics = APGCombustionKinetics(
            pressure_bar=float(p_vals[i]),
            phi=float(phi_vals[i]),
            h2s_pct=float(h2s_vals[i]),
        )
        sol = kinetics.solve_autoignition(t_span=(0.0, 0.04), num_points=num_timesteps)

        # State matrix: (num_timesteps, 9)
        state_mat = np.stack(
            [
                sol["T"],
                sol["Y_CH4"],
                sol["Y_O2"],
                sol["Y_CO2"],
                sol["Y_H2O"],
                sol["Y_H2S"],
                sol["Y_SO2"],
                sol["Y_SO3"],
                sol["Y_NO"],
            ],
            axis=-1,
        )

        trajectories.append(state_mat)
        conditions.append([p_vals[i], phi_vals[i], h2s_vals[i]])

    traj_tensor = torch.tensor(np.array(trajectories), dtype=torch.float32)  # (N, T_steps, 9)
    cond_tensor = torch.tensor(np.array(conditions), dtype=torch.float32)    # (N, 3)
    time_tensor = torch.tensor(sol["time"], dtype=torch.float32)

    torch.save(traj_tensor, os.path.join(output_dir, "trajectories.pt"))
    torch.save(cond_tensor, os.path.join(output_dir, "conditions.pt"))
    torch.save(time_tensor, os.path.join(output_dir, "time.pt"))

    print(f"Dataset generated successfully:")
    print(f"  Trajectories: {traj_tensor.shape}")
    print(f"  Saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    generate_apg_dataset(num_cases=args.cases, num_timesteps=args.steps)
