"""
Training Pipeline for Sour APG Combustion Stiff Neural ODE.
"""

from typing import Dict, List, Tuple
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from physics.apg_combustion_kinetics import APGCombustionKinetics
from data.generate_apg_data import generate_apg_dataset
from data.dataset import get_apg_loaders
from models.neural_ode_combustion import StiffNeuralODE
from utils.visualization import (
    plot_autoignition_temperature_profile,
    plot_species_kinetics_sulfur,
    plot_wobbe_index_methane_number,
    plot_neural_ode_convergence,
)


def run_training(
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 3e-3,
    device_name: str = "cpu",
    output_dir: str = "checkpoints",
    figures_dir: str = "figures",
) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_dir)
    figures_dir = os.path.join(base_dir, figures_dir)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"=== Sour APG Combustion Stiff Neural ODE Training ===")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")

    # Step 1: Ensure dataset exists
    traj_file = os.path.join(data_dir, "trajectories.pt")
    if not os.path.exists(traj_file):
        print("Dataset not found. Generating APG kinetics trajectories...")
        generate_apg_dataset(num_cases=40, num_timesteps=100, output_dir=data_dir)

    train_loader, val_loader, dataset = get_apg_loaders(data_dir=data_dir, batch_size=batch_size)
    time_steps = dataset.time.to(device)

    # Step 2: Initialize Neural ODE model
    model = StiffNeuralODE(state_dim=9, cond_dim=3, hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    start_time = time.time()

    # Temperature scale factor
    T_scale = 2000.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for cond, traj in train_loader:
            cond = cond.to(device)
            traj = traj.to(device)

            # Normalize temperature channel
            traj_norm = traj.clone()
            traj_norm[:, :, 0] = traj[:, :, 0] / T_scale

            y0 = traj_norm[:, 0, :]

            optimizer.zero_grad()
            # Predict trajectory
            pred_traj = model(y0, cond, time_steps)
            loss = criterion(pred_traj, traj_norm)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * cond.size(0)

        scheduler.step()
        train_loss = total_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for cond, traj in val_loader:
                cond = cond.to(device)
                traj = traj.to(device)
                traj_norm = traj.clone()
                traj_norm[:, :, 0] = traj[:, :, 0] / T_scale
                y0 = traj_norm[:, 0, :]
                pred_traj = model(y0, cond, time_steps)
                loss = criterion(pred_traj, traj_norm)
                total_val_loss += loss.item() * cond.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                },
                os.path.join(output_dir, "best_apg_node_model.pt"),
            )

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} s. Best Val Loss: {best_val_loss:.6f}")

    # Step 3: Generate Publication Figures
    print("\nGenerating publication figures...")

    # Figure 1: Autoignition Temperature Profiles across H2S concentrations
    k1 = APGCombustionKinetics(h2s_pct=0.5).solve_autoignition()
    k2 = APGCombustionKinetics(h2s_pct=1.8).solve_autoignition()
    k3 = APGCombustionKinetics(h2s_pct=3.5).solve_autoignition()
    plot_autoignition_temperature_profile(
        h2s_cases=[k1, k2, k3],
        labels=["0.5% H2S (Sweetened)", "1.8% H2S (Rumaila Typical)", "3.5% H2S (High-Sour Flare)"],
        save_path=os.path.join(figures_dir, "apg_autoignition_and_temperature_profile.png"),
    )

    # Figure 2: Species Kinetics & Sulfur Oxidation Cascade
    plot_species_kinetics_sulfur(
        sol=k2, save_path=os.path.join(figures_dir, "species_kinetics_sulfur_oxidation.png")
    )

    # Figure 3: Wobbe Index vs Methane Number Map
    plot_wobbe_index_methane_number(
        save_path=os.path.join(figures_dir, "wobbe_index_methane_number_map.png")
    )

    # Figure 4: Neural ODE Convergence & Parity
    sample_cond, sample_traj = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        s_norm = sample_traj[0:1].clone()
        s_norm[:, :, 0] /= T_scale
        pred = model(s_norm[:, 0, :].to(device), sample_cond[0:1].to(device), time_steps).cpu()

    true_T = sample_traj[0, :, 0].numpy()
    pred_T = pred[0, :, 0].numpy() * T_scale
    t_np = dataset.time.numpy()

    plot_neural_ode_convergence(
        train_losses=train_losses,
        val_losses=val_losses,
        t_span=t_np,
        true_T=true_T,
        pred_T=pred_T,
        save_path=os.path.join(figures_dir, "neural_ode_stiff_convergence.png"),
    )
    print("All 4 publication figures generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
    )
