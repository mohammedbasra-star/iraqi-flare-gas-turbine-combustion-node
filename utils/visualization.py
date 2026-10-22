"""
Publication-Grade Visualizations for Sour Flare Gas Turbine Neural ODE.

Generates 300 DPI figures:
1. APG Autoignition Temperature Trajectories & Delay Time vs H2S Concentration.
2. Species Mass Fractions & Sulfur Oxidation Kinetics (H2S -> SO2 -> SO3).
3. Fuel Operational Envelope: Wobbe Index vs Methane Number for Basra APG Blends.
4. Neural ODE Training Convergence & Kinetic Stiffness Spectrum.
"""

from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
})


def plot_autoignition_temperature_profile(
    h2s_cases: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: str = "figures/apg_autoignition_and_temperature_profile.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    delays = []
    h2s_pcts = [0.5, 1.8, 3.5]

    for sol, label, color in zip(h2s_cases, labels, colors):
        t_ms = sol["time"] * 1e3
        T = sol["T"]
        ax1.plot(t_ms, T, color=color, linewidth=2.2, label=label)
        delays.append(sol["tau_ign"] * 1e3)

    ax1.set_xlabel("Residence Time (ms)")
    ax1.set_ylabel("Combustor Temperature $T$ (K)")
    ax1.set_title("(a) Thermal Runaway & Autoignition Profile")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Delay time vs H2S
    ax2.plot(h2s_pcts, delays, "o-", color="#d62728", linewidth=2.5, markersize=8)
    ax2.set_xlabel(r"$\mathrm{H_2S}$ Concentration in Flare Gas (mol %)")
    ax2.set_ylabel(r"Autoignition Delay $\tau_{\mathrm{ign}}$ (ms)")
    ax2.set_title(r"(b) Combustor Flashback Risk ($\tau_{\mathrm{ign}}$ Reduction)")
    ax2.axhspan(min(delays), 12.0, color="red", alpha=0.15, label="High Flashback Hazard Zone")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Basra Associated Petroleum Gas (APG) Autoignition Dynamics", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_species_kinetics_sulfur(
    sol: Dict[str, np.ndarray],
    save_path: str = "figures/species_kinetics_sulfur_oxidation.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    t_ms = sol["time"] * 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Hydrocarbon & Main species
    ax1.plot(t_ms, sol["Y_CH4"] * 100.0, color="#1f77b4", label=r"$\mathrm{CH_4}$ (wt %)")
    ax1.plot(t_ms, sol["Y_O2"] * 100.0, color="#2ca02c", label=r"$\mathrm{O_2}$ (wt %)")
    ax1.plot(t_ms, sol["Y_CO2"] * 100.0, color="#ff7f0e", label=r"$\mathrm{CO_2}$ (wt %)")
    ax1.plot(t_ms, sol["Y_H2O"] * 100.0, color="#17becf", linestyle="--", label=r"$\mathrm{H_2O}$ (wt %)")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Species Mass Fraction (wt %)")
    ax1.set_title(r"(a) Major Hydrocarbon Reactants & Products")
    ax1.legend(loc="center right")
    ax1.grid(True, alpha=0.3)

    # Sulfur Oxidation Species (H2S, SO2, SO3) and NO
    ax2.semilogy(t_ms, np.maximum(sol["Y_H2S"] * 1e4, 1e-4), color="#8c564b", linewidth=2.0, label=r"$\mathrm{H_2S}$ ($\times 10^{-4}$)")
    ax2.semilogy(t_ms, np.maximum(sol["Y_SO2"] * 1e4, 1e-4), color="#e377c2", linewidth=2.0, label=r"$\mathrm{SO_2}$ ($\times 10^{-4}$)")
    ax2.semilogy(t_ms, np.maximum(sol["Y_SO3"] * 1e4, 1e-4), color="#d62728", linestyle="--", linewidth=2.2, label=r"$\mathrm{SO_3}$ Acid Precursor ($\times 10^{-4}$)")
    ax2.semilogy(t_ms, np.maximum(sol["Y_NO"] * 1e6, 1e-3), color="#7f7f7f", linestyle="-.", label=r"$\mathrm{NO_x}$ (ppmw)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Trace Species Mass Fraction (Log Scale)")
    ax2.set_title(r"(b) Sulfur Oxidation Cascade & Acid Dewpoint Precursors")
    ax2.legend(loc="lower right")
    ax2.grid(True, which="both", alpha=0.3)

    plt.suptitle("Sour Gas Turbine Multi-Species Chemical Kinetics & Sulfur Pathways", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_wobbe_index_methane_number(
    save_path: str = "figures/wobbe_index_methane_number_map.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Gas turbine OEM operational window (e.g. GE Frame 9E / Siemens SGT-800)
    # Wobbe index range: 45 - 53 MJ/Nm^3; Methane number > 65
    ax.fill_between([45.0, 52.5], 65.0, 100.0, color="green", alpha=0.15, label="OEM Standard Fuel Envelope")
    ax.fill_between([40.0, 56.0], 45.0, 65.0, color="yellow", alpha=0.20, label="Derated / Tuning Required Zone")
    ax.fill_between([35.0, 58.0], 20.0, 45.0, color="red", alpha=0.15, label="High Knock & Flashback Risk")

    # Iraqi Field Gas Points
    fields = {
        "Pipeline Sweet Gas (Basra Gas Co.)": (49.8, 88.0, "go"),
        "Rumaila APG Flare Gas (Untreated)": (53.2, 54.0, "r^"),
        "West Qurna APG High-C2 Blend": (54.5, 48.0, "rs"),
        "Zubair Degassed Sour Gas": (47.2, 62.0, "mD"),
        "Majnoon Heavy Associated Gas": (51.8, 42.0, "kX"),
    }

    for name, (wi, mn, marker) in fields.items():
        ax.plot(wi, mn, marker, markersize=10, label=name)

    ax.set_xlabel(r"Modified Wobbe Index $WI$ ($\mathrm{MJ / Nm^3}$)")
    ax.set_ylabel("Methane Number (MN)")
    ax.set_title("Gas Turbine Fuel Compliance: Iraqi Associated Flare Gas vs OEM Specs", fontsize=13)
    ax.set_xlim(36.0, 58.0)
    ax.set_ylim(25.0, 95.0)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_neural_ode_convergence(
    train_losses: List[float],
    val_losses: List[float],
    t_span: np.ndarray,
    true_T: np.ndarray,
    pred_T: np.ndarray,
    save_path: str = "figures/neural_ode_stiff_convergence.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss convergence
    ax1.semilogy(epochs, train_losses, color="#1f77b4", linewidth=2.0, label="Train Loss (MSE)")
    ax1.semilogy(epochs, val_losses, color="#ff7f0e", linestyle="--", linewidth=2.0, label="Val Loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Multi-Species Mean Squared Error")
    ax1.set_title("(a) Stiff Neural ODE Training Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Parity comparison: True vs Neural ODE Temperature Trajectory
    ax2.plot(t_span * 1e3, true_T, color="black", linewidth=2.2, label=r"Radau Stiff Solver $T(t)$")
    ax2.plot(t_span * 1e3, pred_T, color="#d62728", linestyle="--", linewidth=2.0, label=r"Neural ODE Predicted $T(t)$")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Combustor Temperature (K)")
    ax2.set_title("(b) Kinetic Trajectory Tracking Fidelity")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Stiff Neural ODE Diagnostics and Generalization", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
