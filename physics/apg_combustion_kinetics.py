"""
Associated Petroleum Gas (APG) Chemical Kinetics & Sulfur Oxidation Model.

Models gas turbine combustor reaction kinetics for sour associated gas from
Basra / Rumaila oilfields:
- Methane Number (MN) and Wobbe Index (WI)
- Multi-step hydrocarbon and H2S oxidation:
    CH4 + 2 O2 -> CO2 + 2 H2O
    H2S + 1.5 O2 -> SO2 + H2O
    SO2 + 0.5 O2 <-> SO3
- Thermal energy balance and autoignition delay time
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy.integrate import solve_ivp


class APGCombustionKinetics:
    """
    Stiff multi-step Arrhenius kinetics solver for sour flare gas blends.
    """

    def __init__(
        self,
        pressure_bar: float = 16.0,   # Combustor pressure (bar)
        phi: float = 0.65,            # Lean premixed equivalence ratio
        h2s_pct: float = 1.8,         # H2S mole percentage (e.g. 1.8% = 18,000 ppm)
        ch4_pct: float = 78.0,        # CH4 mole percentage
        c2h6_pct: float = 12.0,       # C2H6 mole percentage
        co2_pct: float = 3.5,         # CO2 mole percentage
        n2_pct: float = 4.7,          # N2 mole percentage
    ) -> None:
        self.p_pa = pressure_bar * 1e5
        self.pressure_bar = pressure_bar
        self.phi = phi
        self.h2s_pct = h2s_pct
        self.ch4_pct = ch4_pct
        self.c2h6_pct = c2h6_pct
        self.co2_pct = co2_pct
        self.n2_pct = n2_pct

        # Universal gas constant
        self.R = 8.314462  # J/(mol*K)

        # Molar masses (kg/mol)
        self.W = {
            "T": 1.0,
            "CH4": 0.01604,
            "O2": 0.03200,
            "CO2": 0.04401,
            "H2O": 0.01802,
            "H2S": 0.03408,
            "SO2": 0.06406,
            "SO3": 0.08006,
            "N2": 0.02801,
            "NO": 0.03001,
        }

        # Heat of reactions (J/mol of fuel)
        self.dH_ch4 = 8.02e5      # Exothermic heat of CH4 oxidation (J/mol)
        self.dH_h2s = 5.18e5      # Exothermic heat of H2S -> SO2 (J/mol)
        self.dH_so2_so3 = 9.89e4  # Exothermic heat of SO2 -> SO3 (J/mol)

        # Gas properties
        self.cp_mean = 1250.0     # Mean specific heat J/(kg*K)

    def compute_fuel_indices(self) -> Dict[str, float]:
        """
        Computes Methane Number (MN), Higher/Lower Heating Value,
        Specific Gravity (SG), and Wobbe Index (WI).
        """
        # Molar fractions
        x_ch4 = self.ch4_pct / 100.0
        x_c2h6 = self.c2h6_pct / 100.0
        x_h2s = self.h2s_pct / 100.0
        x_co2 = self.co2_pct / 100.0
        x_n2 = self.n2_pct / 100.0

        molar_mass_fuel = (
            x_ch4 * self.W["CH4"]
            + x_c2h6 * 0.03007
            + x_h2s * self.W["H2S"]
            + x_co2 * self.W["CO2"]
            + x_n2 * self.W["N2"]
        )

        # Specific gravity relative to air (M_air ~ 0.02896 kg/mol)
        sg = molar_mass_fuel / 0.02896

        # LHV in MJ/Nm^3 (standard cubic meter)
        # CH4 ~ 35.8 MJ/Nm^3, C2H6 ~ 64.3 MJ/Nm^3, H2S ~ 23.3 MJ/Nm^3
        lhv_mj_nm3 = (x_ch4 * 35.8) + (x_c2h6 * 64.3) + (x_h2s * 23.3)
        lhv_mj_kg = (lhv_mj_nm3 / 0.0446) / (molar_mass_fuel * 1000.0)

        # Wobbe Index: WI = LHV / sqrt(SG) (MJ/Nm^3)
        wobbe_index = lhv_mj_nm3 / np.sqrt(max(sg, 0.1))

        # Methane Number empirical estimate (AVL method approx):
        # Higher hydrocarbons (C2+) and H2S lower MN; CO2/N2 raise MN.
        mn = 100.0 * x_ch4 - 150.0 * x_c2h6 - 45.0 * x_h2s + 35.0 * x_co2
        mn = np.clip(mn, 25.0, 95.0)

        return {
            "molar_mass": float(molar_mass_fuel),
            "specific_gravity": float(sg),
            "lhv_mj_kg": float(lhv_mj_kg),
            "lhv_mj_nm3": float(lhv_mj_nm3),
            "wobbe_index": float(wobbe_index),
            "methane_number": float(mn),
        }

    def ode_system(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        State vector y:
            y[0] = T (K)
            y[1] = Y_CH4 (mass fraction)
            y[2] = Y_O2
            y[3] = Y_CO2
            y[4] = Y_H2O
            y[5] = Y_H2S
            y[6] = Y_SO2
            y[7] = Y_SO3
            y[8] = Y_NO
        """
        T = max(state[0], 300.0)
        Y_ch4 = max(state[1], 0.0)
        Y_o2 = max(state[2], 0.0)
        Y_h2s = max(state[5], 0.0)
        Y_so2 = max(state[6], 0.0)

        # Gas density
        rho = self.p_pa / (287.05 * T)

        # Molar concentrations (mol/m^3) = rho * Y_k / W_k
        c_ch4 = max(rho * Y_ch4 / self.W["CH4"], 0.0)
        c_o2 = max(rho * Y_o2 / self.W["O2"], 0.0)
        c_h2s = max(rho * Y_h2s / self.W["H2S"], 0.0)
        c_so2 = max(rho * Y_so2 / self.W["SO2"], 0.0)

        # 1. Methane oxidation rate (Westbrook-Dryer mechanism)
        # r1 = A1 * [CH4]^0.7 * [O2]^0.8 * exp(-E1 / RT)
        k1 = 2.8e9 * np.exp(-1.35e5 / (self.R * T))
        r1 = k1 * (c_ch4 ** 0.7) * (c_o2 ** 0.8)

        # 2. H2S oxidation rate (Muller & Frenklach kinetics)
        # r2 = A2 * [H2S] * [O2]^1.5 * exp(-E2 / RT)
        k2 = 1.6e11 * np.exp(-1.12e5 / (self.R * T))
        r2 = k2 * c_h2s * (c_o2 ** 1.5)

        # 3. SO2 to SO3 equilibrium oxidation
        # r3 = A3 * [SO2] * [O2]^0.5 * exp(-E3 / RT)
        k3 = 4.2e7 * np.exp(-9.5e4 / (self.R * T))
        r3 = k3 * c_so2 * (c_o2 ** 0.5)

        # 4. Thermal Zeldovich NOx rate
        k_no = 1.8e8 * np.exp(-3.83e5 / (self.R * T))
        r_no = 2.0 * k_no * (c_o2 ** 0.5) * (0.75 * rho / self.W["N2"])

        # Heat release rate (W/m^3)
        q_dot = (r1 * self.dH_ch4) + (r2 * self.dH_h2s) + (r3 * self.dH_so2_so3)
        # Cooling / residence heat loss
        q_loss = 1.5e4 * (T - 300.0)
        dT_dt = (q_dot - q_loss) / (rho * self.cp_mean)

        # Species mass rates: dY/dt = omega_dot * W / rho
        dY_ch4 = -r1 * self.W["CH4"] / rho
        dY_h2s = -r2 * self.W["H2S"] / rho
        dY_so2 = (r2 - r3) * self.W["SO2"] / rho
        dY_so3 = r3 * self.W["SO3"] / rho
        dY_o2 = -(2.0 * r1 + 1.5 * r2 + 0.5 * r3) * self.W["O2"] / rho
        dY_co2 = r1 * self.W["CO2"] / rho
        dY_h2o = (2.0 * r1 + r2) * self.W["H2O"] / rho
        dY_no = r_no * self.W["NO"] / rho

        return np.array([dT_dt, dY_ch4, dY_o2, dY_co2, dY_h2o, dY_h2s, dY_so2, dY_so3, dY_no])

    def solve_autoignition(
        self,
        t_span: Tuple[float, float] = (0.0, 0.05),
        T_init: float = 950.0,
        num_points: int = 500,
    ) -> Dict[str, np.ndarray]:
        """
        Integrates the chemical kinetics over the combustor residence time window.
        """
        # Initial mass fractions corresponding to equivalence ratio phi and H2S fraction
        Y_ch4_0 = 0.050 * self.phi
        Y_h2s_0 = (self.h2s_pct / 100.0) * Y_ch4_0 * 1.5
        Y_o2_0 = 0.233  # Atmospheric O2 mass fraction
        y0 = np.array([T_init, Y_ch4_0, Y_o2_0, 0.001, 0.001, Y_h2s_0, 0.0, 0.0, 0.0])

        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        sol = solve_ivp(
            fun=self.ode_system,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="Radau",
            rtol=1e-5,
            atol=1e-7,
        )

        T = sol.y[0]
        t = sol.t

        # Autoignition delay time (maximum dT/dt point)
        dT_dt = np.gradient(T, t)
        idx_ign = int(np.argmax(dT_dt))
        tau_ign = float(t[idx_ign])

        return {
            "time": t,
            "T": T,
            "Y_CH4": sol.y[1],
            "Y_O2": sol.y[2],
            "Y_CO2": sol.y[3],
            "Y_H2O": sol.y[4],
            "Y_H2S": sol.y[5],
            "Y_SO2": sol.y[6],
            "Y_SO3": sol.y[7],
            "Y_NO": sol.y[8],
            "tau_ign": tau_ign,
            "T_peak": float(np.max(T)),
        }
