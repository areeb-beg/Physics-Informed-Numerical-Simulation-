"""Quasi-Steady-State (QSS) Lap Time Solver.

Performs three-step numerical integration across track distance nodes:
Step 1: Compute apex and cornering velocity limits from track curvature.
Step 2: Forward spatial integration pass for powertrain traction limits.
Step 3: Backward spatial integration pass for braking limits.
Supports both Euler and 4th-Order Runge-Kutta (RK4) spatial integrators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

try:
    from lap_time_simulator.src.vehicle_model import VehicleModel
except ImportError:
    from src.vehicle_model import VehicleModel


@dataclass
class SimulationResult:
    """Telemetry data output container for a solved lap simulation."""

    distance: np.ndarray
    velocity: np.ndarray
    velocity_kmh: np.ndarray
    v_corner: np.ndarray
    v_fwd: np.ndarray
    v_bwd: np.ndarray
    time: np.ndarray
    dt: np.ndarray
    a_x: np.ndarray
    a_y: np.ndarray
    a_x_g: np.ndarray
    a_y_g: np.ndarray
    a_total_g: np.ndarray
    f_drag: np.ndarray
    f_downforce: np.ndarray
    f_z_total: np.ndarray
    f_tire_max: np.ndarray
    f_x_tire: np.ndarray
    f_y_tire: np.ndarray
    power_engine: np.ndarray
    power_engine_kw: np.ndarray
    f_brake: np.ndarray
    lap_time: float
    top_speed_kmh: float
    avg_speed_kmh: float
    sector_times: Dict[int, float]
    sector_metrics: Dict[int, Dict[str, float]]

    def to_dataframe(self) -> pd.DataFrame:
        """Export core telemetry channels to a Pandas DataFrame."""
        return pd.DataFrame(
            {
                "distance_m": np.round(self.distance, 3),
                "time_s": np.round(self.time, 4),
                "velocity_ms": np.round(self.velocity, 3),
                "velocity_kmh": np.round(self.velocity_kmh, 2),
                "v_corner_kmh": np.round(self.v_corner * 3.6, 2),
                "accel_x_g": np.round(self.a_x_g, 4),
                "accel_y_g": np.round(self.a_y_g, 4),
                "accel_total_g": np.round(self.a_total_g, 4),
                "drag_n": np.round(self.f_drag, 1),
                "downforce_n": np.round(self.f_downforce, 1),
                "power_kw": np.round(self.power_engine_kw, 1),
                "brake_n": np.round(self.f_brake, 1),
            }
        )

    def summary(self) -> str:
        """Generate human-readable summary of lap performance."""
        lines = [
            "=" * 60,
            " LAP SIMULATION TELEMETRY SUMMARY",
            "=" * 60,
            f" Total Lap Time   : {self.lap_time:.3f} s",
            f" Top Speed        : {self.top_speed_kmh:.2f} km/h",
            f" Average Speed    : {self.avg_speed_kmh:.2f} km/h",
            f" Peak Lateral G   : {np.max(np.abs(self.a_y_g)):.2f} g",
            f" Peak Long. Accel : {np.max(self.a_x_g):.2f} g",
            f" Peak Decel       : {np.min(self.a_x_g):.2f} g",
            f" Peak Downforce   : {np.max(self.f_downforce):.1f} N",
            "-" * 60,
            " SECTOR BREAKDOWN",
            "-" * 60,
        ]
        for sec_id, t_sec in sorted(self.sector_times.items()):
            m = self.sector_metrics.get(sec_id, {})
            v_max = m.get("v_max_kmh", 0.0)
            v_min = m.get("v_min_kmh", 0.0)
            lines.append(
                f" Sector {sec_id}: {t_sec:6.3f} s | Min: {v_min:5.1f} km/h | Max: {v_max:5.1f} km/h"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class LapTimeSolver:
    """Quasi-Steady-State Point-Mass Numerical Integrator."""

    def __init__(
        self,
        vehicle: VehicleModel,
        method: Literal["rk4", "euler"] = "rk4",
    ) -> None:
        """Initialize solver with vehicle model and integration scheme.

        Parameters
        ----------
        vehicle : VehicleModel
            Configured vehicle physics model.
        method : Literal["rk4", "euler"]
            Integration scheme. 'rk4' uses 4th-order Runge-Kutta in space;
            'euler' uses forward/backward Euler.
        """
        self.vehicle = vehicle
        self.method = method

    def solve(
        self,
        track_df: pd.DataFrame,
        mode: Literal["flying", "standing"] = "flying",
        v_start: float = 0.0,
        max_cyclic_iterations: int = 5,
        tolerance_v: float = 0.05,
    ) -> SimulationResult:
        """Execute the full 3-step QSS numerical simulation.

        Parameters
        ----------
        track_df : pd.DataFrame
            Track centerline DataFrame with columns: distance, curvature, and optional sector.
        mode : Literal["flying", "standing"]
            'flying' enforces periodic boundary conditions v(0) == v(end).
            'standing' initializes at v_start.
        v_start : float
            Starting speed for standing start [m/s].
        max_cyclic_iterations : int
            Maximum iterations for flying lap periodic boundary convergence.
        tolerance_v : float
            Speed convergence threshold [m/s] between lap cycles.

        Returns
        -------
        SimulationResult
            Complete telemetry channels and lap metrics.
        """
        # Vectorized extraction of track geometry
        s_arr = np.asarray(track_df["distance"].values, dtype=float)
        kappa_arr = np.asarray(track_df["curvature"].values, dtype=float)
        n_nodes = len(s_arr)

        if n_nodes < 2:
            raise ValueError("Track must contain at least 2 nodes.")

        # Compute spatial increments ds
        ds_arr = np.diff(s_arr)
        if np.any(ds_arr <= 0.0):
            raise ValueError("Track distance coordinates must be strictly monotonically increasing.")

        # =========================================================================
        # STEP 1: Vectorized Curvature Apex Velocity Limit (Cornering Boundary)
        # =========================================================================
        # Evaluated purely with vectorized NumPy
        v_corner = self.vehicle.max_cornering_speed(kappa_arr)

        # Initialize working velocity profile
        v_profile = np.copy(v_corner)

        # Boundary condition setup
        if mode == "standing":
            v_init = max(v_start, self.vehicle.params.v_min)
            v_profile[0] = min(v_init, v_corner[0])
            iterations = 1
        else:
            iterations = max_cyclic_iterations

        # Cyclic convergence loop for flying laps (ensures v_0 == v_end)
        for _ in range(iterations):
            v_prev_start = v_profile[0]

            # =====================================================================
            # STEP 2: Forward Numerical Integration (Acceleration Pass)
            # =====================================================================
            v_fwd = self._forward_pass(s_arr, ds_arr, kappa_arr, v_corner, v_profile[0])

            # =====================================================================
            # STEP 3: Backward Numerical Integration (Deceleration Pass)
            # =====================================================================
            v_end_target = v_fwd[0] if mode == "flying" else v_fwd[-1]
            v_bwd = self._backward_pass(s_arr, ds_arr, kappa_arr, v_fwd, v_end_target)

            v_profile = np.copy(v_bwd)

            if mode == "standing" or abs(v_profile[0] - v_prev_start) < tolerance_v:
                break

        # Compute complete telemetry channels
        return self._compute_telemetry(track_df, s_arr, ds_arr, kappa_arr, v_profile, v_corner, v_fwd, v_bwd)

    def _forward_pass(
        self,
        s_arr: np.ndarray,
        ds_arr: np.ndarray,
        kappa_arr: np.ndarray,
        v_corner: np.ndarray,
        v_start: float,
    ) -> np.ndarray:
        """Forward spatial integration of longitudinal acceleration limits."""
        n = len(s_arr)
        v = np.zeros(n, dtype=float)
        v[0] = min(v_start, v_corner[0])

        is_rk4 = self.method == "rk4"

        # Sequential stepping in space: d(v^2)/ds = 2 * a_x
        for i in range(n - 1):
            ds = ds_arr[i]
            v_i = v[i]
            a_y_i = kappa_arr[i] * (v_i**2)
            y_i = v_i**2

            if is_rk4:
                # RK4 integration on y = v^2
                k1 = 2.0 * self.vehicle.max_longitudinal_acceleration(v_i, a_y_i)

                y_mid1 = max(y_i + 0.5 * ds * k1, 0.0)
                v_mid1 = np.sqrt(y_mid1)
                a_y_mid = 0.5 * (kappa_arr[i] + kappa_arr[i + 1]) * y_mid1
                k2 = 2.0 * self.vehicle.max_longitudinal_acceleration(v_mid1, a_y_mid)

                y_mid2 = max(y_i + 0.5 * ds * k2, 0.0)
                v_mid2 = np.sqrt(y_mid2)
                k3 = 2.0 * self.vehicle.max_longitudinal_acceleration(v_mid2, a_y_mid)

                y_end = max(y_i + ds * k3, 0.0)
                v_end = np.sqrt(y_end)
                a_y_next = kappa_arr[i + 1] * y_end
                k4 = 2.0 * self.vehicle.max_longitudinal_acceleration(v_end, a_y_next)

                y_next = y_i + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                # Forward Euler on y = v^2
                a_x = self.vehicle.max_longitudinal_acceleration(v_i, a_y_i)
                y_next = y_i + 2.0 * a_x * ds

            v_candidate = np.sqrt(max(y_next, 0.0))
            # Bound velocity by local cornering limit
            v[i + 1] = min(v_candidate, v_corner[i + 1])

        return v

    def _backward_pass(
        self,
        s_arr: np.ndarray,
        ds_arr: np.ndarray,
        kappa_arr: np.ndarray,
        v_fwd: np.ndarray,
        v_end_target: float,
    ) -> np.ndarray:
        """Backward spatial integration of braking deceleration limits."""
        n = len(s_arr)
        v = np.copy(v_fwd)
        v[-1] = min(v[-1], v_end_target)

        is_rk4 = self.method == "rk4"

        # Backward integration: traveling backward from n-1 down to 0
        for i in range(n - 2, -1, -1):
            ds = ds_arr[i]
            v_next = v[i + 1]
            a_y_next = kappa_arr[i + 1] * (v_next**2)
            y_next = v_next**2

            if is_rk4:
                # RK4 backward step: y_i = y_next + 2 * |a_brake| * ds
                k1 = 2.0 * self.vehicle.max_longitudinal_deceleration(v_next, a_y_next)

                y_mid1 = y_next + 0.5 * ds * k1
                v_mid1 = np.sqrt(max(y_mid1, 0.0))
                a_y_mid = 0.5 * (kappa_arr[i] + kappa_arr[i + 1]) * y_mid1
                k2 = 2.0 * self.vehicle.max_longitudinal_deceleration(v_mid1, a_y_mid)

                y_mid2 = y_next + 0.5 * ds * k2
                v_mid2 = np.sqrt(max(y_mid2, 0.0))
                k3 = 2.0 * self.vehicle.max_longitudinal_deceleration(v_mid2, a_y_mid)

                y_end = y_next + ds * k3
                v_end = np.sqrt(max(y_end, 0.0))
                a_y_prev = kappa_arr[i] * y_end
                k4 = 2.0 * self.vehicle.max_longitudinal_deceleration(v_end, a_y_prev)

                y_prev = y_next + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                # Backward Euler
                a_brake = self.vehicle.max_longitudinal_deceleration(v_next, a_y_next)
                y_prev = y_next + 2.0 * a_brake * ds

            v_brake_limit = np.sqrt(max(y_prev, 0.0))
            # Take the most restrictive limit between forward pass and braking requirement
            v[i] = min(v[i], v_brake_limit)

        return v

    def _compute_telemetry(
        self,
        track_df: pd.DataFrame,
        s_arr: np.ndarray,
        ds_arr: np.ndarray,
        kappa_arr: np.ndarray,
        v_profile: np.ndarray,
        v_corner: np.ndarray,
        v_fwd: np.ndarray,
        v_bwd: np.ndarray,
    ) -> SimulationResult:
        """Vectorized computation of all derived telemetry channels and sector metrics."""
        g = self.vehicle.params.g
        mass = self.vehicle.params.mass

        v = np.copy(v_profile)
        v_kmh = v * 3.6

        # -------------------------------------------------------------------------
        # Vectorized Time Increments (Trapezoidal integration in spatial domain)
        # dt_i = 2 * ds_i / (v_i + v_{i+1})
        # -------------------------------------------------------------------------
        v_mean_step = 0.5 * (v[:-1] + v[1:])
        v_mean_step = np.maximum(v_mean_step, self.vehicle.params.v_min)
        dt_steps = ds_arr / v_mean_step
        time_arr = np.concatenate(([0.0], np.cumsum(dt_steps)))
        dt_full = np.concatenate(([dt_steps[0]], dt_steps))
        lap_time = float(time_arr[-1])

        # -------------------------------------------------------------------------
        # Vectorized Accelerations
        # Centripetal lateral acceleration: a_y = kappa * v^2
        # Longitudinal acceleration: a_x = d(v^2) / (2 * ds)
        # -------------------------------------------------------------------------
        a_y = kappa_arr * (v**2)

        # Central difference for interior nodes, forward/backward at ends
        v_sq = v**2
        d_v_sq = np.gradient(v_sq, s_arr)
        a_x = 0.5 * d_v_sq

        a_x_g = a_x / g
        a_y_g = a_y / g
        a_total_g = np.sqrt(a_x_g**2 + a_y_g**2)

        # -------------------------------------------------------------------------
        # Vectorized Aerodynamics and Forces
        # -------------------------------------------------------------------------
        f_drag = self.vehicle.aerodynamic_drag(v)
        f_downforce = self.vehicle.aerodynamic_downforce(v)
        f_z_total = self.vehicle.total_vertical_load(v)
        mu_arr = self.vehicle.tire_friction_coefficient(f_z_total)
        f_tire_max = mu_arr * f_z_total

        f_y_tire = mass * np.abs(a_y)
        f_y_tire = np.minimum(f_y_tire, f_tire_max)

        f_rr = self.vehicle.params.crr * self.vehicle.static_weight

        # Powertrain and brake force decomposition
        # m * a_x = F_drive - F_drag - F_rr   (when accelerating)
        # m * a_x = -(F_brake + F_drag + F_rr) (when braking)
        f_net_long = mass * a_x
        f_traction_demand = f_net_long + f_drag + f_rr

        power_engine = np.maximum(f_traction_demand * v / self.vehicle.params.drive_efficiency, 0.0)
        power_engine_kw = power_engine / 1000.0

        f_brake = np.maximum(-f_traction_demand, 0.0)
        f_x_tire = np.maximum(f_traction_demand, f_brake)

        top_speed_kmh = float(np.max(v_kmh))
        avg_speed_kmh = float((s_arr[-1] / lap_time) * 3.6)

        # -------------------------------------------------------------------------
        # Sector Partitioning & Performance Metrics
        # -------------------------------------------------------------------------
        sector_times: Dict[int, float] = {}
        sector_metrics: Dict[int, Dict[str, float]] = {}

        if "sector" in track_df.columns:
            sectors = track_df["sector"].values
            unique_sectors = np.unique(sectors)
            for sec in unique_sectors:
                sec_mask = sectors == sec
                sec_indices = np.where(sec_mask)[0]
                if len(sec_indices) == 0:
                    continue

                idx_start = sec_indices[0]
                idx_end = sec_indices[-1]

                t_start = time_arr[idx_start]
                t_end = time_arr[min(idx_end + 1, len(time_arr) - 1)]
                s_elapsed = float(t_end - t_start)
                sector_times[int(sec)] = s_elapsed

                v_sec = v_kmh[sec_mask]
                sector_metrics[int(sec)] = {
                    "sector_time_s": s_elapsed,
                    "start_distance_m": float(s_arr[idx_start]),
                    "end_distance_m": float(s_arr[idx_end]),
                    "entry_speed_kmh": float(v_kmh[idx_start]),
                    "exit_speed_kmh": float(v_kmh[idx_end]),
                    "v_max_kmh": float(np.max(v_sec)),
                    "v_min_kmh": float(np.min(v_sec)),
                }
        else:
            # Default three equal sectors if none specified
            s_total = s_arr[-1]
            sec_bounds = [0.0, s_total / 3.0, 2.0 * s_total / 3.0, s_total]
            for i in range(3):
                mask = (s_arr >= sec_bounds[i]) & (s_arr <= sec_bounds[i + 1])
                idx = np.where(mask)[0]
                t_sec = float(time_arr[idx[-1]] - time_arr[idx[0]])
                sector_times[i + 1] = t_sec
                sector_metrics[i + 1] = {
                    "sector_time_s": t_sec,
                    "start_distance_m": float(sec_bounds[i]),
                    "end_distance_m": float(sec_bounds[i + 1]),
                    "entry_speed_kmh": float(v_kmh[idx[0]]),
                    "exit_speed_kmh": float(v_kmh[idx[-1]]),
                    "v_max_kmh": float(np.max(v_kmh[mask])),
                    "v_min_kmh": float(np.min(v_kmh[mask])),
                }

        return SimulationResult(
            distance=s_arr,
            velocity=v,
            velocity_kmh=v_kmh,
            v_corner=v_corner,
            v_fwd=v_fwd,
            v_bwd=v_bwd,
            time=time_arr,
            dt=dt_full,
            a_x=a_x,
            a_y=a_y,
            a_x_g=a_x_g,
            a_y_g=a_y_g,
            a_total_g=a_total_g,
            f_drag=f_drag,
            f_downforce=f_downforce,
            f_z_total=f_z_total,
            f_tire_max=f_tire_max,
            f_x_tire=f_x_tire,
            f_y_tire=f_y_tire,
            power_engine=power_engine,
            power_engine_kw=power_engine_kw,
            f_brake=f_brake,
            lap_time=lap_time,
            top_speed_kmh=top_speed_kmh,
            avg_speed_kmh=avg_speed_kmh,
            sector_times=sector_times,
            sector_metrics=sector_metrics,
        )
