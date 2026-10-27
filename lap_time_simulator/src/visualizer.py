"""Telemetry Visualization and Performance Diagnostics.

Generates engineering-grade plots including multi-channel telemetry velocity
profiles, G-G friction ellipse diagrams, delta-time comparisons, and 2D track maps.
"""

from __future__ import annotations

import os
import tempfile

# Configure headless-compatible matplotlib directory before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lap_time_simulator.src.solver import SimulationResult
    from lap_time_simulator.src.vehicle_model import VehicleModel
except ImportError:
    from src.solver import SimulationResult
    from src.vehicle_model import VehicleModel


class LapVisualizer:
    """Publication-quality telemetry plotter for vehicle dynamics analysis."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid") -> None:
        """Initialize visualizer with plot styling.

        Parameters
        ----------
        style : str
            Matplotlib style name. Falls back safely if style not available.
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

        # Global typography & layout styling
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.titlesize": 13,
                "lines.linewidth": 1.5,
                "grid.alpha": 0.4,
            }
        )

    def plot_velocity_profile(
        self,
        result: SimulationResult,
        save_path: Optional[str] = None,
        title_suffix: str = "",
    ) -> plt.Figure:
        """Plot synchronized 3-panel telemetry dashboard across track distance.

        Panel 1: Speed profile vs cornering limits & integration envelopes [km/h]
        Panel 2: Longitudinal and lateral accelerations [g]
        Panel 3: Aerodynamic downforce and drag forces [N]

        Parameters
        ----------
        result : SimulationResult
            Simulation output result.
        save_path : Optional[str]
            File path to save the generated figure.
        title_suffix : str
            Optional subtitle / vehicle tag.

        Returns
        -------
        plt.Figure
            Rendered Matplotlib figure.
        """
        fig, (ax_v, ax_g, ax_f) = plt.subplots(
            nrows=3, ncols=1, figsize=(13, 10), sharex=True, constrained_layout=True
        )

        s = result.distance

        # ---------------------------------------------------------------------
        # Panel 1: Velocity Profile
        # ---------------------------------------------------------------------
        ax_v.plot(
            s,
            result.v_corner * 3.6,
            color="#e74c3c",
            linestyle="--",
            alpha=0.65,
            label="Cornering Limit $v_{\\rm corner}$",
        )
        ax_v.plot(
            s,
            result.v_fwd * 3.6,
            color="#9b59b6",
            linestyle=":",
            alpha=0.55,
            label="Forward Accel Pass $v_{\\rm fwd}$",
        )
        ax_v.plot(
            s,
            result.v_bwd * 3.6,
            color="#f39c12",
            linestyle=":",
            alpha=0.55,
            label="Backward Braking Pass $v_{\\rm bwd}$",
        )
        ax_v.plot(
            s,
            result.velocity_kmh,
            color="#2980b9",
            linewidth=2.0,
            label="Final Trajectory $v(s)$",
        )

        # Annotate sector boundaries
        for sec_id, metric in result.sector_metrics.items():
            s_end = metric["end_distance_m"]
            ax_v.axvline(s_end, color="gray", linestyle="-.", alpha=0.5)
            ax_v.text(
                s_end - 150.0,
                np.max(result.velocity_kmh) * 0.92,
                f"S{sec_id}",
                fontweight="bold",
                color="#555555",
            )

        ax_v.set_ylabel("Speed [km/h]")
        ax_v.set_title(
            f"Point-Mass Lap Velocity Profile | Lap Time: {result.lap_time:.3f} s | Top Speed: {result.top_speed_kmh:.1f} km/h {title_suffix}"
        )
        ax_v.legend(loc="lower right", framealpha=0.9)
        ax_v.grid(True)

        # ---------------------------------------------------------------------
        # Panel 2: Acceleration Channels
        # ---------------------------------------------------------------------
        ax_g.plot(s, result.a_x_g, color="#27ae60", label="Longitudinal Accel $a_x$ [g]")
        ax_g.plot(s, result.a_y_g, color="#d35400", label="Lateral Centripetal Accel $a_y$ [g]")
        ax_g.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
        ax_g.set_ylabel("Acceleration [g]")
        ax_g.legend(loc="upper right", framealpha=0.9)
        ax_g.grid(True)

        # ---------------------------------------------------------------------
        # Panel 3: Aerodynamic Forces
        # ---------------------------------------------------------------------
        ax_f.plot(s, result.f_downforce, color="#16a085", label="Downforce $F_z^{\\rm aero}$ [N]")
        ax_f.plot(s, result.f_drag, color="#c0392b", label="Drag $F_d$ [N]")
        ax_f.set_xlabel("Track Distance $s$ [m]")
        ax_f.set_ylabel("Aerodynamic Force [N]")
        ax_f.legend(loc="upper right", framealpha=0.9)
        ax_f.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

        return fig

    def plot_gg_diagram(
        self,
        result: SimulationResult,
        vehicle: Optional[VehicleModel] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot the G-G tire friction diagram (Kamm's Circle) with speed colormap.

        Parameters
        ----------
        result : SimulationResult
            Simulation result containing longitudinal and lateral accelerations.
        vehicle : Optional[VehicleModel]
            Vehicle model for plotting theoretical limit envelopes.
        save_path : Optional[str]
            Path to save figure.

        Returns
        -------
        plt.Figure
            Rendered Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        # Symmetric signed lateral acceleration: signed_a_y = a_y * sign(curvature)
        # For point-mass, display both positive (right) and negative (left) cornering
        a_y_sym = result.a_y_g
        a_x = result.a_x_g
        v_kmh = result.velocity_kmh

        # Scatter plot colored by vehicle speed
        scatter = ax.scatter(
            a_y_sym,
            a_x,
            c=v_kmh,
            cmap="plasma",
            s=12,
            alpha=0.75,
            edgecolors="none",
            label="Trajectory Nodes",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Speed [km/h]")

        # Theoretical envelopes at representative speeds
        if vehicle is not None:
            speeds_eval = [15.0, 45.0, 75.0]  # Low, Mid, High speed [m/s]
            colors = ["#7f8c8d", "#2980b9", "#e74c3c"]
            theta = np.linspace(0, 2 * np.pi, 200)

            for v_eval, col in zip(speeds_eval, colors):
                f_z = vehicle.total_vertical_load(v_eval)
                mu = vehicle.tire_friction_coefficient(f_z)
                f_max = mu * f_z
                max_g = f_max / (vehicle.params.mass * vehicle.params.g)

                ax.plot(
                    max_g * np.cos(theta),
                    max_g * np.sin(theta),
                    color=col,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Envelope @ {v_eval * 3.6:.0f} km/h ({max_g:.2f}g)",
                )

        ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
        ax.axvline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)

        max_limit = max(np.max(np.abs(result.a_y_g)), np.max(np.abs(result.a_x_g))) * 1.15
        max_limit = max(max_limit, 2.5)
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
        ax.set_aspect("equal", "box")

        ax.set_xlabel("Lateral Acceleration $a_y$ [g]")
        ax.set_ylabel("Longitudinal Acceleration $a_x$ [g]")
        ax.set_title("G-G Diagram (Tire Traction & Braking Envelope)")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

        return fig

    def plot_delta_time(
        self,
        result_base: SimulationResult,
        result_mod: SimulationResult,
        label_base: str = "Baseline",
        label_mod: str = "Modified",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot comparative delta-time analysis between two setup iterations.

        Parameters
        ----------
        result_base : SimulationResult
            Baseline simulation result.
        result_mod : SimulationResult
            Modified setup simulation result.
        label_base : str
            Name of baseline setup.
        label_mod : str
            Name of modified setup.
        save_path : Optional[str]
            Path to save figure.

        Returns
        -------
        plt.Figure
            Rendered Matplotlib figure.
        """
        fig, (ax_delta, ax_v, ax_dv) = plt.subplots(
            nrows=3, ncols=1, figsize=(13, 9), sharex=True, constrained_layout=True
        )

        s = result_base.distance
        t_base = result_base.time
        # Interpolate modified time to exact base distance nodes
        t_mod_interp = np.interp(s, result_mod.distance, result_mod.time)
        delta_t = t_mod_interp - t_base  # Negative = modified is faster

        v_base_kmh = result_base.velocity_kmh
        v_mod_kmh = np.interp(s, result_mod.distance, result_mod.velocity_kmh)
        delta_v_kmh = v_mod_kmh - v_base_kmh

        # ---------------------------------------------------------------------
        # Panel 1: Delta Time
        # ---------------------------------------------------------------------
        ax_delta.plot(s, delta_t, color="#8e44ad", linewidth=2.0)
        ax_delta.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax_delta.fill_between(
            s,
            delta_t,
            0.0,
            where=(delta_t <= 0.0),
            color="#27ae60",
            alpha=0.3,
            label=f"{label_mod} Faster",
        )
        ax_delta.fill_between(
            s,
            delta_t,
            0.0,
            where=(delta_t > 0.0),
            color="#e74c3c",
            alpha=0.3,
            label=f"{label_base} Faster",
        )
        ax_delta.set_ylabel("$\\Delta t$ [s]")
        lap_diff = result_mod.lap_time - result_base.lap_time
        sign_str = "-" if lap_diff < 0 else "+"
        ax_delta.set_title(
            f"Comparative Telemetry Delta: {label_mod} vs {label_base} (Lap Delta: {sign_str}{abs(lap_diff):.3f} s)"
        )
        ax_delta.legend(loc="upper left", framealpha=0.9)
        ax_delta.grid(True)

        # ---------------------------------------------------------------------
        # Panel 2: Velocity Overlay
        # ---------------------------------------------------------------------
        ax_v.plot(s, v_base_kmh, color="#34495e", linestyle="--", label=f"{label_base} ({result_base.lap_time:.3f}s)")
        ax_v.plot(s, v_mod_kmh, color="#e67e22", label=f"{label_mod} ({result_mod.lap_time:.3f}s)")
        ax_v.set_ylabel("Speed [km/h]")
        ax_v.legend(loc="lower right", framealpha=0.9)
        ax_v.grid(True)

        # ---------------------------------------------------------------------
        # Panel 3: Speed Difference Delta V
        # ---------------------------------------------------------------------
        ax_dv.plot(s, delta_v_kmh, color="#2980b9", linewidth=1.5)
        ax_dv.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax_dv.set_xlabel("Track Distance $s$ [m]")
        ax_dv.set_ylabel("$\\Delta v$ [km/h]")
        ax_dv.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

        return fig

    def plot_track_map(
        self,
        track_df: pd.DataFrame,
        result: SimulationResult,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Render 2D circuit centerline map colored by local vehicle velocity.

        Parameters
        ----------
        track_df : pd.DataFrame
            Track DataFrame with x and y coordinates.
        result : SimulationResult
            Simulation result with speed.
        save_path : Optional[str]
            Path to save figure.

        Returns
        -------
        plt.Figure
            Rendered Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

        x = track_df["x"].values
        y = track_df["y"].values
        v_kmh = result.velocity_kmh

        scatter = ax.scatter(x, y, c=v_kmh, cmap="turbo", s=18, edgecolors="none")
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Speed [km/h]")

        # Mark Start/Finish
        ax.plot(x[0], y[0], "r*", markersize=14, label="Start / Finish")

        # Mark Sector Points
        for sec_id, metric in result.sector_metrics.items():
            end_d = metric["end_distance_m"]
            idx = int(np.argmin(np.abs(result.distance - end_d)))
            ax.plot(x[idx], y[idx], "s", color="black", markersize=6)
            ax.text(x[idx] + 20, y[idx] + 20, f"S{sec_id}", fontsize=10, fontweight="bold")

        ax.set_aspect("equal", "box")
        ax.set_xlabel("Centerline X [m]")
        ax.set_ylabel("Centerline Y [m]")
        ax.set_title("Track Layout Colored by Velocity Profile")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

        return fig
