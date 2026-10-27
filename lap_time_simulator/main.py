"""Production CLI Entry Point for Lap Time Simulator & Telemetry Analyzer.

Executes the end-to-end computational pipeline:
1. Loads and validates track centerline geometry.
2. Formulates point-mass vehicle dynamics and tire-aero limits.
3. Solves quasi-steady-state lap trajectory via RK4/Euler spatial integration.
4. Generates publication-grade telemetry visualizations.
5. Ingests simulation runs and sector breakdowns into relational database (MySQL/SQLite).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to sys.path to enable absolute imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from lap_time_simulator.src.solver import LapTimeSolver, SimulationResult
from lap_time_simulator.src.telemetry_db import TelemetryDatabase
from lap_time_simulator.src.vehicle_model import VehicleModel, VehicleParameters
from lap_time_simulator.src.visualizer import LapVisualizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production Point-Mass Lap Time Simulator & Telemetry Analyzer"
    )

    # Track and Simulation Options
    parser.add_argument(
        "--track",
        type=str,
        default="",
        help="Path to track centerline CSV (defaults to sample_track.csv)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["flying", "standing"],
        default="flying",
        help="Simulation boundary mode: 'flying' (periodic) or 'standing' start",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["rk4", "euler"],
        default="rk4",
        help="Numerical spatial integration scheme: 'rk4' (default) or 'euler'",
    )
    parser.add_argument(
        "--v-start",
        type=float,
        default=0.0,
        help="Initial velocity for standing start [m/s]",
    )

    # Vehicle Parameter Overrides
    parser.add_argument("--mass", type=float, default=798.0, help="Vehicle mass [kg]")
    parser.add_argument("--power", type=float, default=745.0, help="Engine power [kW]")
    parser.add_argument("--downforce", type=float, default=3.20, help="Lift/Downforce coefficient C_L")
    parser.add_argument("--drag", type=float, default=0.90, help="Drag coefficient C_D")
    parser.add_argument("--mu", type=float, default=1.70, help="Peak tire friction coefficient mu_0")
    parser.add_argument("--load-sens", type=float, default=0.15, help="Tire load sensitivity factor")

    # Comparative Study Mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Execute a comparative sensitivity analysis (Baseline vs High Downforce Setup)",
    )

    # Output & Visualization Options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated plots and telemetry CSV",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable graphic plot generation",
    )

    # Database Options
    parser.add_argument("--db-host", type=str, default="localhost", help="MySQL database host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL database port")
    parser.add_argument("--db-user", type=str, default="root", help="MySQL database user")
    parser.add_argument("--db-pass", type=str, default="", help="MySQL database password")
    parser.add_argument("--db-name", type=str, default="lap_telemetry", help="MySQL database schema")
    parser.add_argument("--no-db", action="store_true", help="Skip database ingestion")

    return parser.parse_args()


def resolve_track_path(specified_path: str) -> str:
    """Locate sample track CSV across project tree locations."""
    if specified_path and os.path.exists(specified_path):
        return specified_path

    candidates = [
        os.path.join(CURRENT_DIR, "data", "sample_track.csv"),
        os.path.join(CURRENT_DIR.parent, "data", "sample_track.csv"),
        os.path.join("data", "sample_track.csv"),
        os.path.join("lap_time_simulator", "data", "sample_track.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Track file not found. Please provide a valid CSV via --track. Checked: {candidates}"
    )


def print_banner() -> None:
    """Print ASCII header banner."""
    banner = r"""
================================================================================
   POINT-MASS LAP TIME SIMULATOR & TELEMETRY ANALYZER (QSS RK4)
   Physics-Informed Vehicle Dynamics & Curvilinear Space Integration
================================================================================
"""
    print(banner)


def run_pipeline() -> int:
    """Execute main execution workflow."""
    args = parse_arguments()
    print_banner()

    # 1. Resolve and Load Track
    track_path = resolve_track_path(args.track)
    print(f"[*] Loading track geometry from: {track_path}")
    track_df = pd.read_csv(track_path)
    total_distance = float(track_df["distance"].iloc[-1])
    print(f"    - Total Nodes   : {len(track_df)}")
    print(f"    - Track Length  : {total_distance:.2f} m ({total_distance / 1000.0:.2f} km)")
    print(f"    - Curvature Max : {track_df['curvature'].max():.5f} m^-1 (R_min = {1.0 / max(track_df['curvature'].max(), 1e-6):.1f} m)")

    # 2. Formulate Baseline Vehicle Model
    params_base = VehicleParameters(
        mass=args.mass,
        power_max=args.power * 1000.0,
        cl=args.downforce,
        cd=args.drag,
        mu_0=args.mu,
        load_sensitivity=args.load_sens,
    )
    vehicle_base = VehicleModel(params_base)
    print("\n[*] Initializing Vehicle Dynamics Model (Baseline):")
    print(f"    - Mass          : {params_base.mass:.1f} kg")
    print(f"    - Power Output  : {params_base.power_max / 1000.0:.1f} kW ({params_base.power_max / 745.7:.0f} hp)")
    print(f"    - Aerodynamics  : C_L = {params_base.cl:.2f} | C_D = {params_base.cd:.2f} | L/D = {params_base.cl / params_base.cd:.2f}")
    print(f"    - Tire Model    : mu_0 = {params_base.mu_0:.2f} | Load Sens = {params_base.load_sensitivity:.2f}")
    print(f"    - Terminal Vel  : {vehicle_base.terminal_velocity() * 3.6:.1f} km/h")

    # 3. Instantiate Solver & Execute Baseline Simulation
    solver_base = LapTimeSolver(vehicle_base, method=args.solver)
    print(f"\n[*] Solving Lap Trajectory [Scheme: {args.solver.upper()} | Mode: {args.mode.upper()}]...")
    result_base = solver_base.solve(track_df, mode=args.mode, v_start=args.v_start)

    # 4. Display Simulation Telemetry
    print("\n" + result_base.summary())

    # 5. Execute Comparative Analysis if requested
    result_mod: Optional[SimulationResult] = None
    vehicle_mod: Optional[VehicleModel] = None
    if args.compare:
        print("\n[*] Executing Comparative Setup Simulation (High Downforce Setup):")
        params_mod = VehicleParameters(
            mass=args.mass,
            power_max=args.power * 1000.0,
            cl=args.downforce * 1.30,  # +30% Downforce
            cd=args.drag * 1.15,       # +15% Drag penalty
            mu_0=args.mu,
            load_sensitivity=args.load_sens,
        )
        vehicle_mod = VehicleModel(params_mod)
        solver_mod = LapTimeSolver(vehicle_mod, method=args.solver)
        result_mod = solver_mod.solve(track_df, mode=args.mode, v_start=args.v_start)

        delta_time = result_mod.lap_time - result_base.lap_time
        print("=" * 60)
        print(" COMPARATIVE SETUP DELTA (High Downforce vs Baseline)")
        print("=" * 60)
        print(f" Baseline Lap Time : {result_base.lap_time:6.3f} s (Top: {result_base.top_speed_kmh:.1f} km/h)")
        print(f" High-DF Lap Time  : {result_mod.lap_time:6.3f} s (Top: {result_mod.top_speed_kmh:.1f} km/h)")
        sign = "-" if delta_time < 0 else "+"
        faster_tag = "FASTER" if delta_time < 0 else "SLOWER"
        print(f" Net Delta Time    : {sign}{abs(delta_time):.3f} s ({faster_tag})")
        print("=" * 60)

    # 6. Generate Telemetry Visualizations
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not args.no_plots:
        print(f"\n[*] Rendering Telemetry Visualizations into: {out_dir}")
        visualizer = LapVisualizer()

        # Velocity profile plot
        p_vel = os.path.join(out_dir, "velocity_profile.png")
        visualizer.plot_velocity_profile(result_base, save_path=p_vel, title_suffix="(Baseline)")
        print(f"    -> [Saved] Velocity Profile  : {p_vel}")

        # G-G diagram
        p_gg = os.path.join(out_dir, "gg_diagram.png")
        visualizer.plot_gg_diagram(result_base, vehicle=vehicle_base, save_path=p_gg)
        print(f"    -> [Saved] G-G Diagram       : {p_gg}")

        # Track map
        p_map = os.path.join(out_dir, "track_map.png")
        visualizer.plot_track_map(track_df, result_base, save_path=p_map)
        print(f"    -> [Saved] 2D Track Map      : {p_map}")

        # Delta-time comparison plot if comparison mode enabled
        if result_mod is not None:
            p_delta = os.path.join(out_dir, "delta_time.png")
            visualizer.plot_delta_time(
                result_base=result_base,
                result_mod=result_mod,
                label_base="Baseline Aero",
                label_mod="High Downforce Aero",
                save_path=p_delta,
            )
            print(f"    -> [Saved] Delta-Time Plot   : {p_delta}")

    # 7. Export Telemetry CSV
    telemetry_csv_path = os.path.join(out_dir, "telemetry_output.csv")
    result_base.to_dataframe().to_csv(telemetry_csv_path, index=False)
    print(f"\n[*] Exported High-Resolution Telemetry CSV to: {telemetry_csv_path}")

    # 8. Relational Database Ingestion Pipeline
    if not args.no_db:
        print("\n[*] Relational Telemetry Database Pipeline:")
        db = TelemetryDatabase(
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_pass,
            database=args.db_name,
            use_fallback=True,
        )
        if db.connect():
            run_id = db.log_simulation(
                result=result_base,
                params=params_base,
                vehicle_name="Baseline Formula Car",
                track_name="Grand Prix Circuit",
                solver_method=args.solver,
                mode=args.mode,
            )
            print(f"    - Baseline Run Persisted with ID: {run_id} (Backend: {db.backend})")

            if result_mod is not None and vehicle_mod is not None:
                mod_run_id = db.log_simulation(
                    result=result_mod,
                    params=vehicle_mod.params,
                    vehicle_name="High-Downforce Formula Car",
                    track_name="Grand Prix Circuit",
                    solver_method=args.solver,
                    mode=args.mode,
                )
                print(f"    - Comparative Run Persisted with ID: {mod_run_id} (Backend: {db.backend})")

            # Query Leaderboard
            runs_df = db.query_all_runs()
            print(f"\n[*] Database Simulation Leaderboard ({len(runs_df)} total records):")
            print(runs_df[["run_id", "vehicle_name", "lap_time_s", "top_speed_kmh", "avg_speed_kmh"]].to_string(index=False))
            db.close()
        else:
            print("    [!] Database logging bypassed (connection failed).")

    print("\n================================================================================")
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("================================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
