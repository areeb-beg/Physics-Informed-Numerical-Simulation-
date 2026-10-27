"""Physics and Numerical Integrity Unit Tests.

Validates:
1. Work-Energy theorem and conservation of energy.
2. Steady-state lateral grip and cornering limits.
3. Aerodynamic drag, downforce, and tire load sensitivity.
4. Combined slip friction ellipse bounds.
5. Numerical solver stability and periodic boundary conditions.
6. Relational telemetry schema and database pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure both lap_time_simulator directory and parent directory are on sys.path
FILE_DIR = Path(__file__).resolve().parent
SIM_ROOT = FILE_DIR.parent
PROJECT_ROOT = SIM_ROOT.parent

for p in [str(PROJECT_ROOT), str(SIM_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import pytest

try:
    from lap_time_simulator.src.vehicle_model import VehicleModel, VehicleParameters
    from lap_time_simulator.src.solver import LapTimeSolver
    from lap_time_simulator.src.telemetry_db import TelemetryDatabase
except ImportError:
    from src.vehicle_model import VehicleModel, VehicleParameters
    from src.solver import LapTimeSolver
    from src.telemetry_db import TelemetryDatabase


@pytest.fixture
def standard_vehicle() -> VehicleModel:
    """Fixture providing a standard F1-class vehicle model."""
    params = VehicleParameters(
        mass=800.0,
        g=9.80665,
        rho=1.225,
        frontal_area=1.60,
        cd=0.90,
        cl=3.20,
        power_max=745000.0,
        f_brake_max=30000.0,
        mu_0=1.70,
        load_sensitivity=0.15,
        crr=0.015,
        drive_efficiency=0.92,
        v_max_mech=105.0,
        v_min=1.0,
    )
    return VehicleModel(params)


@pytest.fixture
def straight_track() -> pd.DataFrame:
    """Fixture providing a pure straight 1000m track (curvature = 0)."""
    s = np.linspace(0.0, 1000.0, 501)  # ds = 2.0 m
    return pd.DataFrame(
        {
            "distance": s,
            "x": s,
            "y": np.zeros_like(s),
            "curvature": np.zeros_like(s),
            "sector": np.ones_like(s, dtype=int),
        }
    )


class TestVehicleModelPhysics:
    """Validates vehicle dynamics equations and physics constraints."""

    def test_aerodynamic_quadratic_scaling(self, standard_vehicle: VehicleModel) -> None:
        """Verify drag and downforce scale strictly quadratically with speed."""
        v1 = 20.0
        v2 = 40.0  # Speed doubles -> forces must quadruple

        f_d1 = standard_vehicle.aerodynamic_drag(v1)
        f_d2 = standard_vehicle.aerodynamic_drag(v2)
        assert np.isclose(f_d2 / f_d1, 4.0, rtol=1e-5)

        f_l1 = standard_vehicle.aerodynamic_downforce(v1)
        f_l2 = standard_vehicle.aerodynamic_downforce(v2)
        assert np.isclose(f_l2 / f_l1, 4.0, rtol=1e-5)

    def test_static_weight_and_normal_force(self, standard_vehicle: VehicleModel) -> None:
        """Verify vertical load equals gravitational weight at rest and increases with speed."""
        w0 = standard_vehicle.params.mass * standard_vehicle.params.g
        f_z_rest = standard_vehicle.total_vertical_load(0.0)
        assert np.isclose(f_z_rest, w0, rtol=1e-5)

        v = 60.0  # High speed
        f_z_high = standard_vehicle.total_vertical_load(v)
        expected_downforce = 0.5 * standard_vehicle.params.rho * standard_vehicle.params.cl * standard_vehicle.params.frontal_area * (v**2)
        assert np.isclose(f_z_high, w0 + expected_downforce, rtol=1e-5)

    def test_tire_load_sensitivity_monotonicity(self, standard_vehicle: VehicleModel) -> None:
        """Verify effective friction coefficient mu decreases monotonically with normal load."""
        w0 = standard_vehicle.static_weight
        loads = np.linspace(w0, 3.0 * w0, 10)
        mus = standard_vehicle.tire_friction_coefficient(loads)

        # Monotonically non-increasing
        diffs = np.diff(mus)
        assert np.all(diffs <= 0.0), "Tire friction coefficient must decrease with increasing load."

    def test_friction_ellipse_combined_slip(self, standard_vehicle: VehicleModel) -> None:
        """Verify tractive and lateral forces strictly respect Kamm's friction ellipse."""
        v = 35.0
        f_z = standard_vehicle.total_vertical_load(v)
        mu = standard_vehicle.tire_friction_coefficient(f_z)
        f_tire_peak = mu * f_z

        # Test varying lateral acceleration demands
        ay_vals = np.linspace(0.0, 0.95 * (f_tire_peak / standard_vehicle.params.mass), 10)
        for ay in ay_vals:
            # Longitudinal acceleration available
            ax = standard_vehicle.max_longitudinal_acceleration(v, a_y=ay)
            fy = standard_vehicle.params.mass * ay

            # Compute actual tire drive force
            f_drag = standard_vehicle.aerodynamic_drag(v)
            f_rr = standard_vehicle.params.crr * standard_vehicle.static_weight
            fx = standard_vehicle.params.mass * ax + f_drag + f_rr

            # Check normalized ellipse: (fx / f_peak)^2 + (fy / f_peak)^2 <= 1.0 + epsilon
            norm_ellipse = (fx / f_tire_peak) ** 2 + (fy / f_tire_peak) ** 2
            assert norm_ellipse <= 1.0001, f"Tire grip ellipse violated: {norm_ellipse} > 1.0"

    def test_terminal_velocity_force_equilibrium(self, standard_vehicle: VehicleModel) -> None:
        """Verify terminal velocity satisfies engine tractive force = resisting forces."""
        v_term = standard_vehicle.terminal_velocity()
        f_engine = (standard_vehicle.params.drive_efficiency * standard_vehicle.params.power_max) / v_term
        f_drag = standard_vehicle.aerodynamic_drag(v_term)
        f_rr = standard_vehicle.params.crr * standard_vehicle.static_weight

        # Net acceleration at terminal velocity must be approximately zero
        assert np.isclose(f_engine, f_drag + f_rr, rtol=1e-3)


class TestConservationOfEnergy:
    """Validates thermodynamic work-energy theorem and kinematics."""

    def test_work_energy_theorem_in_acceleration(
        self, standard_vehicle: VehicleModel, straight_track: pd.DataFrame
    ) -> None:
        """Verify Delta E_k = Integral F_net ds on a straight line."""
        solver = LapTimeSolver(standard_vehicle, method="rk4")
        result = solver.solve(straight_track, mode="standing", v_start=15.0)

        # Kinetic energy change: Delta E_k = 0.5 * m * (v_f^2 - v_i^2)
        m = standard_vehicle.params.mass
        v_i = result.velocity[0]
        v_f = result.velocity[-1]
        delta_ek = 0.5 * m * (v_f**2 - v_i**2)

        # Net work: W_net = Integral (m * a_x) ds
        # Using trapezoidal rule on discrete distance increments
        ds = np.diff(result.distance)
        f_net = m * result.a_x
        f_net_mid = 0.5 * (f_net[:-1] + f_net[1:])
        work_net = np.sum(f_net_mid * ds)

        # Work-energy theorem equivalence within spatial discretization error (< 1.5%)
        assert np.isclose(delta_ek, work_net, rtol=0.015), (
            f"Work-energy theorem failed: Delta E_k = {delta_ek:.1f} J vs Work = {work_net:.1f} J"
        )


class TestSolverCorneringAndConvergence:
    """Validates QSS solver constraints and numerical stability."""

    def test_cornering_velocity_envelope_bound(
        self, standard_vehicle: VehicleModel
    ) -> None:
        """Verify solved velocity profile never exceeds maximum cornering limit."""
        # Synthetic technical track with varied radii
        s = np.linspace(0.0, 2000.0, 1001)
        # S-curve curvature
        kappa = 0.025 * np.abs(np.sin(2 * np.pi * s / 500.0))
        track_df = pd.DataFrame(
            {
                "distance": s,
                "x": s,
                "y": np.zeros_like(s),
                "curvature": kappa,
                "sector": np.ones_like(s, dtype=int),
            }
        )

        solver = LapTimeSolver(standard_vehicle, method="rk4")
        result = solver.solve(track_df, mode="flying")

        # Check: v(s) <= v_corner(s) + epsilon everywhere
        tolerance = 1e-3
        excess = result.velocity - result.v_corner
        assert np.all(excess <= tolerance), f"Max cornering limit breached by {np.max(excess)} m/s"

    def test_rk4_vs_euler_consistency(
        self, standard_vehicle: VehicleModel, straight_track: pd.DataFrame
    ) -> None:
        """Verify RK4 and Euler integrators produce consistent lap results."""
        solver_rk4 = LapTimeSolver(standard_vehicle, method="rk4")
        solver_euler = LapTimeSolver(standard_vehicle, method="euler")

        res_rk4 = solver_rk4.solve(straight_track, mode="standing", v_start=10.0)
        res_euler = solver_euler.solve(straight_track, mode="standing", v_start=10.0)

        # Lap times should agree to within 0.5% on fine spatial discretization
        assert np.isclose(res_rk4.lap_time, res_euler.lap_time, rtol=0.005)


class TestTelemetryDatabasePipeline:
    """Validates relational database ingestion and schema integrity."""

    def test_db_initialization_and_logging(
        self, standard_vehicle: VehicleModel, straight_track: pd.DataFrame
    ) -> None:
        """Verify database schema creation, run persistence, and query pipeline."""
        solver = LapTimeSolver(standard_vehicle, method="rk4")
        result = solver.solve(straight_track, mode="standing", v_start=20.0)

        db = TelemetryDatabase(use_fallback=True)
        connected = db.connect()
        assert connected is True
        assert db.is_connected is True

        # Log simulation run
        run_id = db.log_simulation(
            result=result,
            params=standard_vehicle.params,
            vehicle_name="Test-F1-Physics",
            track_name="Straight 1000m",
            solver_method="rk4",
            mode="standing",
        )
        assert run_id > 0

        # Query all runs
        runs_df = db.query_all_runs()
        assert len(runs_df) >= 1
        assert "lap_time_s" in runs_df.columns

        # Query run details
        details = db.query_run_details(run_id)
        assert details["run"]["run_id"] == run_id
        assert np.isclose(details["run"]["lap_time_s"], result.lap_time, rtol=1e-3)
        assert details["parameters"]["mass_kg"] == standard_vehicle.params.mass

        db.close()
