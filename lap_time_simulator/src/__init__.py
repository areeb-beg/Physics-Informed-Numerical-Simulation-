"""Lap Time Simulator and Telemetry Analyzer Package."""

try:
    from lap_time_simulator.src.vehicle_model import VehicleModel, VehicleParameters
    from lap_time_simulator.src.solver import LapTimeSolver, SimulationResult
    from lap_time_simulator.src.telemetry_db import TelemetryDatabase
    from lap_time_simulator.src.visualizer import LapVisualizer
except ImportError:
    from src.vehicle_model import VehicleModel, VehicleParameters
    from src.solver import LapTimeSolver, SimulationResult
    from src.telemetry_db import TelemetryDatabase
    from src.visualizer import LapVisualizer

__all__ = [
    "VehicleModel",
    "VehicleParameters",
    "LapTimeSolver",
    "SimulationResult",
    "TelemetryDatabase",
    "LapVisualizer",
]
