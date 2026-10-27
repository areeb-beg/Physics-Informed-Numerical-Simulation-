"""Root Entry Point for Lap Time Simulator & Telemetry Analyzer."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lap_time_simulator.main import run_pipeline

if __name__ == "__main__":
    sys.exit(run_pipeline())
