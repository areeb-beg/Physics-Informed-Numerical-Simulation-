"""MySQL Telemetry Database Wrapper and Ingestion Pipeline.

Manages relational persistence of simulation runs, vehicle setup parameters,
and sector breakdown metrics with full transactional integrity.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from lap_time_simulator.src.solver import SimulationResult
    from lap_time_simulator.src.vehicle_model import VehicleParameters
except ImportError:
    from src.solver import SimulationResult
    from src.vehicle_model import VehicleParameters

logger = logging.getLogger("TelemetryDB")


class TelemetryDatabase:
    """Relational database wrapper for telemetry ingestion and querying.

    Supports native MySQL with automatic fallback to an in-memory/file relational
    engine (SQLite) for environments where the MySQL daemon is not running.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "lap_telemetry",
        use_fallback: bool = True,
    ) -> None:
        """Initialize database connection parameters.

        Parameters
        ----------
        host : str
            MySQL server host.
        port : int
            MySQL server TCP port.
        user : str
            Database user account.
        password : str
            Database user password.
        database : str
            Database schema name.
        use_fallback : bool
            Whether to fall back to an embedded relational engine if MySQL server
            is unreachable.
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.use_fallback = use_fallback
        self.is_connected = False
        self.backend: str = "none"

        self._mysql_conn = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None

    def connect(self) -> bool:
        """Establish connection to MySQL server, or initialize fallback engine.

        Returns
        -------
        bool
            True if connection established, False otherwise.
        """
        # Try native MySQL connector first
        try:
            import mysql.connector
            from mysql.connector import Error

            logger.info("Attempting connection to MySQL server at %s:%d...", self.host, self.port)
            # First connect to server without database to ensure database exists
            server_conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                connection_timeout=3,
            )
            cursor = server_conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.close()
            server_conn.close()

            # Now connect to specific database
            self._mysql_conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            self.backend = "mysql"
            self.is_connected = True
            logger.info("Successfully connected to MySQL database: %s", self.database)
            self.init_schema()
            return True

        except Exception as err:
            logger.warning("MySQL connection could not be established: %s", err)
            if self.use_fallback:
                logger.info("Initializing embedded relational fallback engine...")
                self._sqlite_conn = sqlite3.connect(":memory:")
                self._sqlite_conn.execute("PRAGMA foreign_keys = ON")
                self.backend = "sqlite_fallback"
                self.is_connected = True
                self.init_schema()
                logger.info("Fallback relational engine active (in-memory mode).")
                return True
            else:
                self.is_connected = False
                return False

    def close(self) -> None:
        """Close database connections and release resources."""
        if self._mysql_conn is not None:
            try:
                self._mysql_conn.close()
            except Exception:
                pass
            self._mysql_conn = None

        if self._sqlite_conn is not None:
            try:
                self._sqlite_conn.close()
            except Exception:
                pass
            self._sqlite_conn = None

        self.is_connected = False
        self.backend = "none"

    def init_schema(self) -> None:
        """Create normalized relational schema for telemetry logging."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected. Call connect() first.")

        if self.backend == "mysql":
            ddl_sim_runs = """
            CREATE TABLE IF NOT EXISTS sim_runs (
                run_id INT AUTO_INCREMENT PRIMARY KEY,
                run_uuid VARCHAR(64) UNIQUE NOT NULL,
                timestamp DATETIME NOT NULL,
                vehicle_name VARCHAR(128) NOT NULL,
                track_name VARCHAR(128) NOT NULL,
                lap_time_s DOUBLE NOT NULL,
                top_speed_kmh DOUBLE NOT NULL,
                avg_speed_kmh DOUBLE NOT NULL,
                solver_method VARCHAR(32) NOT NULL,
                mode VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL
            ) ENGINE=InnoDB;
            """

            ddl_vehicle_params = """
            CREATE TABLE IF NOT EXISTS vehicle_params (
                param_id INT AUTO_INCREMENT PRIMARY KEY,
                run_id INT NOT NULL,
                mass_kg DOUBLE NOT NULL,
                power_w DOUBLE NOT NULL,
                cd DOUBLE NOT NULL,
                cl DOUBLE NOT NULL,
                frontal_area_m2 DOUBLE NOT NULL,
                mu_0 DOUBLE NOT NULL,
                load_sensitivity DOUBLE NOT NULL,
                crr DOUBLE NOT NULL,
                drive_efficiency DOUBLE NOT NULL,
                FOREIGN KEY (run_id) REFERENCES sim_runs(run_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """

            ddl_sector_times = """
            CREATE TABLE IF NOT EXISTS sector_times (
                sector_id INT AUTO_INCREMENT PRIMARY KEY,
                run_id INT NOT NULL,
                sector_number INT NOT NULL,
                start_distance_m DOUBLE NOT NULL,
                end_distance_m DOUBLE NOT NULL,
                sector_time_s DOUBLE NOT NULL,
                entry_speed_kmh DOUBLE NOT NULL,
                exit_speed_kmh DOUBLE NOT NULL,
                v_max_kmh DOUBLE NOT NULL,
                v_min_kmh DOUBLE NOT NULL,
                FOREIGN KEY (run_id) REFERENCES sim_runs(run_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """
            cursor = self._mysql_conn.cursor()
            cursor.execute(ddl_sim_runs)
            cursor.execute(ddl_vehicle_params)
            cursor.execute(ddl_sector_times)
            self._mysql_conn.commit()
            cursor.close()

        elif self.backend == "sqlite_fallback":
            ddl_sim_runs = """
            CREATE TABLE IF NOT EXISTS sim_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_uuid TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                vehicle_name TEXT NOT NULL,
                track_name TEXT NOT NULL,
                lap_time_s REAL NOT NULL,
                top_speed_kmh REAL NOT NULL,
                avg_speed_kmh REAL NOT NULL,
                solver_method TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """

            ddl_vehicle_params = """
            CREATE TABLE IF NOT EXISTS vehicle_params (
                param_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                mass_kg REAL NOT NULL,
                power_w REAL NOT NULL,
                cd REAL NOT NULL,
                cl REAL NOT NULL,
                frontal_area_m2 REAL NOT NULL,
                mu_0 REAL NOT NULL,
                load_sensitivity REAL NOT NULL,
                crr REAL NOT NULL,
                drive_efficiency REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES sim_runs(run_id) ON DELETE CASCADE
            );
            """

            ddl_sector_times = """
            CREATE TABLE IF NOT EXISTS sector_times (
                sector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                sector_number INTEGER NOT NULL,
                start_distance_m REAL NOT NULL,
                end_distance_m REAL NOT NULL,
                sector_time_s REAL NOT NULL,
                entry_speed_kmh REAL NOT NULL,
                exit_speed_kmh REAL NOT NULL,
                v_max_kmh REAL NOT NULL,
                v_min_kmh REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES sim_runs(run_id) ON DELETE CASCADE
            );
            """
            cursor = self._sqlite_conn.cursor()
            cursor.execute(ddl_sim_runs)
            cursor.execute(ddl_vehicle_params)
            cursor.execute(ddl_sector_times)
            self._sqlite_conn.commit()
            cursor.close()

    def log_simulation(
        self,
        result: SimulationResult,
        params: VehicleParameters,
        vehicle_name: str = "Formula F1-Spec",
        track_name: str = "Grand Prix Circuit",
        solver_method: str = "rk4",
        mode: str = "flying",
    ) -> int:
        """Persist a complete simulation run and parameters into database tables.

        Parameters
        ----------
        result : SimulationResult
            Computed simulation result object.
        params : VehicleParameters
            Vehicle parameter dataclass used for the run.
        vehicle_name : str
            Identifier label for the vehicle configuration.
        track_name : str
            Identifier label for the track.
        solver_method : str
            Solver algorithm ('rk4' or 'euler').
        mode : str
            Simulation mode ('flying' or 'standing').

        Returns
        -------
        int
            Inserted primary key `run_id`.
        """
        if not self.is_connected:
            self.connect()

        run_uuid = str(uuid.uuid4())
        now_dt = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        insert_run_sql = """
        INSERT INTO sim_runs (
            run_uuid, timestamp, vehicle_name, track_name, lap_time_s,
            top_speed_kmh, avg_speed_kmh, solver_method, mode, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        insert_params_sql = """
        INSERT INTO vehicle_params (
            run_id, mass_kg, power_w, cd, cl, frontal_area_m2,
            mu_0, load_sensitivity, crr, drive_efficiency
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        insert_sector_sql = """
        INSERT INTO sector_times (
            run_id, sector_number, start_distance_m, end_distance_m, sector_time_s,
            entry_speed_kmh, exit_speed_kmh, v_max_kmh, v_min_kmh
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        if self.backend == "mysql":
            insert_run_sql = insert_run_sql.replace("?", "%s")
            insert_params_sql = insert_params_sql.replace("?", "%s")
            insert_sector_sql = insert_sector_sql.replace("?", "%s")

            cursor = self._mysql_conn.cursor()
            try:
                # 1. Insert sim run
                cursor.execute(
                    insert_run_sql,
                    (
                        run_uuid,
                        now_dt,
                        vehicle_name,
                        track_name,
                        float(result.lap_time),
                        float(result.top_speed_kmh),
                        float(result.avg_speed_kmh),
                        solver_method,
                        mode,
                        "COMPLETED",
                    ),
                )
                run_id = cursor.lastrowid

                # 2. Insert vehicle setup parameters
                cursor.execute(
                    insert_params_sql,
                    (
                        run_id,
                        params.mass,
                        params.power_max,
                        params.cd,
                        params.cl,
                        params.frontal_area,
                        params.mu_0,
                        params.load_sensitivity,
                        params.crr,
                        params.drive_efficiency,
                    ),
                )

                # 3. Insert sector breakdown
                for sec_num, sec_data in result.sector_metrics.items():
                    cursor.execute(
                        insert_sector_sql,
                        (
                            run_id,
                            int(sec_num),
                            float(sec_data.get("start_distance_m", 0.0)),
                            float(sec_data.get("end_distance_m", 0.0)),
                            float(sec_data.get("sector_time_s", 0.0)),
                            float(sec_data.get("entry_speed_kmh", 0.0)),
                            float(sec_data.get("exit_speed_kmh", 0.0)),
                            float(sec_data.get("v_max_kmh", 0.0)),
                            float(sec_data.get("v_min_kmh", 0.0)),
                        ),
                    )

                self._mysql_conn.commit()
                return int(run_id)
            except Exception as e:
                self._mysql_conn.rollback()
                raise e
            finally:
                cursor.close()

        else:  # sqlite fallback
            cursor = self._sqlite_conn.cursor()
            try:
                cursor.execute(
                    insert_run_sql,
                    (
                        run_uuid,
                        now_dt,
                        vehicle_name,
                        track_name,
                        float(result.lap_time),
                        float(result.top_speed_kmh),
                        float(result.avg_speed_kmh),
                        solver_method,
                        mode,
                        "COMPLETED",
                    ),
                )
                run_id = cursor.lastrowid

                cursor.execute(
                    insert_params_sql,
                    (
                        run_id,
                        params.mass,
                        params.power_max,
                        params.cd,
                        params.cl,
                        params.frontal_area,
                        params.mu_0,
                        params.load_sensitivity,
                        params.crr,
                        params.drive_efficiency,
                    ),
                )

                for sec_num, sec_data in result.sector_metrics.items():
                    cursor.execute(
                        insert_sector_sql,
                        (
                            run_id,
                            int(sec_num),
                            float(sec_data.get("start_distance_m", 0.0)),
                            float(sec_data.get("end_distance_m", 0.0)),
                            float(sec_data.get("sector_time_s", 0.0)),
                            float(sec_data.get("entry_speed_kmh", 0.0)),
                            float(sec_data.get("exit_speed_kmh", 0.0)),
                            float(sec_data.get("v_max_kmh", 0.0)),
                            float(sec_data.get("v_min_kmh", 0.0)),
                        ),
                    )

                self._sqlite_conn.commit()
                return int(run_id)
            except Exception as e:
                self._sqlite_conn.rollback()
                raise e
            finally:
                cursor.close()

    def query_all_runs(self) -> pd.DataFrame:
        """Fetch all logged simulation runs into a Pandas DataFrame."""
        if not self.is_connected:
            self.connect()

        query = """
        SELECT run_id, run_uuid, timestamp, vehicle_name, track_name,
               lap_time_s, top_speed_kmh, avg_speed_kmh, solver_method, mode
        FROM sim_runs
        ORDER BY lap_time_s ASC
        """
        if self.backend == "mysql":
            cursor = self._mysql_conn.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            return pd.DataFrame(rows)
        else:
            return pd.read_sql_query(query, self._sqlite_conn)

    def query_run_details(self, run_id: int) -> Dict[str, Any]:
        """Fetch complete details (run info, vehicle parameters, and sector breakdown).

        Parameters
        ----------
        run_id : int
            Primary key ID of the run.

        Returns
        -------
        Dict[str, Any]
            Nested dictionary with run, parameters, and sector list.
        """
        if not self.is_connected:
            self.connect()

        run_q = "SELECT * FROM sim_runs WHERE run_id = ?"
        param_q = "SELECT * FROM vehicle_params WHERE run_id = ?"
        sector_q = "SELECT * FROM sector_times WHERE run_id = ? ORDER BY sector_number ASC"

        if self.backend == "mysql":
            run_q = run_q.replace("?", "%s")
            param_q = param_q.replace("?", "%s")
            sector_q = sector_q.replace("?", "%s")
            cursor = self._mysql_conn.cursor(dictionary=True)
            cursor.execute(run_q, (run_id,))
            run_row = cursor.fetchone()
            cursor.execute(param_q, (run_id,))
            param_row = cursor.fetchone()
            cursor.execute(sector_q, (run_id,))
            sector_rows = cursor.fetchall()
            cursor.close()
        else:
            self._sqlite_conn.row_factory = sqlite3.Row
            cursor = self._sqlite_conn.cursor()
            cursor.execute(run_q, (run_id,))
            r = cursor.fetchone()
            run_row = dict(r) if r else {}
            cursor.execute(param_q, (run_id,))
            p = cursor.fetchone()
            param_row = dict(p) if p else {}
            cursor.execute(sector_q, (run_id,))
            sector_rows = [dict(s) for s in cursor.fetchall()]
            cursor.close()

        return {
            "run": run_row,
            "parameters": param_row,
            "sectors": sector_rows,
        }
