# Physics-Informed Numerical Lap Time Simulator & Telemetry Analyzer

A production-grade, modular, first-principles Point-Mass Lap Time Simulator and Telemetry Ingestion Pipeline written in Python, NumPy, Pandas, Matplotlib, and MySQL (`mysql-connector-python`).

---

## 1. System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                  SYSTEM ARCHITECTURE PIPELINE                                     |
+---------------------------------------------------------------------------------------------------+

      [ Track Centerline Coordinates ]
         (x, y, curvature kappa, distance s, sector)
                    |
                    v
    +-------------------------------+
    |   VEHICLE DYNAMICS MODEL      |<------------------ [ Vehicle Setup Parameters ]
    |  - Aerodynamic Drag F_d(v)    |                    - Mass m, Power P_max
    |  - Aero Downforce F_down(v)   |                    - Drag C_D, Downforce C_L
    |  - Normal Load F_z(v)         |                    - Friction mu_0, Load Sens k_load
    |  - Tire Load Sensitivity mu   |                    - Rolling Resistance C_rr
    |  - Kamm's Friction Ellipse    |                    - Gearing Top Speed v_max
    +-------------------------------+
                    |
                    | Closed-form apex velocities & traction/braking derivatives
                    v
    +-------------------------------+
    |    QUASI-STEADY-STATE SOLVER  |
    |-------------------------------|
    | Step 1: Apex Cornering Limits |  v_corner(s) = f(kappa(s), F_z, mu(F_z))
    | Step 2: Forward Accel Pass    |  v_{i+1} via RK4 / Euler on d(v^2)/ds = 2*a_x
    | Step 3: Backward Braking Pass |  v_i via RK4 / Euler on d(v^2)/ds = -2*|a_brake|
    | Periodic Lap Continuity       |  Flying lap cycle convergence v(0) == v(end)
    +-------------------------------+
                    |
                    | High-resolution vectorized telemetry channels
                    v
    +-------------------------------+           +-------------------------------+
    |   TELEMETRY VISUALIZATION     |           |   RELATIONAL DATABASE (MySQL) |
    |-------------------------------|           |-------------------------------|
    | - Multi-panel Speed Profile   |           | Table `sim_runs`              |
    | - G-G Diagram (Kamm's Circle) |           | Table `vehicle_params`        |
    | - Delta-Time Comparison dt(s) |           | Table `sector_times`          |
    | - 2D Colored Track Map        |           | In-Memory Fallback Engine     |
    +-------------------------------+           +-------------------------------+
```

---

## 2. Theoretical Mathematical Derivations

### 2.1 Curvilinear Coordinate System & Kinematics
Let the racetrack centerline be parametrized by arc-length $s \in [0, S_{\rm total}]$ with unit tangent $\mathbf{t}(s)$ and principal normal $\mathbf{n}(s)$ in the Frenet-Serret frame. The local centerline curvature is defined by:

$$\kappa(s) = \left\| \frac{d\mathbf{t}}{ds} \right\| = \frac{x'(s) y''(s) - y'(s) x''(s)}{\left[x'(s)^2 + y'(s)^2\right]^{3/2}} = \frac{1}{R(s)}$$

where $R(s)$ is the instantaneous corner radius of curvature.

In the spatial domain, the vehicle velocity $v(s) = \frac{ds}{dt}$. Applying the chain rule to longitudinal acceleration $a_x$:

$$a_x = \frac{dv}{dt} = \frac{dv}{ds} \frac{ds}{dt} = v \frac{dv}{ds} = \frac{1}{2} \frac{d(v^2)}{ds}$$

Letting $y(s) \equiv v(s)^2$, the spatial kinematic governing equation reduces to:

$$\frac{dy}{ds} = 2 a_x(v, a_y)$$

### 2.2 Aerodynamics & Vertical Normal Load
The vehicle generates aerodynamic drag $F_d(v)$ and aerodynamic downforce $F_{\rm down}(v)$ as quadratic functions of relative air speed:

$$F_d(v) = \frac{1}{2} \rho C_D A v^2$$

$$F_{\rm down}(v) = \frac{1}{2} \rho C_L A v^2$$

where $\rho$ is ambient air density, $A$ is the frontal reference area, $C_D$ is the coefficient of aerodynamic drag, and $C_L$ is the lift/downforce coefficient (positive downwards).

The total instantaneous normal load $F_z(v)$ supported by all four tires is:

$$F_z(v) = m g + F_{\rm down}(v) = m g + \frac{1}{2} \rho C_L A v^2$$

### 2.3 Tire Normal Load Sensitivity
Pneumatic racing tires exhibit non-linear contact patch mechanics where the effective friction coefficient $\mu$ diminishes as vertical normal load increases. A normalized first-order load sensitivity model relative to static vehicle weight $W_0 = m g$ is formulated as:

$$\mu(F_z) = \mu_0 \left[ 1 - k_{\rm load} \left( \frac{F_z(v) - W_0}{W_0} \right) \right]$$

bounded by $\mu_{\rm min} \le \mu(F_z) \le 1.05 \mu_0$ to preserve physical realism at ultra-high downforce. The total available tire grip capacity is:

$$F_{\rm tire, max}(v) = \mu(F_z(v)) \cdot F_z(v)$$

### 2.4 Analytical Derivation of Apex Cornering Velocity
At a steady-state corner apex, lateral centripetal force demand is:

$$F_{y, \rm req}(v) = m a_y = m \kappa v^2$$

Steady-state lateral equilibrium occurs when demand equals peak grip:

$$m \kappa v^2 = \mu(F_z(v)) \cdot F_z(v)$$

Letting $u = v^2$, $W_0 = m g$, and $K_L = \frac{1}{2} \rho C_L A$:

$$F_z(u) = W_0 + K_L u$$

$$\mu(u) = \mu_0 \left( 1 - \frac{k_{\rm load} K_L}{W_0} u \right)$$

Substituting into the equilibrium condition:

$$m \kappa u = \mu_0 \left( 1 - \frac{k_{\rm load} K_L}{W_0} u \right) (W_0 + K_L u)$$

Expanding the right-hand side:

$$m \kappa u = \mu_0 \left[ W_0 + (1 - k_{\rm load}) K_L u - \frac{k_{\rm load} K_L^2}{W_0} u^2 \right]$$

Rearranging into standard quadratic form $A_{\rm quad} u^2 + B_{\rm quad} u + C_{\rm quad} = 0$:

$$A_{\rm quad} = \frac{\mu_0 k_{\rm load} K_L^2}{W_0}$$

$$B_{\rm quad} = m \kappa - \mu_0 (1 - k_{\rm load}) K_L$$

$$C_{\rm quad} = -\mu_0 W_0$$

Because $A_{\rm quad} > 0$ and $C_{\rm quad} < 0$, the discriminant $\Delta = B_{\rm quad}^2 - 4 A_{\rm quad} C_{\rm quad} > 0$ strictly guarantees a unique positive physical root:

$$u^* = \frac{-B_{\rm quad} + \sqrt{B_{\rm quad}^2 - 4 A_{\rm quad} C_{\rm quad}}}{2 A_{\rm quad}}$$

$$v_{\rm corner}(s) = \sqrt{u^*}$$

When curvature $\kappa \to 0$ (straight line), $v_{\rm corner} = v_{\rm top\_speed}$. This closed form eliminates numerical root-finding loops and is 100% vectorized in NumPy across all track nodes.

### 2.5 Kamm's Friction Ellipse (Combined Lateral & Longitudinal Grip)
Tire shear force capacity under simultaneous steering and acceleration/braking is bounded by the friction ellipse:

$$\left( \frac{F_{x, \rm tire}}{F_{\rm tire, max}} \right)^2 + \left( \frac{F_{y, \rm tire}}{F_{\rm tire, max}} \right)^2 \le 1.0$$

Given lateral force demand $F_y = m |a_y| = m \kappa v^2$, the maximum longitudinal tractive or braking force available from the tires is:

$$F_{x, \rm tire, avail}(v, a_y) = F_{\rm tire, max}(v) \sqrt{\max\left( 0, 1 - \left( \frac{m |a_y|}{F_{\rm tire, max}(v)} \right)^2 \right)}$$

### 2.6 Powertrain Acceleration & Mechanical Braking Limits
- **Acceleration**:
  Engine tractive force is power-limited by transmission efficiency $\eta$ and maximum brake power $P_{\rm max}$:
  $$F_{\rm drive}(v) = \min\left( F_{x, \rm tire, avail}, \frac{\eta P_{\rm max}}{\max(v, v_{\rm min})} \right)$$
  Net acceleration:
  $$a_{x, \rm accel}(v, a_y) = \frac{F_{\rm drive}(v) - F_d(v) - C_{rr} m g}{m}$$

- **Deceleration**:
  Braking incorporates mechanical brake clamp limits $F_{\rm brake, max}$, tire friction limits, aerodynamic drag, and rolling resistance:
  $$F_{\rm brake, total}(v, a_y) = \min(F_{\rm brake, max}, F_{x, \rm tire, avail}) + F_d(v) + C_{rr} m g$$
  Deceleration magnitude:
  $$|a_{x, \rm brake}(v, a_y)| = \frac{F_{\rm brake, total}(v, a_y)}{m}$$

### 2.7 Space-Domain Numerical Integration (Runge-Kutta 4th Order)
The quasi-steady-state solver implements a 4th-order Runge-Kutta (RK4) integration scheme in the spatial domain $s$.
With state variable $y = v^2$ and ODE $\frac{dy}{ds} = f(s, y) = 2 a_x(\sqrt{y}, a_y(s))$:

1. **Forward Acceleration Pass**:
   $$k_1 = 2 a_x(\sqrt{y_i}, a_y(s_i))$$
   $$k_2 = 2 a_x\left(\sqrt{y_i + \frac{\Delta s}{2} k_1}, a_y\left(s_i + \frac{\Delta s}{2}\right)\right)$$
   $$k_3 = 2 a_x\left(\sqrt{y_i + \frac{\Delta s}{2} k_2}, a_y\left(s_i + \frac{\Delta s}{2}\right)\right)$$
   $$k_4 = 2 a_x\left(\sqrt{y_i + \Delta s k_3}, a_y(s_i + \Delta s)\right)$$
   $$y_{i+1} = y_i + \frac{\Delta s}{6} (k_1 + 2 k_2 + 2 k_3 + k_4)$$
   $$v_{i+1} = \min\left( \sqrt{y_{i+1}}, v_{\rm corner}(s_{i+1}) \right)$$

2. **Backward Deceleration Pass**:
   Integrating backwards from $i+1$ to $i$ with $g(y, a_y) = 2 |a_{x, \rm brake}(\sqrt{y}, a_y)|$:
   $$y_i = y_{i+1} + \frac{\Delta s}{6} (k_1 + 2 k_2 + 2 k_3 + k_4)$$
   $$v_i = \min\left( v_i, \sqrt{y_i} \right)$$

3. **Temporal Integration & Derived Telemetry**:
   Using the trapezoidal rule over each spatial interval $\Delta s_i$:
   $$\Delta t_i = \frac{2 \Delta s_i}{v_i + v_{i+1}}$$
   $$T_{\rm lap} = \sum_{i=0}^{N-1} \Delta t_i$$

---

## 3. Directory Structure

```
lap_time_simulator/
├── src/
│   ├── __init__.py           # Package exports
│   ├── vehicle_model.py      # Vehicle parameters, aerodynamic drag/downforce, grip limits
│   ├── solver.py             # Forward-backward numerical integration pass (Euler/RK4)
│   ├── telemetry_db.py       # MySQL schema initialization, ingestion, and query pipeline
│   └── visualizer.py         # Velocity profile, GG-diagram, delta-time, and track map plotting
├── tests/
│   └── test_physics.py       # Unit tests validating conservation of energy and cornering limits
├── data/
│   └── sample_track.csv      # Synthetic track centerline coordinates (x, y, curvature, distance)
├── main.py                   # CLI entry point executing the full pipeline
└── README.md                 # Academic documentation with LaTeX derivations
```

---

## 4. MySQL Relational Database Schema

The relational telemetry pipeline records simulation runs into three normalized tables:

```sql
-- 1. Simulation Run Metadata
CREATE TABLE sim_runs (
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

-- 2. Vehicle Setup Parameters
CREATE TABLE vehicle_params (
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

-- 3. Sector Telemetry & Speed Traps
CREATE TABLE sector_times (
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
```

*Note: If a live MySQL server is not reachable, `TelemetryDatabase` automatically falls back to an embedded in-memory relational engine (SQLite) with zero code modifications, ensuring seamless testing and local execution.*

---

## 5. Installation & Requirements

### System Requirements
- Python 3.10+
- NumPy, Pandas, Matplotlib, `mysql-connector-python`, PyTest

```bash
pip install numpy pandas matplotlib mysql-connector-python pytest
```

---

## 6. Reproducible CLI Execution Instructions

### 6.1 Baseline Flying Lap Simulation (RK4)
Run the default Grand Prix circuit flying lap with 4th-Order Runge-Kutta integration:

```bash
python main.py
```

### 6.2 Comparative Sensitivity Study (High Downforce vs. Baseline)
Compare aerodynamic tradeoffs (+30% downforce, +15% drag penalty) with automatic delta-time analysis:

```bash
python main.py --compare
```

### 6.3 Standing Start with Forward/Backward Euler Integrator
```bash
python main.py --mode standing --v-start 0.0 --solver euler
```

### 6.4 Custom Vehicle Parameters & MySQL Database Connection
```bash
python main.py \
    --mass 790.0 \
    --power 780.0 \
    --downforce 3.50 \
    --drag 0.95 \
    --mu 1.75 \
    --db-host 127.0.0.1 \
    --db-user root \
    --db-pass secret \
    --db-name lap_telemetry
```

---

## 7. Numerical Verification & Unit Tests

The test suite validates physical conservation laws and numerical bounds:
1. **Work-Energy Theorem**: $\Delta E_k = \int F_{\rm net} ds$ to within $< 1.5\%$ spatial discretization error.
2. **Terminal Velocity Equilibrium**: $F_{\rm engine} = F_{\rm drag} + F_{\rm rr}$.
3. **Cornering Speed Envelope**: Centripetal lateral acceleration $a_y(s) \le \mu(F_z) F_z / m$ everywhere.
4. **Kamm Friction Ellipse**: $(F_x / F_{x, \rm max})^2 + (F_y / F_{y, \rm max})^2 \le 1.0$.
5. **Aero Quadratic Scaling**: $F_d \propto v^2$, $F_{\rm down} \propto v^2$.
6. **Relational Database Pipeline**: Schema creation, transactional insert, and queries.

Run the test suite:

```bash
python -m pytest tests/test_physics.py -v
```

Output:
```text
tests/test_physics.py::TestVehicleModelPhysics::test_aerodynamic_quadratic_scaling PASSED
tests/test_physics.py::TestVehicleModelPhysics::test_static_weight_and_normal_force PASSED
tests/test_physics.py::TestVehicleModelPhysics::test_tire_load_sensitivity_monotonicity PASSED
tests/test_physics.py::TestVehicleModelPhysics::test_friction_ellipse_combined_slip PASSED
tests/test_physics.py::TestVehicleModelPhysics::test_terminal_velocity_force_equilibrium PASSED
tests/test_physics.py::TestConservationOfEnergy::test_work_energy_theorem_in_acceleration PASSED
tests/test_physics.py::TestSolverCorneringAndConvergence::test_cornering_velocity_envelope_bound PASSED
tests/test_physics.py::TestSolverCorneringAndConvergence::test_rk4_vs_euler_consistency PASSED
tests/test_physics.py::TestTelemetryDatabasePipeline::test_db_initialization_and_logging PASSED
============================== 9 passed in 1.18s ==============================
```
