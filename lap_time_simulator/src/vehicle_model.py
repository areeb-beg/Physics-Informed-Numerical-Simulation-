"""Vehicle Dynamics and Tire-Aero Interaction Model.

Implements point-mass vehicle dynamics, aerodynamic downforce and drag,
powertrain tractive capability, braking limits, and load-sensitive tire grip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class VehicleParameters:
    """Rigid vehicle and physical parameters for lap time simulation.

    Attributes
    ----------
    mass : float
        Total vehicle mass including driver and fluids [kg].
    g : float
        Gravitational acceleration [m/s^2].
    rho : float
        Ambient atmospheric air density [kg/m^3].
    frontal_area : float
        Vehicle frontal reference cross-sectional area [m^2].
    cd : float
        Aerodynamic drag coefficient C_D [-].
    cl : float
        Aerodynamic downforce (lift) coefficient C_L [-] (positive down).
    power_max : float
        Maximum powertrain brake power output [W].
    f_brake_max : float
        Peak mechanical brake system clamp capacity [N].
    mu_0 : float
        Nominal tire friction coefficient at static normal weight [-].
    load_sensitivity : float
        Tire normal load sensitivity coefficient k_load [-].
    crr : float
        Rolling resistance coefficient C_rr [-].
    drive_efficiency : float
        Driveline and gearbox mechanical transmission efficiency eta [-].
    v_max_mech : float
        Physical gearing rev-limit top speed limit [m/s].
    v_min : float
        Numerical regularizing velocity floor [m/s] to prevent 1/v blowup.
    """

    mass: float = 798.0
    g: float = 9.80665
    rho: float = 1.225
    frontal_area: float = 1.60
    cd: float = 0.90
    cl: float = 3.20
    power_max: float = 745700.0
    f_brake_max: float = 30000.0
    mu_0: float = 1.70
    load_sensitivity: float = 0.15
    crr: float = 0.015
    drive_efficiency: float = 0.92
    v_max_mech: float = 105.0
    v_min: float = 1.0


class VehicleModel:
    """Physics-informed point-mass vehicle dynamics model."""

    def __init__(self, params: VehicleParameters | None = None) -> None:
        """Initialize vehicle model with physical parameters.

        Parameters
        ----------
        params : VehicleParameters | None
            Vehicle parameter dataclass. Defaults to standard F1-class setup.
        """
        self.params = params if params is not None else VehicleParameters()

    @property
    def static_weight(self) -> float:
        """Static gravitational normal weight W = m * g [N]."""
        return self.params.mass * self.params.g

    def aerodynamic_drag(self, v: ArrayLike) -> ArrayLike:
        """Compute aerodynamic drag force F_d = 0.5 * rho * C_D * A * v^2 [N].

        Parameters
        ----------
        v : ArrayLike
            Vehicle forward velocity [m/s].

        Returns
        -------
        ArrayLike
            Aerodynamic resistance force [N].
        """
        p = self.params
        return 0.5 * p.rho * p.cd * p.frontal_area * (v**2)

    def aerodynamic_downforce(self, v: ArrayLike) -> ArrayLike:
        """Compute aerodynamic downforce F_z_aero = 0.5 * rho * C_L * A * v^2 [N].

        Parameters
        ----------
        v : ArrayLike
            Vehicle forward velocity [m/s].

        Returns
        -------
        ArrayLike
            Aerodynamic normal force pushing vehicle into track surface [N].
        """
        p = self.params
        return 0.5 * p.rho * p.cl * p.frontal_area * (v**2)

    def total_vertical_load(self, v: ArrayLike) -> ArrayLike:
        """Compute total normal tire load F_z = m * g + F_downforce(v) [N].

        Parameters
        ----------
        v : ArrayLike
            Vehicle forward velocity [m/s].

        Returns
        -------
        ArrayLike
            Combined gravitational and aerodynamic vertical normal force [N].
        """
        return self.static_weight + self.aerodynamic_downforce(v)

    def tire_friction_coefficient(self, f_z: ArrayLike) -> ArrayLike:
        """Compute load-sensitive tire friction coefficient mu(F_z).

        Real pneumatic racing tires exhibit diminishing returns with increasing
        normal load (tire load sensitivity). The normalized formulation is:
            mu(F_z) = mu_0 * [1 - k_load * ((F_z - W) / W)]
        clamped to a physical minimum of 0.5 * mu_0.

        Parameters
        ----------
        f_z : ArrayLike
            Total normal load [N].

        Returns
        -------
        ArrayLike
            Effective tire friction coefficient mu [-].
        """
        p = self.params
        w0 = self.static_weight
        delta_ratio = (f_z - w0) / w0
        raw_mu = p.mu_0 * (1.0 - p.load_sensitivity * delta_ratio)
        # Clamping to maintain physical realism at extreme aerodynamic loads
        min_mu = 0.50 * p.mu_0
        max_mu = p.mu_0 * 1.05
        return np.clip(raw_mu, min_mu, max_mu)

    def max_cornering_speed(self, curvature: ArrayLike) -> ArrayLike:
        """Calculate maximum cornering velocity governed by steady-state lateral grip.

        Equating required lateral centripetal force:
            F_y,req = m * kappa * v^2
        to peak lateral tire grip:
            F_y,max = mu(F_z(v)) * F_z(v)
        Where:
            F_z(v) = W + K_L * v^2
            K_L = 0.5 * rho * C_L * A
            mu(F_z) = mu_0 * [1 - k_load * (K_L * v^2 / W)]

        This forms a closed-form quadratic equation in u = v^2:
            A_q * u^2 + B_q * u + C_q = 0
            A_q = mu_0 * k_load * K_L^2 / W
            B_q = m * kappa - mu_0 * (1 - k_load) * K_L
            C_q = -mu_0 * W

        Parameters
        ----------
        curvature : ArrayLike
            Track curvature magnitude kappa = 1 / R [1/m].

        Returns
        -------
        ArrayLike
            Maximum cornering speed [m/s].
        """
        p = self.params
        w = self.static_weight
        k_l = 0.5 * p.rho * p.cl * p.frontal_area

        is_scalar = np.isscalar(curvature)
        kappa = np.atleast_1d(np.asarray(curvature, dtype=float))
        abs_kappa = np.abs(kappa)

        # Output speed array initialized to mechanical top speed
        v_out = np.full_like(abs_kappa, p.v_max_mech, dtype=float)

        # Mask nodes where curvature is non-zero (curves)
        curve_mask = abs_kappa > 1e-6
        if not np.any(curve_mask):
            return float(v_out[0]) if is_scalar else v_out

        k_active = abs_kappa[curve_mask]

        if p.load_sensitivity > 1e-9 and k_l > 1e-9:
            a_q = p.mu_0 * p.load_sensitivity * (k_l**2) / w
            b_q = p.mass * k_active - p.mu_0 * (1.0 - p.load_sensitivity) * k_l
            c_q = -p.mu_0 * w

            # Quadratic formula for u = v^2: u = (-B + sqrt(B^2 - 4AC)) / (2A)
            disc = np.maximum(b_q**2 - 4.0 * a_q * c_q, 0.0)
            u_root = (-b_q + np.sqrt(disc)) / (2.0 * a_q)
        else:
            # Linear form when load sensitivity is zero
            denom = p.mass * k_active - p.mu_0 * k_l
            denom = np.maximum(denom, 1e-9)
            u_root = (p.mu_0 * w) / denom

        u_root = np.maximum(u_root, 0.0)
        v_active = np.sqrt(u_root)
        v_active = np.clip(v_active, p.v_min, p.v_max_mech)

        v_out[curve_mask] = v_active

        # Terminal velocity clamp
        v_term = self.terminal_velocity()
        v_out = np.minimum(v_out, v_term)

        return float(v_out[0]) if is_scalar else v_out

    def max_longitudinal_acceleration(
        self, v: ArrayLike, a_y: ArrayLike = 0.0
    ) -> ArrayLike:
        """Compute maximum longitudinal acceleration a_x under traction and grip limits.

        Employs the Kamm friction ellipse:
            (F_x,tire / F_x,max)^2 + (F_y / F_y,max)^2 <= 1.0
        Powertrain is power-limited by P_max and tire-traction-limited.

        Parameters
        ----------
        v : ArrayLike
            Vehicle speed [m/s].
        a_y : ArrayLike
            Lateral centripetal acceleration [m/s^2].

        Returns
        -------
        ArrayLike
            Maximum forward acceleration a_x [m/s^2].
        """
        p = self.params
        v_safe = np.maximum(v, p.v_min)

        # Normal load and tire grip
        f_z = self.total_vertical_load(v_safe)
        mu = self.tire_friction_coefficient(f_z)
        f_tire_peak = mu * f_z

        # Lateral force and friction ellipse budget
        f_y = p.mass * np.abs(a_y)
        lat_ratio = np.clip(f_y / np.maximum(f_tire_peak, 1e-6), 0.0, 1.0)
        f_x_tire_avail = f_tire_peak * np.sqrt(1.0 - lat_ratio**2)

        # Powertrain tractive force
        f_engine = (p.drive_efficiency * p.power_max) / v_safe
        f_drive = np.minimum(f_engine, f_x_tire_avail)

        # Resistances
        f_drag = self.aerodynamic_drag(v_safe)
        f_rr = p.crr * self.static_weight

        # Net acceleration
        f_net = f_drive - f_drag - f_rr
        return f_net / p.mass

    def max_longitudinal_deceleration(
        self, v: ArrayLike, a_y: ArrayLike = 0.0
    ) -> ArrayLike:
        """Compute maximum longitudinal deceleration magnitude |a_x,brake| [m/s^2].

        Deceleration combines mechanical braking, aerodynamic drag, and rolling
        resistance within the available tire friction ellipse:
            |F_brake,net| = min(F_brake_max, F_x,tire_avail) + F_drag + F_rr

        Parameters
        ----------
        v : ArrayLike
            Vehicle speed [m/s].
        a_y : ArrayLike
            Lateral centripetal acceleration [m/s^2].

        Returns
        -------
        ArrayLike
            Magnitude of maximum braking deceleration [m/s^2] (positive quantity).
        """
        p = self.params
        v_safe = np.maximum(v, 0.0)

        # Normal load and tire grip
        f_z = self.total_vertical_load(v_safe)
        mu = self.tire_friction_coefficient(f_z)
        f_tire_peak = mu * f_z

        # Friction ellipse budget
        f_y = p.mass * np.abs(a_y)
        lat_ratio = np.clip(f_y / np.maximum(f_tire_peak, 1e-6), 0.0, 1.0)
        f_x_tire_avail = f_tire_peak * np.sqrt(1.0 - lat_ratio**2)

        # Mechanical brake system clamp force
        f_brake_mech = np.minimum(p.f_brake_max, f_x_tire_avail)

        # Drag and rolling resistance contribute positively to deceleration
        f_drag = self.aerodynamic_drag(v_safe)
        f_rr = p.crr * self.static_weight

        f_decel_total = f_brake_mech + f_drag + f_rr
        return f_decel_total / p.mass

    def terminal_velocity(self) -> float:
        """Compute steady-state straight-line aerodynamic/powertrain terminal velocity.

        Equates net engine tractive force to total resistive forces:
            (eta * P_max) / v = 0.5 * rho * C_D * A * v^2 + C_rr * m * g
        Solved via vectorized 1D root search.

        Returns
        -------
        float
            Terminal velocity [m/s].
        """
        p = self.params
        # Fast bisection search in physical interval [10, v_max_mech]
        v_low = 10.0
        v_high = p.v_max_mech
        for _ in range(30):
            v_mid = 0.5 * (v_low + v_high)
            f_engine = (p.drive_efficiency * p.power_max) / v_mid
            f_resist = self.aerodynamic_drag(v_mid) + p.crr * self.static_weight
            if f_engine > f_resist:
                v_low = v_mid
            else:
                v_high = v_mid
        return float(np.minimum(0.5 * (v_low + v_high), p.v_max_mech))
