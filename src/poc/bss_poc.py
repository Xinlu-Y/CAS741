"""
Binary Star System (BSS) – Proof of Concept
============================================
Solves the two-body gravitational IVP and plots orbits.

Physical assumptions (SRS §4):
  A1 – Two-body system; third-body effects ignored.
  A2 – Only mutual Newtonian gravity; no non-gravitational forces.
  A3 – Newton's law of universal gravitation.
  A4 – Classical (non-relativistic) mechanics.
  A5 – Point-mass model; finite stellar size ignored.
  A6 – Constant stellar masses.
  A7 – Inertial (COM) reference frame; COM at origin with zero velocity.
  A8 – Planar (2-D) motion.
  A9 – No collisions; r12(t) > 0 for all t.

Equations of motion (SRS IM1, IM2):
  m1 · a1(t) = -G · m1·m2 / |r12|³ · r12
  m2 · a2(t) = +G · m1·m2 / |r12|³ · r12
where r12 = r1 - r2.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical constant (CODATA 2018, per SRS §3)
# ---------------------------------------------------------------------------
G = 6.67430e-11  # m³ kg⁻¹ s⁻²

# ---------------------------------------------------------------------------
# Specification parameter bounds (SRS §4, Table of constraints)
# ---------------------------------------------------------------------------
M_MIN = 1.0e29   # kg  – minimum stellar mass
M_MAX = 1.0e32   # kg  – maximum stellar mass
R_MAX = 1.0e13   # m   – maximum initial position magnitude
V_MAX = 1.0e6    # m/s – maximum initial speed magnitude
T_MIN = 1.0e3    # s   – minimum simulation duration
T_MAX = 1.0e10   # s   – maximum simulation duration

# Relative tolerance for COM constraint checks
_COM_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class InputError(ValueError):
    """Raised when BSS inputs violate physical or specification constraints."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_inputs(m1, m2, r1, r2, v1, v2, t_final):
    """
    Validate all BSS inputs against SRS constraints.

    Parameters
    ----------
    m1, m2    : float          – stellar masses [kg]
    r1, r2    : array-like (2,) – initial positions [m]
    v1, v2    : array-like (2,) – initial velocities [m/s]
    t_final   : float          – end time [s]

    Returns
    -------
    r1, r2, v1, v2 : ndarray – validated, converted to float arrays

    Raises
    ------
    InputError
        If any constraint is violated.
    """
    # --- Masses ---
    for name, m in (("m1", m1), ("m2", m2)):
        if m <= 0:
            raise InputError(f"{name} must be positive, got {m}")
        if not (M_MIN <= m <= M_MAX):
            raise InputError(
                f"{name} = {m:.3e} kg is outside the allowed range "
                f"[{M_MIN:.1e}, {M_MAX:.1e}] kg"
            )

    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    # --- Position bounds ---
    for name, vec in (("r1", r1), ("r2", r2)):
        mag = np.linalg.norm(vec)
        if mag > R_MAX:
            raise InputError(
                f"|{name}| = {mag:.3e} m exceeds r_max = {R_MAX:.1e} m"
            )

    # --- Velocity bounds ---
    for name, vec in (("v1", v1), ("v2", v2)):
        mag = np.linalg.norm(vec)
        if mag > V_MAX:
            raise InputError(
                f"|{name}| = {mag:.3e} m/s exceeds v_max = {V_MAX:.1e} m/s"
            )

    # --- Simulation duration ---
    if t_final <= 0:
        raise InputError(f"t_final must be positive, got {t_final}")
    if not (T_MIN <= t_final <= T_MAX):
        raise InputError(
            f"t_final = {t_final:.3e} s is outside the allowed range "
            f"[{T_MIN:.1e}, {T_MAX:.1e}] s"
        )

    # --- Centre-of-mass position constraint (SRS A7): m1·r1 + m2·r2 = 0 ---
    com_pos = m1 * r1 + m2 * r2
    scale_r = max(
        m1 * np.linalg.norm(r1),
        m2 * np.linalg.norm(r2),
        1.0,  # guard against both stars at origin
    )
    if np.linalg.norm(com_pos) / scale_r > _COM_TOL:
        raise InputError(
            f"Centre-of-mass position constraint violated: "
            f"m1·r1 + m2·r2 = {com_pos} ≠ 0  "
            f"(relative error {np.linalg.norm(com_pos)/scale_r:.2e})"
        )

    # --- Centre-of-mass velocity constraint (SRS A7): m1·v1 + m2·v2 = 0 ---
    com_vel = m1 * v1 + m2 * v2
    scale_v = max(
        m1 * np.linalg.norm(v1),
        m2 * np.linalg.norm(v2),
        1.0,
    )
    if np.linalg.norm(com_vel) / scale_v > _COM_TOL:
        raise InputError(
            f"Centre-of-mass velocity constraint violated: "
            f"m1·v1 + m2·v2 = {com_vel} ≠ 0  "
            f"(relative error {np.linalg.norm(com_vel)/scale_v:.2e})"
        )

    return r1, r2, v1, v2


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def _bss_ode(t, y, m1, m2):
    """
    ODE RHS for the binary star system.

    State vector layout:
      y = [r1x, r1y, r2x, r2y, v1x, v1y, v2x, v2y]

    Returns dy/dt = [v1, v2, a1, a2].
    """
    r1 = y[0:2]
    r2 = y[2:4]
    v1 = y[4:6]
    v2 = y[6:8]

    r12 = r1 - r2               # displacement r1 - r2
    dist = np.linalg.norm(r12)  # |r12|

    if dist == 0.0:
        raise RuntimeError("Stars collided (|r12| = 0); simulation aborted (A9 violated).")

    coeff = G / dist**3         # G / |r12|³

    # a1 = -G·m2/|r12|³ · r12  (attracted toward r2)
    # a2 = +G·m1/|r12|³ · r12  (attracted toward r1)
    a1 = -coeff * m2 * r12
    a2 =  coeff * m1 * r12

    return np.concatenate([v1, v2, a1, a2])


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def simulate(m1, m2, r1, r2, v1, v2, t_final,
             n_eval=1000, rtol=1e-10, atol=1e-12):
    """
    Simulate the binary star system.

    Parameters
    ----------
    m1, m2    : float          – stellar masses [kg]
    r1, r2    : array-like (2,) – initial positions [m]
    v1, v2    : array-like (2,) – initial velocities [m/s]
    t_final   : float          – simulation end time [s]
    n_eval    : int            – number of output time points
    rtol, atol: float          – ODE solver relative/absolute tolerances

    Returns
    -------
    t    : ndarray (n_eval,)    – time points [s]
    r1_t : ndarray (n_eval, 2)  – Star 1 position trajectory [m]
    r2_t : ndarray (n_eval, 2)  – Star 2 position trajectory [m]
    """
    r1, r2, v1, v2 = validate_inputs(m1, m2, r1, r2, v1, v2, t_final)

    y0 = np.concatenate([r1, r2, v1, v2])
    t_eval = np.linspace(0.0, t_final, n_eval)

    sol = solve_ivp(
        _bss_ode,
        (0.0, t_final),
        y0,
        args=(m1, m2),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    r1_t = sol.y[0:2].T   # (n_eval, 2)
    r2_t = sol.y[2:4].T   # (n_eval, 2)

    return sol.t, r1_t, r2_t


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_inputs(m1, m2, r1, r2, v1, v2, t_final):
    """Echo input parameters to stdout."""
    print("=" * 62)
    print("  Binary Star System Simulator – Input Parameters")
    print("=" * 62)
    print(f"  m1       = {m1:.6e} kg")
    print(f"  m2       = {m2:.6e} kg")
    print(f"  r1(0)    = [{r1[0]:.6e}, {r1[1]:.6e}] m")
    print(f"  r2(0)    = [{r2[0]:.6e}, {r2[1]:.6e}] m")
    print(f"  v1(0)    = [{v1[0]:.6e}, {v1[1]:.6e}] m/s")
    print(f"  v2(0)    = [{v2[0]:.6e}, {v2[1]:.6e}] m/s")
    print(f"  t_final  = {t_final:.6e} s")
    print("=" * 62)


def print_trajectory(t, r1_t, r2_t, n_print=10):
    """Print a uniformly-spaced sample of the trajectory."""
    print(f"\nTrajectory sample ({n_print} points):")
    hdr = (f"{'t [s]':>14}  {'r1x [m]':>14}  {'r1y [m]':>14}  "
           f"{'r2x [m]':>14}  {'r2y [m]':>14}")
    print(hdr)
    print("-" * len(hdr))
    for i in np.linspace(0, len(t) - 1, n_print, dtype=int):
        print(
            f"{t[i]:14.6e}  {r1_t[i, 0]:14.6e}  {r1_t[i, 1]:14.6e}  "
            f"{r2_t[i, 0]:14.6e}  {r2_t[i, 1]:14.6e}"
        )


def plot_trajectories(t, r1_t, r2_t, save_path=None):
    """
    Plot orbital trajectories and position vs. time.

    Parameters
    ----------
    t, r1_t, r2_t : outputs of simulate()
    save_path      : str or None – if given, save figure instead of showing
    """
    fig, (ax_orb, ax_time) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Orbit (xy) plot ──
    ax_orb.plot(r1_t[:, 0], r1_t[:, 1], color="steelblue", label="Star 1")
    ax_orb.plot(r2_t[:, 0], r2_t[:, 1], color="tomato",    label="Star 2")
    ax_orb.plot(*r1_t[0], "o", color="steelblue", ms=8, zorder=5)
    ax_orb.plot(*r2_t[0], "o", color="tomato",    ms=8, zorder=5)
    ax_orb.plot(0, 0, "+k", ms=12, label="COM (origin)")
    ax_orb.set_xlabel("x  [m]")
    ax_orb.set_ylabel("y  [m]")
    ax_orb.set_title("Orbital Trajectories")
    ax_orb.legend()
    ax_orb.set_aspect("equal")
    ax_orb.grid(True, alpha=0.3)

    # ── Position vs. time plot (displayed in AU for readability) ──
    AU = 1.496e11  # metres per astronomical unit
    ax_time.plot(t, r1_t[:, 0] / AU, color="steelblue",                  label="r1x")
    ax_time.plot(t, r1_t[:, 1] / AU, color="steelblue", ls="--",
                 label="r1y")
    ax_time.plot(t, r2_t[:, 0] / AU, color="tomato",    label="r2x")
    ax_time.plot(t, r2_t[:, 1] / AU, color="tomato",    ls="--",
                 label="r2y")
    ax_time.set_xlabel("t  [s]")
    ax_time.set_ylabel("position  [AU]")
    ax_time.set_title("Position vs. Time")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nPlot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example 1: equal-mass circular orbit (exactly one period)
# ---------------------------------------------------------------------------

def _example_circular():
    """
    Two equal-mass stars on a circular orbit in the COM frame.

    m = 1e30 kg  (≈ 0.5 M_sun)
    a = 1e11 m   (≈ 0.67 AU) – orbital radius of each star from COM

    Separation: r12 = 2a.
    Angular velocity from centripetal balance:
        m·ω²·a = G·m² / (2a)²  →  ω = sqrt(G·m / (4·a³))
    """
    m = 1.0e30
    a = 1.0e11
    omega = np.sqrt(G * m / (4.0 * a**3))
    T     = 2.0 * np.pi / omega

    m1, m2 = m, m
    r1 = np.array([ a, 0.0])
    r2 = np.array([-a, 0.0])
    v1 = np.array([0.0,  omega * a])
    v2 = np.array([0.0, -omega * a])

    return m1, m2, r1, r2, v1, v2, T


# ---------------------------------------------------------------------------
# Example 2: unequal-mass elliptical orbit (two full periods)
# ---------------------------------------------------------------------------

def _example_elliptical():
    """
    Two stars with different masses on an elliptical orbit (e = 0.5).

    m1 = 1.5e30 kg  (≈ 0.75 M_sun)
    m2 = 1.0e30 kg  (≈ 0.50 M_sun)
    a_rel = 3e11 m  – semi-major axis of the relative coordinate r = r1 - r2
    e     = 0.5     – eccentricity

    Initial conditions are placed at periapsis (closest approach), where the
    relative velocity is perpendicular to the separation vector.

    Vis-viva at periapsis:
        v_rel = sqrt(G·M·(1+e) / (a_rel·(1-e)))

    COM positions/velocities:
        r1 =  (m2/M)·r_rel,   r2 = -(m1/M)·r_rel
        v1 =  (m2/M)·v_rel,   v2 = -(m1/M)·v_rel
    """
    m1, m2  = 1.5e30, 1.0e30
    M       = m1 + m2
    a_rel   = 3.0e11   # m
    e       = 0.5

    # Relative-orbit periapsis distance and speed
    r_peri = a_rel * (1.0 - e)                              # 1.5e11 m
    v_peri = np.sqrt(G * M * (1.0 + e) / (a_rel * (1.0 - e)))  # ≈ 4.1e4 m/s

    # Orbital period of the relative coordinate
    T = 2.0 * np.pi * np.sqrt(a_rel**3 / (G * M))

    # At periapsis: separation along +x, velocity along +y
    r1 = np.array([ (m2 / M) * r_peri, 0.0])
    r2 = np.array([-(m1 / M) * r_peri, 0.0])
    v1 = np.array([0.0,  (m2 / M) * v_peri])
    v2 = np.array([0.0, -(m1 / M) * v_peri])

    return m1, m2, r1, r2, v1, v2, 2.0 * T   # simulate two full orbits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_example(label, m1, m2, r1, r2, v1, v2, t_final):
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print_inputs(m1, m2, r1, r2, v1, v2, t_final)

    print("\nValidating inputs …", end=" ")
    validate_inputs(m1, m2, r1, r2, v1, v2, t_final)
    print("OK")

    print("Running simulation …", end=" ", flush=True)
    t, r1_t, r2_t = simulate(m1, m2, r1, r2, v1, v2, t_final)
    print(f"done  ({len(t)} time steps)")

    print_trajectory(t, r1_t, r2_t)
    plot_trajectories(t, r1_t, r2_t)


if __name__ == "__main__":
    _run_example("Example 1 – Equal-mass circular orbit",
                 *_example_circular())

    _run_example("Example 2 – Unequal-mass elliptical orbit (e = 0.5, 2 periods)",
                 *_example_elliptical())
