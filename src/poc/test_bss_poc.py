"""
pytest test suite for bss_poc.py
=================================

Test groups
-----------
1. TestCOMConservation   – equal-mass symmetric case; COM must stay at origin.
2. TestCircularOrbit     – numerical solution compared to analytical circular orbit.
3. TestInputValidation   – InputError raised for every constraint violation.
"""

import numpy as np
import pytest

from bss_poc import (
    G,
    M_MIN, M_MAX, R_MAX, V_MAX, T_MIN, T_MAX,
    InputError,
    validate_inputs,
    simulate,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _circular_orbit_ics(m, a):
    """
    Return equal-mass circular-orbit initial conditions in the COM frame.

    Two stars of mass m each orbit the COM at radius a.
    Centripetal balance: m·ω²·a = G·m²/(2a)²  →  ω = sqrt(G·m / (4·a³)).

    Returns
    -------
    m1, m2  : float
    r1, r2  : ndarray (2,)
    v1, v2  : ndarray (2,)
    T       : float  – orbital period [s]
    omega   : float  – angular velocity [rad/s]
    """
    omega = np.sqrt(G * m / (4.0 * a**3))
    T     = 2.0 * np.pi / omega

    r1 = np.array([ a, 0.0])
    r2 = np.array([-a, 0.0])
    v1 = np.array([0.0,  omega * a])
    v2 = np.array([0.0, -omega * a])

    return m, m, r1, r2, v1, v2, T, omega


# ===========================================================================
# 1. Centre-of-Mass Conservation
# ===========================================================================

class TestCOMConservation:
    """
    For an equal-mass symmetric system the COM must remain at the origin
    throughout the simulation (SRS A7).
    """

    M = 1.0e30   # kg
    A = 1.0e11   # m

    def _run(self, **kw):
        m1, m2, r1, r2, v1, v2, T, _ = _circular_orbit_ics(self.M, self.A)
        return simulate(m1, m2, r1, r2, v1, v2, T, **kw)

    def test_com_position_at_origin_throughout(self):
        """COM = (m1·r1 + m2·r2)/(m1+m2) must remain ≈ 0 for all t."""
        m = self.M
        t, r1_t, r2_t = self._run(n_eval=500)

        com = (m * r1_t + m * r2_t) / (2.0 * m)   # equal masses → (r1+r2)/2
        max_drift = np.max(np.abs(com))

        # Allow up to 1 km drift (r_orbit = 1e11 m → relative tolerance ~ 1e-8)
        assert max_drift < 1.0e3, (
            f"COM drifted: max |COM| = {max_drift:.3e} m  (limit 1e3 m)"
        )

    def test_com_zero_initially(self):
        """The initial conditions must satisfy the COM constraint exactly."""
        m1, m2, r1, r2, v1, v2, T, _ = _circular_orbit_ics(self.M, self.A)
        com0 = m1 * r1 + m2 * r2
        assert np.allclose(com0, 0.0, atol=1.0), (
            f"Initial COM ≠ 0: {com0}"
        )

    def test_com_conserved_unequal_masses(self):
        """COM conservation must also hold when m1 ≠ m2."""
        m1, m2 = 1.0e30, 2.0e30
        M = m1 + m2
        # Place stars so COM = 0:  r1 = -m2/M * d·x̂,  r2 = m1/M * d·x̂
        d = 2.0e11  # separation [m]
        r1 = np.array([-m2 / M * d, 0.0])
        r2 = np.array([ m1 / M * d, 0.0])

        # Circular orbit angular velocity of the relative coordinate:
        # μ·ω²·d = G·m1·m2/d²  →  ω = sqrt(G·M / d³)
        omega = np.sqrt(G * M / d**3)
        T = 2.0 * np.pi / omega

        # Velocities in COM frame: vi = ω × ri  (perpendicular)
        v1 = np.array([0.0, -omega * r1[0]])   # r1[0] < 0 → v1y > 0
        v2 = np.array([0.0, -omega * r2[0]])   # r2[0] > 0 → v2y < 0

        t, r1_t, r2_t = simulate(m1, m2, r1, r2, v1, v2, T, n_eval=500)

        com = (m1 * r1_t + m2 * r2_t) / M
        max_drift = np.max(np.abs(com))
        assert max_drift < 1.0e3, (
            f"COM drifted (unequal masses): max |COM| = {max_drift:.3e} m"
        )


# ===========================================================================
# 2. Circular Orbit – Analytical Comparison
# ===========================================================================

class TestCircularOrbit:
    """
    The numerical solution of the equal-mass circular orbit must match the
    known analytical solution r1(t) = a·[cos(ωt), sin(ωt)].
    """

    M = 1.0e30
    A = 1.0e11
    # Use tight solver tolerances so errors are dominated by physics, not numerics
    RTOL = 1.0e-11
    ATOL = 1.0e-13

    def _run(self, n_eval=2000):
        m1, m2, r1, r2, v1, v2, T, omega = _circular_orbit_ics(self.M, self.A)
        t, r1_t, r2_t = simulate(
            m1, m2, r1, r2, v1, v2, T, n_eval=n_eval,
            rtol=self.RTOL, atol=self.ATOL,
        )
        return t, r1_t, r2_t, T, omega

    def test_orbital_radius_star1_constant(self):
        """Star 1 must stay at distance a from COM throughout the orbit."""
        t, r1_t, _, T, _ = self._run()
        r1_norm = np.linalg.norm(r1_t, axis=1)
        rel_err = np.max(np.abs(r1_norm - self.A) / self.A)
        assert rel_err < 1.0e-5, (
            f"Star 1 orbital radius not constant: max rel err = {rel_err:.2e}"
        )

    def test_orbital_radius_star2_constant(self):
        """Star 2 must stay at distance a from COM throughout the orbit."""
        t, _, r2_t, T, _ = self._run()
        r2_norm = np.linalg.norm(r2_t, axis=1)
        rel_err = np.max(np.abs(r2_norm - self.A) / self.A)
        assert rel_err < 1.0e-5, (
            f"Star 2 orbital radius not constant: max rel err = {rel_err:.2e}"
        )

    def test_period_recovery_star1(self):
        """After exactly one period T, Star 1 must return to its starting position."""
        t, r1_t, _, T, _ = self._run(n_eval=5000)
        pos_err = np.linalg.norm(r1_t[-1] - r1_t[0])
        rel_err = pos_err / self.A
        assert rel_err < 1.0e-4, (
            f"Star 1 did not close after 1 period: |Δr|/a = {rel_err:.2e}"
        )

    def test_period_recovery_star2(self):
        """After exactly one period T, Star 2 must return to its starting position."""
        t, _, r2_t, T, _ = self._run(n_eval=5000)
        pos_err = np.linalg.norm(r2_t[-1] - r2_t[0])
        rel_err = pos_err / self.A
        assert rel_err < 1.0e-4, (
            f"Star 2 did not close after 1 period: |Δr|/a = {rel_err:.2e}"
        )

    def test_trajectory_matches_analytical_star1(self):
        """
        r1(t) should match the analytical circular solution
            r1(t) = a · [cos(ωt), sin(ωt)]
        to within a relative error of 1e-4.
        """
        t, r1_t, _, T, omega = self._run(n_eval=1000)
        r1_analytic = np.column_stack([
            self.A * np.cos(omega * t),
            self.A * np.sin(omega * t),
        ])
        err = np.max(np.linalg.norm(r1_t - r1_analytic, axis=1)) / self.A
        assert err < 1.0e-4, (
            f"Trajectory deviates from analytic circular orbit: {err:.2e}"
        )

    def test_trajectory_matches_analytical_star2(self):
        """
        r2(t) = -r1(t) for equal masses; compare with analytical solution.
        """
        t, _, r2_t, T, omega = self._run(n_eval=1000)
        r2_analytic = np.column_stack([
            -self.A * np.cos(omega * t),
            -self.A * np.sin(omega * t),
        ])
        err = np.max(np.linalg.norm(r2_t - r2_analytic, axis=1)) / self.A
        assert err < 1.0e-4, (
            f"Star 2 trajectory deviates from analytic solution: {err:.2e}"
        )

    def test_separation_constant(self):
        """The distance |r12(t)| must remain equal to 2a throughout."""
        t, r1_t, r2_t, T, _ = self._run()
        r12 = np.linalg.norm(r1_t - r2_t, axis=1)
        rel_err = np.max(np.abs(r12 - 2.0 * self.A) / (2.0 * self.A))
        assert rel_err < 1.0e-5, (
            f"Separation not constant: max rel err = {rel_err:.2e}"
        )

    def test_energy_conserved(self):
        """
        Total mechanical energy E = KE + PE must be conserved.
        Computed via finite differences (first-order) – allow 0.1 % tolerance.
        """
        m = self.M
        t, r1_t, r2_t, T, omega = self._run(n_eval=5000)

        # Approximate velocities by central differences (interior points)
        dt = np.diff(t)
        # Use midpoint-centred differences for interior; forward diff at ends
        v1_t = np.gradient(r1_t, t, axis=0)
        v2_t = np.gradient(r2_t, t, axis=0)

        ke = 0.5 * m * (np.sum(v1_t**2, axis=1) + np.sum(v2_t**2, axis=1))
        r12 = np.linalg.norm(r1_t - r2_t, axis=1)
        pe = -G * m * m / r12
        E = ke + pe

        # Exclude first and last few points (boundary FD artifacts)
        E_mid = E[10:-10]
        rel_var = (E_mid.max() - E_mid.min()) / abs(E_mid.mean())
        assert rel_var < 1.0e-3, (
            f"Energy not conserved: relative variation = {rel_var:.3e}"
        )


# ===========================================================================
# 3. Input Validation – Error Handling
# ===========================================================================

class TestInputValidation:
    """
    validate_inputs() must raise InputError with informative messages
    for every constraint defined in the SRS.
    """

    # ── Canonical valid ICs ──────────────────────────────────────────────────
    @staticmethod
    def _valid():
        m, a = 1.0e30, 1.0e11
        m1, m2, r1, r2, v1, v2, T, _ = _circular_orbit_ics(m, a)
        return m1, m2, r1, r2, v1, v2, T

    # ── Passes ──────────────────────────────────────────────────────────────

    def test_valid_inputs_do_not_raise(self):
        validate_inputs(*self._valid())   # must not raise

    # ── Mass constraints ─────────────────────────────────────────────────────

    def test_m1_zero_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        with pytest.raises(InputError, match="m1"):
            validate_inputs(0.0, m2, r1, r2, v1, v2, t)

    def test_m1_negative_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        with pytest.raises(InputError, match="m1"):
            validate_inputs(-1.0e30, m2, r1, r2, v1, v2, t)

    def test_m2_zero_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        with pytest.raises(InputError, match="m2"):
            validate_inputs(m1, 0.0, r1, r2, v1, v2, t)

    def test_m1_below_m_min_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        with pytest.raises(InputError, match="m1"):
            validate_inputs(M_MIN / 10.0, m2, r1, r2, v1, v2, t)

    def test_m1_above_m_max_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        with pytest.raises(InputError, match="m1"):
            validate_inputs(M_MAX * 10.0, m2, r1, r2, v1, v2, t)

    def test_m2_below_m_min_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        with pytest.raises(InputError, match="m2"):
            validate_inputs(m1, M_MIN / 10.0, r1, r2, v1, v2, t)

    def test_m2_above_m_max_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        with pytest.raises(InputError, match="m2"):
            validate_inputs(m1, M_MAX * 10.0, r1, r2, v1, v2, t)

    # ── Position constraints ─────────────────────────────────────────────────

    def test_r1_exceeds_r_max_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        bad_r1 = np.array([R_MAX * 2.0, 0.0])
        with pytest.raises(InputError, match="r1"):
            validate_inputs(m1, m2, bad_r1, r2, v1, v2, t)

    def test_r2_exceeds_r_max_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        bad_r2 = np.array([0.0, R_MAX * 2.0])
        with pytest.raises(InputError, match="r2"):
            validate_inputs(m1, m2, r1, bad_r2, v1, v2, t)

    # ── Velocity constraints ─────────────────────────────────────────────────

    def test_v1_exceeds_v_max_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        bad_v1 = np.array([V_MAX * 2.0, 0.0])
        with pytest.raises(InputError, match="v1"):
            validate_inputs(m1, m2, r1, r2, bad_v1, v2, t)

    def test_v2_exceeds_v_max_raises(self):
        m1, m2, r1, r2, v1, v2, t = self._valid()
        bad_v2 = np.array([0.0, V_MAX * 2.0])
        with pytest.raises(InputError, match="v2"):
            validate_inputs(m1, m2, r1, r2, v1, bad_v2, t)

    # ── Time constraints ─────────────────────────────────────────────────────

    def test_t_final_zero_raises(self):
        m1, m2, r1, r2, v1, v2, _ = self._valid()
        with pytest.raises(InputError, match="t_final"):
            validate_inputs(m1, m2, r1, r2, v1, v2, 0.0)

    def test_t_final_negative_raises(self):
        m1, m2, r1, r2, v1, v2, _ = self._valid()
        with pytest.raises(InputError, match="t_final"):
            validate_inputs(m1, m2, r1, r2, v1, v2, -1.0e6)

    def test_t_final_below_t_min_raises(self):
        m1, m2, r1, r2, v1, v2, _ = self._valid()
        with pytest.raises(InputError, match="t_final"):
            validate_inputs(m1, m2, r1, r2, v1, v2, T_MIN / 2.0)

    def test_t_final_above_t_max_raises(self):
        m1, m2, r1, r2, v1, v2, _ = self._valid()
        with pytest.raises(InputError, match="t_final"):
            validate_inputs(m1, m2, r1, r2, v1, v2, T_MAX * 2.0)

    # ── COM constraints ──────────────────────────────────────────────────────

    def test_com_position_violated_raises(self):
        """Shift r1 so that m1·r1 + m2·r2 ≠ 0."""
        m1, m2, r1, r2, v1, v2, t = self._valid()
        bad_r1 = r1 + np.array([1.0e10, 0.0])   # small relative to r_max
        with pytest.raises(InputError, match="[Cc]entre-of-mass position"):
            validate_inputs(m1, m2, bad_r1, r2, v1, v2, t)

    def test_com_velocity_violated_raises(self):
        """Add a bulk drift to v1 so that m1·v1 + m2·v2 ≠ 0."""
        m1, m2, r1, r2, v1, v2, t = self._valid()
        bad_v1 = v1 + np.array([1.0e4, 0.0])    # ~1e4 m/s drift; |bad_v1| < v_max
        with pytest.raises(InputError, match="[Cc]entre-of-mass velocity"):
            validate_inputs(m1, m2, r1, r2, bad_v1, v2, t)

    def test_error_message_is_informative(self):
        """The InputError message must mention the offending parameter."""
        m1, m2, r1, r2, v1, v2, t = self._valid()
        try:
            validate_inputs(0.0, m2, r1, r2, v1, v2, t)
        except InputError as exc:
            assert "m1" in str(exc), (
                f"Error message does not mention 'm1': {exc}"
            )
        else:
            pytest.fail("InputError was not raised for m1 = 0")
