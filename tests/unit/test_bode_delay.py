import math

import numpy as np
import pytest

from nyquist_research.bode import bode, gain_frequencies
from nyquist_research.pade import approximate_delay, pade
from nyquist_research.time_delay import critical_delay
from nyquist_research.transfer_function import TransferFunction


def test_bode_simple_gain_crossing_without_finite_phase_crossing():
    system = TransferFunction([2], [1, 2, 1])
    result = bode(system, [0, 1, 10])
    assert len(result.gain_crossings) == 1
    assert result.gain_crossings[0] == pytest.approx((1, 90))
    assert result.phase_crossings == ()
    assert result.magnitude_db[1] == pytest.approx(0)
    assert result.phase_deg[1] == pytest.approx(-90)


def test_marginal_rational_loop_returns_matching_crossings():
    result = bode(TransferFunction([8], [1, 3, 3, 1]))
    assert result.gain_crossings[0] == pytest.approx((math.sqrt(3), 0), abs=1e-10)
    assert result.phase_crossings[0] == pytest.approx((math.sqrt(3), 1))
    assert result.phase_crossings[0].gain_margin_db == pytest.approx(0, abs=1e-12)


def test_nonminimum_phase_gain_margin():
    system = TransferFunction([-1, 1], [1, 3, 2])
    result = bode(system)
    assert result.gain_crossings == ()
    assert result.phase_crossings[0] == pytest.approx((math.sqrt(5), 3))


def test_open_loop_instability_does_not_eliminate_dc_phase_crossing():
    result = bode(TransferFunction([2], [1, -1]))
    assert result.phase_crossings[0] == pytest.approx((0, 0.5))


def test_multiple_gain_crossings_solve_quadratic_in_omega_squared():
    system = TransferFunction([3, 0.5], [1, 2, 1])
    expected = np.sqrt([(7 - math.sqrt(46)) / 2, (7 + math.sqrt(46)) / 2])
    np.testing.assert_allclose(gain_frequencies(system), expected, rtol=1e-11)


def test_gain_tangency_detected_once():
    system = TransferFunction([2, 0], [1, 2, 1])
    assert gain_frequencies(system) == pytest.approx([1], rel=1e-7)


def test_unity_and_negative_axis_continua_are_flagged():
    result = bode(TransferFunction([-1], [1]))
    assert result.gain_continuum and result.phase_continuum
    assert result.gain_crossings == ((0, 0),)
    assert result.phase_crossings == ((0, 1),)


def test_delayed_phase_uses_actual_band_and_finds_all_crossings_in_example():
    system = TransferFunction([2], [1, 1], delay=0.8)
    frequencies = np.array([0.0, 1.0, 2.0, 15.0])
    result = bode(system, frequencies)
    expected = -np.degrees(np.arctan(frequencies) + 0.8 * frequencies)
    np.testing.assert_allclose(result.phase_deg, expected, atol=1e-12)
    assert result.phase_band == (0, 15)
    assert len(result.phase_crossings) == 2
    for i, crossing in enumerate(result.phase_crossings):
        assert math.atan(crossing.omega) + 0.8 * crossing.omega == pytest.approx(
            (2 * i + 1) * math.pi
        )
        assert crossing.gain_margin == pytest.approx(math.hypot(1, crossing.omega) / 2)


def test_phase_does_not_rewrap_when_display_starts_at_high_frequency():
    system = TransferFunction([1], [1, 3, 3, 1])
    frequencies = [100, 200]
    result = bode(system, frequencies)
    np.testing.assert_allclose(result.phase_deg, -3 * np.degrees(np.arctan(frequencies)))


def test_integrator_pole_at_dc_is_excluded_and_phase_limit_is_visible():
    system = TransferFunction([1], [1, 0, 0, 0])
    result = bode(system, [0, 1])
    assert np.isnan(result.phase_deg[0])
    assert result.phase_deg[1] == pytest.approx(-270)
    assert result.gain_crossings[0].omega == pytest.approx(1)


@pytest.mark.parametrize("frequencies", [[], [1, 1], [2, 1], [-1, 1], [1, np.inf], [1j]])
def test_invalid_bode_frequency_grid(frequencies):
    with pytest.raises(ValueError):
        bode(TransferFunction([1], [1, 1]), frequencies)


@pytest.mark.parametrize(
    ("delay", "order", "expected"),
    [(0, 3, [1]), (1, 0, [1]), (2, 1, [1, 1]), (1, 3, [1 / 120, 0.1, 0.5, 1])],
)
def test_pade_recurrence_matches_reference_coefficients(delay, order, expected):
    numerator, denominator = pade(delay, order)
    np.testing.assert_allclose(denominator, expected, rtol=1e-15)
    np.testing.assert_allclose(
        numerator, np.asarray(expected) * (-1) ** np.arange(len(expected) - 1, -1, -1)
    )


@pytest.mark.parametrize(("delay", "order"), [(-1, 2), (np.inf, 2), (1, -1), (1, True), (1, 1e6)])
def test_pade_rejects_invalid_or_unrepresentable_arguments(delay, order):
    with pytest.raises((TypeError, ValueError)):
        pade(delay, order)


def test_pade_matches_exact_magnitude_and_preserves_loop_gain():
    system = TransferFunction([2], [1, 1], gain=3, delay=0.8)
    approximate = approximate_delay(system, 4)
    omega = np.geomspace(1e-4, 1e2)
    assert approximate.delay == 0 and approximate.gain == system.gain
    np.testing.assert_allclose(
        np.abs(approximate.evaluate(1j * omega)),
        np.abs(system.evaluate(1j * omega)),
        rtol=1e-12,
    )


def test_exact_critical_delay_is_absolute():
    system = TransferFunction([2], [1, 1], delay=0.8)
    result = critical_delay(system)
    assert result.status == "finite"
    assert result.delay == pytest.approx(2 * np.pi / (3 * np.sqrt(3)), rel=1e-11)
    assert result.frequency == pytest.approx(np.sqrt(3))
    assert result.candidates[0].direction == "destabilizing"


def test_delay_tangency_and_unbounded_case_have_distinct_statuses():
    tangent = critical_delay(TransferFunction([2, 0], [1, 2, 1]))
    assert tangent.delay == pytest.approx(np.pi, rel=1e-7)
    assert tangent.candidates[0].direction == "tangent"
    unbounded = critical_delay(TransferFunction([0.5], [1, 1]))
    assert unbounded.delay is None and unbounded.status == "unbounded"


@pytest.mark.parametrize(
    "system",
    [TransferFunction([1], [1, -2]), TransferFunction([-1], [1]), TransferFunction([1], [1])],
)
def test_critical_delay_reports_indeterminate_baseline_or_neutral_tail(system):
    result = critical_delay(system)
    assert result.delay is None and result.status == "indeterminate"


def test_critical_delay_can_be_zero_for_neutral_loop():
    result = critical_delay(TransferFunction([2, 3], [1, 1]))
    assert result.status == "finite"
    assert result.delay == 0 and result.frequency is None
