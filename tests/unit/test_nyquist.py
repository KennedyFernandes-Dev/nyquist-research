from math import pi, sqrt

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nyquist_research.frequency_response import frequency_grid, frequency_response
from nyquist_research.nyquist import nyquist
from nyquist_research.transfer_function import TransferFunction
from nyquist_research.winding_number import winding_number


def test_complex_response_and_polynomial_operations():
    system = TransferFunction([0, 4, 2], [2, 6, 4], gain=2)
    assert system.numerator == (2.0, 1.0)
    np.testing.assert_allclose(sorted(system.poles), [-2, -1])
    np.testing.assert_allclose(system.zeros, [-0.5])
    np.testing.assert_allclose(system.characteristic(), [1, 7, 4])
    values = frequency_response(system, [0, 1, 12])
    points = np.array([0, 1j, 12j])
    np.testing.assert_allclose(values, 2 * (2 * points + 1) / (points**2 + 3 * points + 2))
    assert np.all(np.diff(frequency_grid(system, points=37)) > 0)


@pytest.mark.parametrize(
    "numerator,denominator,gain,delay",
    [
        ([1], [0], 1, 0),
        ([1, 0], [1], 1, 0),
        ([1], [1, -1], float("nan"), 0),
        ([1], [1, -1], 1, -1),
        ([1j], [1, -1], 1, 0),
        ([1], [1, float("inf")], 1, 0),
    ],
)
def test_invalid_model(numerator, denominator, gain, delay):
    with pytest.raises(ValueError):
        TransferFunction(numerator, denominator, gain, delay)


def test_winding_orientation_and_multiplicity():
    loop = -1 + 0.4 * np.exp(-1j * np.linspace(0, 4 * pi, 129))
    assert winding_number(loop).turns == 2
    assert winding_number(loop[::-1]).turns == -2
    assert winding_number(loop, center=-2).turns == 0
    with pytest.raises(ValueError):
        winding_number([-1, 0, 1j])


@pytest.mark.parametrize(
    "system, expected",
    [
        (TransferFunction([6], [1, 6, 11, 6]), (0, 0, 0, "stable")),
        (TransferFunction([6], [1, 6, 11, 6], 11), (0, 2, 2, "unstable")),
        (TransferFunction([1], [1, 1, -2], 4), (1, -1, 0, "stable")),
        (TransferFunction([1], [1, 1, -2]), (1, 0, 1, "unstable")),
        (TransferFunction([-1, 1], [1, 3, 2], 4), (0, 2, 2, "unstable")),
        (TransferFunction([2], [1, 1], delay=1.5), (0, 2, 2, "unstable")),
        (TransferFunction([1], [1, 0]), (0, 0, 0, "stable")),
        (TransferFunction([8], [1, 3, 3, 1]), (0, None, None, "marginal")),
        (TransferFunction([2], [1, 1], delay=2 * pi / (3 * sqrt(3))), (0, None, None, "marginal")),
    ],
)
def test_contour_and_closed_loop_counts(system, expected):
    result = nyquist(system)
    assert (result.p, result.n, result.z, result.status) == expected
    assert result.contour[0] == result.contour[-1]
    assert np.all(result.contour.real >= 0)
    assert np.all(np.isfinite(result.mapped))


def test_delay_model_and_biproper_limit():
    delayed = TransferFunction([1], [1, 1], delay=2)
    np.testing.assert_allclose(delayed.evaluate(3j), np.exp(-6j) / (1 + 3j))
    with pytest.raises(ValueError):
        delayed.characteristic()
    assert nyquist(TransferFunction([2, 0], [1, 1], delay=2)).status == "indeterminate"
    assert nyquist(TransferFunction([0.5, 0], [1, 1], delay=2)).status == "stable"
    assert nyquist(TransferFunction([-1, 0], [1, 1])).status == "indeterminate"


@settings(max_examples=20, deadline=None)
@given(pole=st.floats(0.01, 100), gain=st.floats(0.01, 100))
def test_positive_first_order_always_stable(pole, gain):
    system = TransferFunction([1], [1, pole], gain)
    assert nyquist(system).status == "stable"
    np.testing.assert_allclose(np.roots(system.characteristic()), [-pole - gain])


def test_horner_ratio_at_large_complex_arguments():
    system = TransferFunction([1, 0, 0], [1, 0, 0, 0, 1])
    assert system.evaluate(1e150j) == pytest.approx(-1e-300)


def test_axis_mode_is_preserved_when_cancelled():
    for system in (
        TransferFunction([0], [1, 0]),
        TransferFunction([1, 0], [1, 1, 0]),
    ):
        result = nyquist(system)
        assert result.status == "marginal"
        assert result.z == 0
        assert "cancelado" in result.reason


@pytest.mark.parametrize("delay, expected", [(1.2, 0), (4.8, 2), (5.0, 4), (10.0, 6)])
def test_local_refinement_resolves_thin_and_multiple_turns(delay, expected):
    result = nyquist(TransferFunction([2], [1, 1], delay=delay))
    assert result.z == expected
    assert result.refinements == 2


def test_almost_axis_open_pole_does_not_produce_false_stability():
    result = nyquist(TransferFunction([1e-12], [1, -5e-11]))
    assert result.status == "indeterminate"


@pytest.mark.parametrize("gain", [-1e-12, -1e-4])
def test_open_integrator_indent_does_not_exclude_closed_unstable_pole(gain):
    result = nyquist(TransferFunction([1], [1, 0], gain))
    assert result.status == "unstable"
    assert result.z == 1


def test_closed_boundary_with_strictly_unstable_pole():
    result = nyquist(TransferFunction([4], [1, -1, 1, -5]))
    assert result.status == "unstable"
    assert result.z is None
    assert "também" in result.reason
