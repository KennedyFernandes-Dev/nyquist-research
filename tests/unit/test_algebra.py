from itertools import permutations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nyquist_research.root_locus import _assignment, root_locus
from nyquist_research.routh_hurwitz import routh_hurwitz
from nyquist_research.step_response import step_response
from nyquist_research.transfer_function import TransferFunction


def test_routh_classical_table_and_pivot():
    classical = routh_hurwitz([1, 6, 11, 6])
    np.testing.assert_allclose(classical.table[:, 0], [1, 6, 10, 6])
    assert classical.status == "stable"
    pivot = routh_hurwitz([1, 0, 2, 1])
    assert (pivot.rhp, pivot.axis, pivot.status) == (2, 0, "unstable")
    assert any("Pivô nulo" in event for event in pivot.events)


@pytest.mark.parametrize(
    ("poly", "expected"),
    [
        ([1, 2, 1, 2], (0, 2, "marginal")),
        ([1, 0, -1], (1, 0, "unstable")),
        ([1, 1, 0, 0], (0, 2, "marginal")),
        ([0, 3, 0, 0], (0, 2, "marginal")),
        ([3], (0, 0, "stable")),
        ([1, 0, -3, 0, -4], (1, 2, "unstable")),
        ([1, 1e-14, 1], (0, 0, "stable")),
        ([1, -1e-14, 1], (2, 0, "unstable")),
    ],
)
def test_routh_counts(poly, expected):
    result = routh_hurwitz(poly)
    assert (result.rhp, result.axis, result.status) == expected


def test_routh_repeated_imaginary_pairs_are_counted_with_multiplicity():
    polynomial = np.polynomial.polynomial.polypow([1, 0, 1], 5)[::-1]
    result = routh_hurwitz(np.convolve(polynomial, [1, 3]))
    assert (result.rhp, result.axis, result.status) == (0, 10, "marginal")


def test_routh_handles_large_coefficient_scales():
    cases = [
        ([1e-310, 1e300, 1e-300], 0),
        ([1e300, 1e-300, -1e300], 1),
    ]
    for coefficients, rhp in cases:
        assert routh_hurwitz(coefficients).rhp == rhp


@pytest.mark.parametrize("bad_poly", [[0, 0], [1, np.inf], [[1, 1]], np.asarray([1, 1j])])
def test_routh_rejects_invalid_polynomial(bad_poly):
    with pytest.raises(ValueError):
        routh_hurwitz(bad_poly)


@given(
    st.lists(st.integers(-5, 5), min_size=1, max_size=8),
    st.sampled_from([-8.0, 2.0**-150, 2.0**150]),
)
@settings(max_examples=70)
def test_routh_root_counts_and_power_of_two_scaling(designed_roots, scale):
    polynomial = np.poly(designed_roots)
    result = routh_hurwitz(polynomial * scale)
    assert result.rhp == sum(root > 0 for root in designed_roots)
    assert result.axis == designed_roots.count(0)


def test_assignment_minimizes_global_cost():
    costs = np.asarray([[1.0, 2.0, 7.0], [2.0, 90.0, 5.0], [3.0, 4.0, 3.0]])
    assignment = _assignment(costs)
    minimum = min(sum(costs[row, col] for row, col in enumerate(p)) for p in permutations(range(3)))
    assert len(set(assignment)) == 3
    assert costs[np.arange(3), assignment].sum() == minimum


def test_root_locus_analytic_crossing_and_repeated_roots():
    natural = TransferFunction([1], [1, 3, 3, 1], gain=3)
    result = root_locus(natural, [0, 7.9, 8.1, 12])
    assert len(result.crossings) == 2
    np.testing.assert_allclose([k for k, _ in result.crossings], [8, 8])
    np.testing.assert_allclose([pole for _, pole in result.crossings], [-1j * np.sqrt(3), 1j * np.sqrt(3)])
    np.testing.assert_allclose(result.roots[0], [-1, -1, -1], atol=9e-6)
    for gain, roots in zip(result.gains, result.roots, strict=True):
        assert np.max(np.abs(np.polyval(natural.characteristic(gain), roots))) < 1e-10


def test_root_locus_tracks_degree_drop():
    result = root_locus(TransferFunction([-1, 1], [1, 2]), [0, 0.5, 1, 1.5])
    assert result.degree_drops == (1.0,)
    assert np.isnan(result.roots[2, 0])
    np.testing.assert_allclose(result.roots[[0, 1, 3], 0], [-2, -5, 7])
    assert result.crossings == ()


def test_root_locus_axis_origin_and_rational_only():
    assert root_locus(TransferFunction([1], [1, 1, -2]), [0, 4]).crossings == ((2, 0j),)
    with pytest.raises(ValueError, match="Padé"):
        root_locus(TransferFunction([1], [1, 1], delay=0.5), [0, 1])
    for invalid in ([0, 0], [-1, 1], [1, 0], [np.nan], []):
        with pytest.raises(ValueError):
            root_locus(TransferFunction([1], [1, 1]), invalid)


def test_step_response_matches_exact_first_order_on_nonuniform_grid():
    times = np.asarray([0.13, 0.3, 0.31, 0.65, 4.0])
    response = step_response(TransferFunction([3], [1, 2], gain=2), times)
    np.testing.assert_allclose(response, 0.75 * (1 - np.exp(-8 * times)), atol=3e-14)


def test_step_response_repeated_closed_loop_poles_and_feedthrough():
    times = np.linspace(0, 8, 81)
    repeated = step_response(TransferFunction([1], [1, 4, 3]), times)
    np.testing.assert_allclose(repeated, (1 - (1 + 2 * times) * np.exp(-2 * times)) / 4, atol=5e-14)
    feedthrough = step_response(TransferFunction([2, 3], [1, 2]), times)
    np.testing.assert_allclose(feedthrough, 0.6 + (2 / 3 - 0.6) * np.exp(-5 * times / 3), atol=5e-14)


def test_step_response_marginal_origin_and_singular_loop():
    times = np.linspace(0, 6, 50)
    np.testing.assert_allclose(step_response(TransferFunction([1], [1, 0, -1]), times), times**2 / 2)
    with pytest.raises(ValueError, match="singular"):
        step_response(TransferFunction([-1, 1], [1, 2]), times)
    with pytest.raises(ValueError, match="crescentes"):
        step_response(TransferFunction([1], [1, 1]), [1, 1])


def test_root_locus_detects_tangency_off_the_supplied_gain_grid():
    system = TransferFunction([1, 1, 4], [1, 1, 1, 0])
    result = root_locus(system, [0.5, 1.5])
    assert len(result.crossings) == 2
    np.testing.assert_allclose([gain for gain, _ in result.crossings], [1, 1], atol=3e-8)
    np.testing.assert_allclose([pole for _, pole in result.crossings], [-1j * np.sqrt(2), 1j * np.sqrt(2)])
    for gain in result.gains:
        assert routh_hurwitz(system.characteristic(gain)).status == "stable"


def test_step_response_delayed_model_agrees_with_scipy():
    scipy_signal = pytest.importorskip("scipy.signal")
    control = pytest.importorskip("control")
    times = np.linspace(0, 8, 161)
    p, q = control.pade(0.8, 3)
    reference_numerator = np.asarray(p) * 2
    reference_denominator = np.polyadd(np.polymul(q, [1, 1]), reference_numerator)
    _, reference = scipy_signal.step((reference_numerator, reference_denominator), T=times)
    system = TransferFunction([2], [1, 1], delay=0.8)
    np.testing.assert_allclose(step_response(system, times), reference, atol=3e-12)


def test_step_response_zero_external_output_with_unstable_internal_mode():
    system = TransferFunction([0], [1, -100])
    np.testing.assert_equal(step_response(system, [0, 100]), [0, 0])
