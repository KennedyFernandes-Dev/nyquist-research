"""Independent oracles for the main SISO experiments."""

from math import acos, pi, sqrt

import control as ct
import mpmath as mp
import numpy as np
import sympy as sp
from scipy import optimize, signal

from nyquist_research.transfer_function import TransferFunction


def rational_model(system: TransferFunction) -> ct.TransferFunction:
    """Explicitly omit the pure delay from the rational control object."""
    return ct.tf(system.gain * np.asarray(system.numerator), system.denominator)


def frequency_reference(system: TransferFunction, omega: np.ndarray) -> np.ndarray:
    """SciPy evaluates the rational part; the model's delay is analytic."""
    model = signal.TransferFunction(
        system.gain * np.asarray(system.numerator), system.denominator
    )
    _, response = signal.freqresp(model, w=omega)
    return response * np.exp(-1j * omega * system.delay)


def match_roots(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Find a minimum-distance pairing independently of the numerical core."""
    left = np.asarray(observed, dtype=complex)
    right = np.asarray(reference, dtype=complex)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Conjuntos de raízes incompatíveis.")
    if not left.size:
        return np.empty(0)
    distances = abs(left[:, None] - right[None, :])
    rows, columns = optimize.linear_sum_assignment(distances)
    return distances[rows, columns]


def root_error(observed: np.ndarray, reference: np.ndarray) -> float:
    distances = match_roots(observed, reference)
    return float(np.max(distances, initial=0.0))


def symbolic_roots(coefficients: np.ndarray) -> np.ndarray:
    """SymPy numerically solves a polynomial reconstructed as exact rationals."""
    x = sp.Symbol("s")
    polynomial = sp.Poly.from_list([sp.Rational(float(c)) for c in coefficients], x)
    return np.asarray([complex(r) for r in polynomial.nroots(n=35, maxsteps=200)])


def d_delay_analytic() -> tuple[float, float]:
    """First imaginary-axis contact for L=2 exp(-tau*s)/(s+1)."""
    omega = sqrt(3)
    return acos(-0.5) / omega, omega


def d_lambert_roots(delay: float, branches: int = 10) -> np.ndarray:
    """Return sampled Lambert W branches of s+1+2 exp(-tau*s)=0."""
    if delay == 0:
        return np.asarray([-3.0 + 0j])
    with mp.workdps(65):
        tau = mp.mpf(delay)
        argument = -2 * tau * mp.exp(tau)
        return np.asarray(
            [
                complex(mp.lambertw(argument, branch) / tau - 1)
                for branch in range(-branches, branches + 1)
            ]
        )


def d_rhp_analytic(delay: float, tolerance: float = 1e-10) -> tuple[int, str]:
    """Count crossings exactly by their increasing delay and RHP direction."""
    critical, frequency = d_delay_analytic()
    period = 2 * pi / frequency
    location = (delay - critical) / period
    nearest = round(location)
    if nearest >= 0 and abs(location - nearest) <= tolerance:
        return 2 * nearest, "marginal" if nearest == 0 else "unstable"
    count = 2 * max(0, int(np.floor(location)) + 1)
    return count, "unstable" if count else "stable"


def pade_reference(delay: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Padé from a high-precision Taylor-series linear solve via Mpmath."""
    with mp.workdps(80):
        coefficients = mp.taylor(lambda s: mp.exp(-mp.mpf(delay) * s), 0, 2 * order)
        numerator, denominator = mp.pade(coefficients, order, order)
        return np.asarray(numerator[::-1], dtype=float), np.asarray(denominator[::-1], dtype=float)
