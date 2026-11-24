"""Resposta em frequência e cruzamentos de Bode em malha aberta."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.typing import ArrayLike, NDArray

from .frequency_response import frequency_grid
from .transfer_function import TransferFunction

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_TWO_PI = 2 * np.pi
_CROSSING_TOL = 1e-8
_MAX_DELAY_STEPS = 50_000


class GainCrossing(NamedTuple):
    omega: float
    phase_margin_deg: float


class PhaseCrossing(NamedTuple):
    omega: float
    gain_margin: float

    @property
    def gain_margin_db(self) -> float:
        return 20 * math.log10(self.gain_margin)


@dataclass(frozen=True)
class BodeResult:
    omega: FloatArray
    magnitude_db: FloatArray
    phase_deg: FloatArray
    gain_crossings: tuple[GainCrossing, ...]
    phase_crossings: tuple[PhaseCrossing, ...]
    gain_continuum: bool = False
    phase_continuum: bool = False
    phase_band: tuple[float, float] | None = None
    phase_band_limited: bool = False


def _rational(system: TransferFunction) -> TransferFunction:
    if system.delay == 0:
        return system
    return TransferFunction(system.numerator, system.denominator, gain=system.gain)


def _scaled_polynomials(system: TransferFunction) -> tuple[FloatArray, FloatArray]:
    """Aplica a mesma escala a K B e A antes dos produtos quadráticos."""
    numerator = np.asarray(system.numerator, dtype=float)
    denominator = np.asarray(system.denominator, dtype=float)
    numerator_scale = float(np.max(np.abs(numerator)))
    denominator_scale = float(np.max(np.abs(denominator)))
    normalized_denominator = denominator / denominator_scale
    if system.gain == 0 or numerator_scale == 0:
        return np.zeros(1), normalized_denominator
    log_ratio = math.log(abs(system.gain)) + math.log(numerator_scale) - math.log(denominator_scale)
    sign = math.copysign(1.0, system.gain)
    normalized_numerator = sign * (numerator / numerator_scale)
    if log_ratio > 0:
        return normalized_numerator, normalized_denominator * math.exp(-log_ratio)
    return normalized_numerator * math.exp(log_ratio), normalized_denominator


def _even_odd(coefficients: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Decompõe P(jω) em E(ω²)+jω O(ω²), com E e O em ordem crescente."""
    ascending = coefficients[::-1]
    even = ascending[::2] * (-1.0) ** np.arange(ascending[::2].size)
    odd = ascending[1::2] * (-1.0) ** np.arange(ascending[1::2].size)
    return even, odd if odd.size else np.zeros(1)


def _magnitude_squared(even: FloatArray, odd: FloatArray) -> FloatArray:
    return np.asarray(
        poly.polyadd(poly.polymul(even, even), np.r_[0.0, poly.polymul(odd, odd)]), dtype=float
    )


def _frequencies(coefficients: FloatArray, *, include_extrema: bool = False) -> FloatArray:
    """Calcula candidatos com x=ω²; o chamador verifica na função original."""
    trimmed = np.trim_zeros(coefficients, "b")
    if trimmed.size < 2:
        return np.empty(0)
    roots = poly.polyroots(trimmed / np.max(np.abs(trimmed)))
    if include_extrema and trimmed.size > 2:
        derivative = poly.polyder(trimmed)
        roots = np.r_[roots, poly.polyroots(derivative / np.max(np.abs(derivative)))]
    real = np.real(roots)
    imag = np.abs(np.imag(roots))
    eligible = np.isfinite(real) & (real > 0) & (imag <= 1e-7 * real)
    return np.sqrt(real[eligible])


def _distinct(frequencies: ArrayLike) -> FloatArray:
    values = np.sort(np.asarray(frequencies, dtype=float))
    if values.size <= 1:
        return values
    close = np.diff(values) <= 1e-7 * np.maximum(values[1:], values[:-1])
    lower = values[np.r_[True, ~close]]
    upper = values[np.r_[~close, True]]
    return lower + (upper - lower) / 2


def _unity(system: TransferFunction) -> tuple[FloatArray, bool]:
    numerator, denominator = _scaled_polynomials(system)
    magnitude_n = _magnitude_squared(*_even_odd(numerator))
    magnitude_d = _magnitude_squared(*_even_odd(denominator))
    difference = np.asarray(poly.polysub(magnitude_n, magnitude_d), dtype=float)
    magnitude_scale: FloatArray = np.asarray(
        poly.polyadd(np.abs(magnitude_n), np.abs(magnitude_d)), dtype=float
    )
    difference_padded = np.pad(difference, (0, magnitude_scale.size - difference.size))
    continuum = bool(np.all(np.abs(difference_padded) <= 8 * np.finfo(float).eps * magnitude_scale))
    candidates = (
        np.r_[0.0, _frequencies(difference, include_extrema=True)] if not continuum else np.zeros(1)
    )
    response = _rational(system).evaluate(1j * candidates)
    valid = np.isfinite(response) & (np.abs(np.abs(response) - 1) <= _CROSSING_TOL)
    return _distinct(candidates[valid]), continuum


def gain_frequencies(system: TransferFunction) -> FloatArray:
    """Localiza frequências não negativas de ganho unitário, inclusive DC e tangências.

    Quando o ganho é unitário em toda a banda, retorna somente DC se definido.
    O indicador de contínuo está disponível em `bode(system).gain_continuum`.
    """
    return _unity(system)[0]


def _negative_real(response: ComplexArray) -> NDArray[np.bool_]:
    return (
        np.isfinite(response)
        & (np.real(response) < 0)
        & (np.abs(np.imag(response)) <= _CROSSING_TOL * np.abs(response))
    )


def _rational_phase_crossings(system: TransferFunction) -> tuple[FloatArray, bool]:
    numerator, denominator = _scaled_polynomials(system)
    en, on = _even_odd(numerator)
    ed, od = _even_odd(denominator)
    imaginary = np.asarray(poly.polysub(poly.polymul(on, ed), poly.polymul(en, od)), dtype=float)
    if not np.any(imaginary):
        axis_points = sorted(
            abs(root.imag)
            for root in np.r_[system.poles, system.zeros]
            if abs(root.real) <= 1e-8 * abs(root.imag) and root.imag != 0
        )
        bounds = np.asarray([0.0, *axis_points, max(axis_points, default=0.0) * 2 + 1.0])
        midpoints = (bounds[1:] + bounds[:-1]) / 2
        continuum = bool(np.any(_negative_real(system.evaluate(1j * midpoints))))
        dc = np.zeros(1)
        return dc[_negative_real(system.evaluate(1j * dc))], continuum

    candidates = np.r_[0.0, _frequencies(imaginary, include_extrema=True)]
    return _distinct(candidates[_negative_real(system.evaluate(1j * candidates))]), False


def _phase_numerator(even: FloatArray, odd: FloatArray) -> FloatArray:
    """Numerador de d arg P(jω)/dω após divisão por |P(jω)|²."""
    angular_term = poly.polysub(
        poly.polymul(even, poly.polyder(odd)),
        poly.polymul(odd, poly.polyder(even)),
    )
    return np.asarray(
        poly.polyadd(poly.polymul(even, odd), np.r_[0.0, 2 * angular_term]), dtype=float
    )


def _phase_extrema(system: TransferFunction) -> FloatArray:
    numerator, denominator = _scaled_polynomials(system)
    en, on = _even_odd(numerator)
    ed, od = _even_odd(denominator)
    mn = _magnitude_squared(en, on)
    md = _magnitude_squared(ed, od)
    scale = max(1.0, system.delay)
    derivative = poly.polysub(
        poly.polymul(_phase_numerator(en, on), md), poly.polymul(_phase_numerator(ed, od), mn)
    )
    numerator = poly.polysub(derivative / scale, (system.delay / scale) * poly.polymul(mn, md))
    return _frequencies(numerator)


def _delayed_phase_crossings(
    system: TransferFunction, display_omega: FloatArray
) -> tuple[FloatArray, tuple[float, float], bool]:
    max_allowed = _MAX_DELAY_STEPS * (np.pi / 4) / system.delay
    horizon = min(float(display_omega[-1]), max_allowed)
    steps = max(1, math.ceil(horizon * system.delay / (np.pi / 4)))
    extrema = _phase_extrema(system)
    features = [*extrema]
    for root in np.r_[system.poles, system.zeros]:
        width = abs(root.real)
        center = abs(root.imag)
        features.extend(center + width * np.array([-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0]))
    features_array = np.asarray(features)
    features_array = features_array[(features_array > 0) & (features_array < horizon)]
    mesh = np.unique(
        np.r_[
            np.linspace(0, horizon, steps + 1),
            features_array,
            display_omega[display_omega < horizon],
        ]
    )
    response = system.evaluate(1j * mesh)
    imaginary = np.imag(response)
    finite = np.isfinite(response)
    bracket = finite[:-1] & finite[1:] & (np.signbit(imaginary[:-1]) != np.signbit(imaginary[1:]))
    lower, upper = mesh[:-1][bracket], mesh[1:][bracket]
    lower_sign = np.signbit(imaginary[:-1][bracket])
    if len(lower):
        for _ in range(52):
            midpoint = lower + (upper - lower) / 2
            same_sign = np.signbit(np.imag(system.evaluate(1j * midpoint))) == lower_sign
            lower = np.where(same_sign, midpoint, lower)
            upper = np.where(same_sign, upper, midpoint)
    refined = (lower + upper) / 2
    # Deixa o bracketing fixar raízes simples e verifica separadamente tangências e DC.
    candidates = np.r_[refined, 0.0, horizon, extrema[(extrema > 0) & (extrema < horizon)]]
    valid = _negative_real(system.evaluate(1j * candidates))
    return _distinct(candidates[valid]), (0.0, horizon), horizon < display_omega[-1]


def _continuous_phase(system: TransferFunction, omega: FloatArray) -> FloatArray:
    rational = _rational(system)
    response = rational.evaluate(1j * omega)
    valid = np.isfinite(response) & (np.abs(response) > 0)
    if not np.any(valid):
        return np.full(omega.size, np.nan)
    direct = np.angle(response)
    leading_sign = math.copysign(1.0, rational.gain) * math.copysign(1.0, rational.numerator[0])
    factored_dc = np.pi if leading_sign < 0 else 0.0
    factor_phase = np.full(omega.size, factored_dc)
    # Cada fator tem uma fase contínua, salvo sobre seu próprio zero no eixo.
    for roots, sign in ((rational.zeros, 1), (rational.poles, -1)):
        for root in roots:
            delta = omega - root.imag
            angles = np.arctan2(delta, -root.real)
            angles = angles + _TWO_PI * ((root.real > 0) & (delta < 0))
            factor_phase += sign * angles
            if root.imag == 0:
                angle_dc = np.pi / 2 if root == 0 else np.pi if root.real > 0 else 0.0
            else:
                angle_dc = np.arctan2(-root.imag, -root.real)
                angle_dc += _TWO_PI * (root.real > 0 and root.imag > 0)
            factored_dc += sign * angle_dc
    lowest_n = np.trim_zeros(rational.numerator, "b")
    lowest_d = np.trim_zeros(rational.denominator, "b")
    dc_sign = math.copysign(1.0, rational.gain) * math.copysign(1.0, lowest_n[-1])
    dc_sign *= math.copysign(1.0, lowest_d[-1])
    origin_zeros = len(rational.numerator) - len(lowest_n)
    origin_poles = len(rational.denominator) - len(lowest_d)
    dc_phase = (0.0 if dc_sign > 0 else -np.pi) + (origin_zeros - origin_poles) * np.pi / 2
    branch = np.rint((factored_dc - dc_phase) / _TWO_PI)
    turns = np.rint((factor_phase - direct) / _TWO_PI)
    phase = direct + _TWO_PI * (turns - branch) - system.delay * omega
    return np.where(valid, np.degrees(phase), np.nan)


def bode(system: TransferFunction, omega: ArrayLike | None = None) -> BodeResult:
    """Calcula Bode e margens; as passagens de fase atrasadas têm banda explícita."""
    if omega is not None and np.iscomplexobj(omega):
        raise ValueError("As frequências devem ser reais.")
    frequencies = np.asarray(frequency_grid(system) if omega is None else omega, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies < 0)
        or np.any(np.diff(frequencies) <= 0)
    ):
        raise ValueError("As frequências devem ser finitas, crescentes e não negativas.")
    gain_omega, gain_continuum = _unity(system)

    phase_continuum = False
    phase_band = None
    phase_band_limited = False
    if system.delay:
        phase_omega, phase_band, phase_band_limited = _delayed_phase_crossings(system, frequencies)
    else:
        phase_omega, phase_continuum = _rational_phase_crossings(system)

    gains_at_crossing = system.evaluate(1j * gain_omega)
    margins_deg = np.degrees(np.angle(-gains_at_crossing))
    phase_magnitude = np.abs(system.evaluate(1j * phase_omega))
    gain_crossings = tuple(
        GainCrossing(float(w), 180.0 if margin <= -180.0 else float(margin))
        for w, margin in zip(gain_omega, margins_deg, strict=True)
    )
    phase_crossings = tuple(
        PhaseCrossing(float(w), float(1 / magnitude))
        for w, magnitude in zip(phase_omega, phase_magnitude, strict=True)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude_db = 20 * np.log10(np.abs(system.evaluate(1j * frequencies)))
    return BodeResult(
        frequencies,
        magnitude_db,
        _continuous_phase(system, frequencies),
        gain_crossings,
        phase_crossings,
        gain_continuum,
        phase_continuum,
        phase_band,
        phase_band_limited,
    )
