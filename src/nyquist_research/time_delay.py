"""Fronteiras de estabilidade por atraso puro no laço de realimentação."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bode import gain_frequencies
from .transfer_function import TransferFunction


@dataclass(frozen=True)
class DelayCandidate:
    delay: float
    frequency: float
    direction: str


@dataclass(frozen=True)
class DelayResult:
    delay: float | None
    frequency: float | None
    status: str
    reason: str
    candidates: tuple[DelayCandidate, ...] = ()


def _crossing_direction(system: TransferFunction, omega: float, delay: float) -> str:
    s = 1j * omega
    log_derivative = np.sum(1 / (s - system.zeros)) - np.sum(1 / (s - system.poles))
    denominator = log_derivative - delay
    tolerance = 64 * np.finfo(float).eps
    if abs(denominator) <= tolerance * max(abs(log_derivative), abs(delay)):
        return "multiple"
    derivative = s / denominator
    if not np.isfinite(derivative):
        return "indeterminate"
    if abs(derivative.real) <= 1e-8 * abs(derivative):
        return "tangent"
    return "destabilizing" if derivative.real > 0 else "stabilizing"


def critical_delay(system: TransferFunction) -> DelayResult:
    """Retorna o primeiro atraso absoluto de fronteira a partir de τ=0 estável."""
    from .routh_hurwitz import routh_hurwitz

    no_delay = TransferFunction(system.numerator, system.denominator, gain=system.gain)
    try:
        characteristic = no_delay.characteristic()
        if characteristic[0] == 0:
            raise ValueError("A equação característica perde ordem em τ=0.")
        baseline = routh_hurwitz(characteristic)
    except ValueError as error:
        return DelayResult(None, None, "indeterminate", str(error))
    if baseline.status != "stable":
        return DelayResult(
            None,
            None,
            "indeterminate",
            f"A hipótese de malha assintoticamente estável em τ=0 falhou ({baseline.status}).",
        )

    if len(system.numerator) == len(system.denominator):
        asymptotic_gain = abs(system.gain * (system.numerator[0] / system.denominator[0]))
        if np.isclose(asymptotic_gain, 1, rtol=1e-10, atol=0):
            return DelayResult(
                None,
                None,
                "indeterminate",
                "Ganho assintótico unitário em malha com atraso tipo neutro.",
            )
        if asymptotic_gain > 1:
            return DelayResult(
                0.0,
                None,
                "finite",
                "Para todo atraso positivo, há polos assintóticos no semiplano direito.",
            )

    frequencies = gain_frequencies(no_delay)
    frequencies = frequencies[frequencies > 0]
    if not len(frequencies):
        return DelayResult(
            None,
            None,
            "unbounded",
            "A malha de atraso zero é estável e não há cruzamentos unitários positivos.",
        )

    responses = no_delay.evaluate(1j * frequencies)
    phase_offsets = np.mod(np.angle(-responses), 2 * np.pi)
    candidates = sorted(
        (
            DelayCandidate(
                float(offset / omega),
                float(omega),
                _crossing_direction(no_delay, omega, offset / omega),
            )
            for omega, offset in zip(frequencies, phase_offsets, strict=True)
        ),
        key=lambda candidate: candidate.delay,
    )
    first = candidates[0]
    if first.direction == "destabilizing":
        reason = "Primeira raiz imaginária positiva com migração para o semiplano direito."
    else:
        reason = f"Primeira fronteira encontrada. Direção local: {first.direction}."
    return DelayResult(first.delay, first.frequency, "finite", reason, tuple(candidates))
