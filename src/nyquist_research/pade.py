"""Aproximação racional diagonal de atrasos puros."""

from __future__ import annotations

import math
import operator

import numpy as np
from numpy.typing import NDArray

from .transfer_function import TransferFunction


def pade(delay: float, order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Retorna os coeficientes decrescentes de Padé [ordem/ordem] de e^(-s τ)."""
    tau = float(delay)
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("O atraso deve ser finito e não negativo.")
    n = operator.index(order)
    if isinstance(order, bool) or n < 0:
        raise ValueError("A ordem de Padé deve ser um inteiro não negativo.")
    if n == 0 or tau == 0:
        return np.ones(1), np.ones(1)

    coefficients = [1.0]
    for k in range(n):
        ratio = (n - k) / ((k + 1) * (2 * n - k))
        coefficient = coefficients[-1] * (tau * ratio)
        if coefficient == 0 or not math.isfinite(coefficient):
            raise ValueError("Os coeficientes de Padé excedem a faixa de float64.")
        coefficients.append(coefficient)
    denominator = np.asarray(coefficients[::-1])
    signs = np.where(np.arange(n, -1, -1) % 2, -1.0, 1.0)
    return signs * denominator, denominator


def approximate_delay(system: TransferFunction, order: int) -> TransferFunction:
    """Incorpora Padé ao numerador e denominador e preserva o ganho nominal."""
    numerator, denominator = pade(system.delay, order)
    return TransferFunction(
        np.convolve(system.numerator, numerator),
        np.convolve(system.denominator, denominator),
        gain=system.gain,
    )
