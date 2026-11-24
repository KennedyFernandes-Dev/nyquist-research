"""Degrau unitário em realimentação negativa, com realização companheira."""

from math import frexp, ldexp

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transfer_function import TransferFunction


def _matrix_exponential(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Taylor para norma pequena, seguida de quadraturas sucessivas."""
    norm = float(np.linalg.norm(matrix, ord=np.inf))
    if not np.isfinite(norm):
        raise ValueError("O intervalo temporal produz uma matriz fora da escala suportada.")
    scale = max(0, frexp(norm)[1] + 1) if norm else 0
    matrix = matrix * ldexp(1.0, -scale)
    term = np.eye(matrix.shape[0])
    result = term.copy()
    for order in range(1, 32):
        term = (term @ matrix) / order
        result += term
        if np.linalg.norm(term, ord=np.inf) <= np.finfo(float).eps * np.linalg.norm(
            result, ord=np.inf
        ):
            break
    for _ in range(scale):
        result = result @ result
    return result


def step_response(
    system: TransferFunction, time: ArrayLike, pade_order: int = 3
) -> NDArray[np.float64]:
    """Obtém o degrau em malha fechada com condições iniciais nulas.

    Se houver atraso, o cálculo usa Padé da ordem `pade_order`. A resposta
    calculada pode conter a antecipação inicial própria de um modelo racional.
    """
    raw = np.asarray(time)
    if np.iscomplexobj(raw):
        raise ValueError("Os instantes de tempo devem ser reais.")
    samples = np.asarray(raw, dtype=float)
    if (
        samples.ndim != 1
        or not samples.size
        or not np.all(np.isfinite(samples))
        or np.any(samples < 0)
        or np.any(np.diff(samples) <= 0)
    ):
        raise ValueError("Use instantes finitos, não negativos e estritamente crescentes.")

    if system.gain == 0 or system.numerator == (0.0,):
        return np.zeros_like(samples)
    if system.delay:
        from .pade import approximate_delay

        system = approximate_delay(system, pade_order)

    denominator = system.characteristic()
    if denominator[0] == 0:
        raise ValueError("Laço algébrico singular. A resposta ao degrau não é própria ou definida.")
    order = denominator.size - 1
    numerator = np.zeros_like(denominator)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            numerator[-len(system.numerator) :] = system.gain * np.asarray(system.numerator)
            numerator /= denominator[0]
            denominator = denominator / denominator[0]
            direct = numerator[0]
            if order == 0:
                return np.full(samples.size, direct)

            # Estado aumentado por uma constante unitária para integrar a entrada.
            augmented = np.zeros((order + 1, order + 1))
            augmented[0, :order] = -denominator[1:]
            augmented[0, -1] = 1
            augmented[1:order, : order - 1] = np.eye(order - 1)
            output = numerator[1:] - direct * denominator[1:]
            if not np.any(output):
                return np.full(samples.size, direct)
            state = np.zeros(order + 1)
            state[-1] = 1
            values = np.empty_like(samples)
            delta_reference = 0.0
            transition = np.eye(order + 1)
            previous = 0.0
            for index, instant in enumerate(samples):
                delta = float(instant - previous)
                if not np.isclose(delta, delta_reference, rtol=1e-12, atol=0):
                    transition = _matrix_exponential(augmented * delta)
                    delta_reference = delta
                state = transition @ state
                values[index] = output @ state[:order] + direct
                previous = float(instant)
    except FloatingPointError as error:
        raise ValueError("Resposta ao degrau excedeu a escala numérica suportada.") from error
    return values
