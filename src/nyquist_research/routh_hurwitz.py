"""Tabela de Routh para polinômios reais em ordem decrescente."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class RouthResult:
    table: NDArray[np.float64]
    rhp: int | None
    axis: int
    status: str
    events: tuple[str, ...]


def _floating_table(
    coefficients: NDArray[np.float64], tolerance: float
) -> NDArray[np.float64] | None:
    order = coefficients.size - 1
    width = (order + 2) // 2
    table = np.zeros((order + 1, width))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        normalized = coefficients / coefficients[0]
    if not np.all(np.isfinite(normalized)) or np.any(
        (coefficients != 0) & (normalized == 0)
    ):
        return None
    table[0] = normalized[::2]
    if order == 0:
        return table
    table[1, : normalized[1::2].size] = normalized[1::2]
    for row in range(2, order + 1):
        previous, above = table[row - 1], table[row - 2]
        if abs(previous[0]) <= tolerance * np.max(np.abs(previous)):
            return None
        scale = max(abs(previous[0]), abs(above[0]))
        left = previous[0] / scale * above[1:]
        right = above[0] / scale * previous[1:]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            difference = left - right
            terms = np.abs(left) + np.abs(right)
            cancellation = np.abs(difference) <= tolerance * terms
            if np.any(cancellation & ((left != 0) | (right != 0))):
                return None
            table[row, :-1] = difference / (previous[0] / scale)
        if not np.all(np.isfinite(table[row])):
            return None
    if np.any(table[:, 0] == 0):
        return None
    return table


def _exact_table(
    coefficients: list[Fraction], epsilon: Fraction
) -> tuple[list[list[Fraction]], tuple[str, ...]]:
    order = len(coefficients) - 1
    width = (order + 2) // 2
    zero = Fraction(0)
    table = [[zero] * width for _ in range(order + 1)]
    table[0][: len(coefficients[::2])] = coefficients[::2]
    if order > 0:
        table[1][: len(coefficients[1::2])] = coefficients[1::2]
    events = []
    for row in range(1, order + 1):
        if row >= 2:
            previous, above = table[row - 1], table[row - 2]
            table[row][:-1] = [
                upper - above[0] * lower / previous[0]
                for upper, lower in zip(above[1:], previous[1:], strict=True)
            ]
        if not any(table[row]):
            auxiliary_degree = order - row + 1
            table[row] = [
                coefficient * (auxiliary_degree - 2 * column)
                for column, coefficient in enumerate(table[row - 1])
            ]
            events.append(f"Linha nula em s^{order - row}, substituída pela derivada do auxiliar.")
        if table[row][0] == 0:
            row_scale = max(map(abs, table[row]))
            table[row][0] = epsilon * row_scale
            events.append(f"Pivô nulo em s^{order - row}, tratado por epsilon positivo.")
    return table, tuple(events)


def _polynomial_remainder(
    dividend: list[Fraction], divisor: list[Fraction]
) -> list[Fraction]:
    """Resto com coeficientes em ordem crescente, descartando zeros dominantes."""
    remainder = dividend.copy()
    while remainder and len(remainder) >= len(divisor):
        offset = len(remainder) - len(divisor)
        factor = remainder[-1] / divisor[-1]
        for index, value in enumerate(divisor):
            remainder[offset + index] -= factor * value
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return remainder


def _variations(values: list[int]) -> int:
    signs = [value for value in values if value]
    return sum(first * second < 0 for first, second in pairwise(signs))


def _sign(value: float | Fraction) -> int:
    return int(value > 0) - int(value < 0)


def _derivative(poly: list[Fraction]) -> list[Fraction]:
    return [degree * coefficient for degree, coefficient in enumerate(poly) if degree]


def _axis_pairs(coefficients: list[Fraction]) -> int:
    """Raízes negativas comuns das partes par e ímpar em t=s², com multiplicidade."""
    even, odd = coefficients[::-1][::2], coefficients[::-1][1::2]
    while even and even[-1] == 0:
        even.pop()
    while odd and odd[-1] == 0:
        odd.pop()
    while odd:
        even, odd = odd, _polynomial_remainder(even, odd)
    count = 0
    while len(even) > 1:
        # Sturm conta raízes distintas no intervalo (-infinito, 0).
        previous, current = even, _derivative(even)
        sequence = [previous, current]
        while remainder := _polynomial_remainder(previous, current):
            previous, current = current, [-value for value in remainder]
            sequence.append(current)
        signs_infinity = [_sign(poly[-1]) * (-1) ** (len(poly) - 1) for poly in sequence]
        signs_zero = []
        for polynomial in sequence:
            power, coefficient = next(
                (index, value) for index, value in enumerate(polynomial) if value
            )
            signs_zero.append(_sign(coefficient) * (-1) ** power)
        count += _variations(signs_infinity) - _variations(signs_zero)
        # O último divisor é o mdc de p e p', que retém uma cópia a menos.
        even = current
    return count


def _display_table(rows: list[list[Fraction]]) -> NDArray[np.float64]:
    # Mantém as entradas usuais e reescala apenas linhas além do alcance do float64.
    max_float = Fraction(float(np.finfo(float).max))
    values = []
    for row in rows:
        largest = max(map(abs, row))
        factor = largest if largest > max_float else Fraction(1)
        values.append([float(value / factor) for value in row])
    return np.asarray(values, dtype=float)


def routh_hurwitz(coefficients: ArrayLike, *, tolerance: float = 64 * np.finfo(float).eps) -> RouthResult:
    """Conta raízes à direita via primeira coluna e raízes no eixo separadamente.

    A tabela refere-se ao polinômio após deflação das raízes exatas na origem.
    Se há cancelamento numérico ou pivô nulo, a tabela é refeita com frações
    binárias exatas dos coeficientes fornecidos. Em pivôs isolados comparam-se
    duas substituições positivas pequenas para detectar sensibilidade a epsilon.
    """
    raw = np.asarray(coefficients)
    if np.iscomplexobj(raw):
        raise ValueError("Os coeficientes devem ser reais.")
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Forneça um vetor finito de coeficientes reais.")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("A tolerância deve estar entre zero e um.")
    occupied = np.flatnonzero(values)
    if not occupied.size:
        raise ValueError("O polinômio nulo não tem uma tabela de Routh definida.")
    origins = int(values.size - occupied[-1] - 1)
    values = values[occupied[0] : occupied[-1] + 1]
    events: tuple[str, ...] = (
        (f"Raiz na origem com multiplicidade {origins}.",) if origins else ()
    )
    table = _floating_table(values, tolerance)
    rhp: int | None
    axis = origins
    if table is None:
        exact = [Fraction(float(value)) for value in values]
        exact = [value / exact[0] for value in exact]
        rows, special = _exact_table(exact, Fraction(1, 10**48))
        rhp = _variations([_sign(row[0]) for row in rows])
        if any(event.startswith("Pivô") for event in special):
            trial, _ = _exact_table(exact, Fraction(1, 10**24))
            if rhp != _variations([_sign(row[0]) for row in trial]):
                rhp = None
                events += ("A contagem varia com epsilon.",)
        table = _display_table(rows)
        axis += 2 * _axis_pairs(exact)
        events += special or ("Tabela recalculada em frações para reduzir cancelamento.",)
    else:
        rhp = _variations([_sign(float(value)) for value in table[:, 0]])
    status = (
        "indeterminate" if rhp is None else "unstable" if rhp else "marginal" if axis else "stable"
    )
    return RouthResult(table, rhp, axis, status, events)
