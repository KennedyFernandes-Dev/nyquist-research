"""Lugar das raízes racional por continuidade e equações de fronteira."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transfer_function import TransferFunction, horner


@dataclass(frozen=True)
class RootLocusResult:
    gains: NDArray[np.float64]
    roots: NDArray[np.complex128]
    crossings: tuple[tuple[float, complex], ...]
    degree_drops: tuple[float, ...]
    events: tuple[str, ...] = ()


def _assignment(cost: NDArray[np.float64]) -> NDArray[np.intp]:
    """Atribuição quadrada de custo mínimo pelo algoritmo húngaro primal-dual."""
    size = cost.shape[0]
    row_potential = np.zeros(size + 1)
    column_potential = np.zeros(size + 1)
    paired = np.zeros(size + 1, dtype=np.intp)
    predecessor = np.zeros(size + 1, dtype=np.intp)
    for row in range(1, size + 1):
        paired[0] = row
        column = 0
        reduced_min = np.full(size + 1, np.inf)
        visited = np.zeros(size + 1, dtype=bool)
        while True:
            visited[column] = True
            active_row = paired[column]
            reduced = cost[active_row - 1] - row_potential[active_row] - column_potential[1:]
            improved = (~visited[1:]) & (reduced < reduced_min[1:])
            reduced_min[1:][improved] = reduced[improved]
            predecessor[1:][improved] = column
            next_column = int(np.argmin(np.where(visited, np.inf, reduced_min)))
            delta = reduced_min[next_column]
            row_potential[paired[visited]] += delta
            column_potential[visited] -= delta
            reduced_min[~visited] -= delta
            column = next_column
            if paired[column] == 0:
                break
        while column:
            previous_column = predecessor[column]
            paired[column] = paired[previous_column]
            column = previous_column
    assignment = np.empty(size, dtype=np.intp)
    assignment[paired[1:] - 1] = np.arange(size, dtype=np.intp)
    return assignment


def _link(
    previous: NDArray[np.complex128], current: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    valid_previous = np.isfinite(previous)
    valid_current = np.isfinite(current)
    # A escala comum preserva a minimização e limita o custo mesmo perto do infinito.
    scale = max(
        float(np.max(np.abs(previous[valid_previous]), initial=1.0)),
        float(np.max(np.abs(current[valid_current]), initial=1.0)),
    )
    old = np.where(valid_previous, previous / scale, 0)
    new = np.where(valid_current, current / scale, 0)
    cost = np.abs(old[:, None] - new[None, :]) ** 2
    cost[valid_previous[:, None] != valid_current[None, :]] = 5
    return current[_assignment(cost)]


def _roots_at(
    system: TransferFunction, gain: float, singular_gain: float | None
) -> NDArray[np.complex128]:
    polynomial = system.characteristic(gain)
    if gain == singular_gain:
        polynomial[0] = 0.0
    polynomial = np.trim_zeros(polynomial, "f")
    values = np.asarray(np.roots(polynomial), dtype=np.complex128)
    return np.pad(values, (0, system.order - values.size), constant_values=complex(np.nan, np.nan))


def _boundary_events(
    system: TransferFunction, minimum: float, maximum: float
) -> tuple[tuple[tuple[float, complex], ...], tuple[str, ...]]:
    numerator = np.asarray(system.numerator)
    denominator = np.asarray(system.denominator)
    numerator_scale = float(np.max(np.abs(numerator)))
    if numerator_scale == 0:
        return (), ("Numerador nulo. O ganho não desloca os polos.",)
    a = denominator / np.max(np.abs(denominator))
    b = numerator / numerator_scale
    b_reflected = b * (-1) ** np.arange(b.size - 1, -1, -1)
    odd = np.convolve(a, b_reflected)[::-1][1::2]
    odd_in_w_squared = np.trim_zeros((odd * (-1) ** np.arange(odd.size))[::-1], "f")
    frequencies = [0.0]
    events: list[str] = []
    if odd_in_w_squared.size == 0:
        events.append(
            "Equação de fase degenerada. Pode haver polos no eixo em faixas contínuas de ganho."
        )
    for candidate in np.roots(odd_in_w_squared):
        if candidate.real <= 0 or abs(candidate.imag) > 1e-7 * candidate.real:
            continue
        omega = sqrt(float(candidate.real))
        if not any(np.isclose(omega, other, rtol=1e-7, atol=0) for other in frequencies):
            frequencies.append(omega)

    crossings = []
    for omega in frequencies:
        pole = complex(0, omega)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if omega > 1:
                inverse = 1 / pole
                open_numerator = complex(horner(numerator[::-1], inverse))
                open_denominator = complex(horner(denominator[::-1], inverse))
                open_numerator *= inverse ** (denominator.size - numerator.size)
            else:
                open_numerator = complex(horner(numerator, pole))
                open_denominator = complex(horner(denominator, pole))
            if not np.isfinite(open_denominator) or not np.isfinite(open_numerator):
                continue
            if open_numerator == 0:
                if open_denominator == 0:
                    events.append(f"Polo em j{omega:.8g} permanece no eixo por fator comum.")
                continue
            crossing = -open_denominator / open_numerator
        if not np.isfinite(crossing) or abs(crossing.imag) > 1e-7 * max(abs(crossing.real), 1e-12):
            continue
        gain = crossing.real
        window_tolerance = 1e-10 * max(1.0, abs(gain))
        if not minimum - window_tolerance <= gain <= maximum + window_tolerance:
            continue
        gain = min(maximum, max(minimum, gain))
        residual = abs(open_denominator + gain * open_numerator)
        scale = abs(open_denominator) + abs(gain * open_numerator)
        if residual > 1e-7 * max(scale, 1e-300):
            continue
        crossings.append((gain, pole))
        if omega > 0:
            crossings.append((gain, pole.conjugate()))
    return tuple(sorted(crossings, key=lambda entry: (entry[0], entry[1].imag))), tuple(events)


def root_locus(system: TransferFunction, gains: ArrayLike) -> RootLocusResult:
    """Calcula polos de A+KB, conectados por uma atribuição de custo mínimo.

    Os ramos são discretizados nos ganhos fornecidos. Singularidades da equação
    característica dividem o traçado e aparecem em `degree_drops`. Uma célula
    NaN sinaliza um ramo ausente em um ganho que reduz o grau.
    """
    if system.delay:
        raise ValueError("O lugar das raízes finito requer um modelo racional. Selecione Padé.")
    raw = np.asarray(gains)
    if np.iscomplexobj(raw):
        raise ValueError("Informe ganhos reais.")
    grid = np.asarray(raw, dtype=float)
    if (
        grid.ndim != 1
        or not grid.size
        or not np.all(np.isfinite(grid))
        or np.any(grid < 0)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("Informe ganhos finitos, não negativos e estritamente crescentes.")
    first, last = float(grid[0]), float(grid[-1])
    singular_gain = None
    if len(system.numerator) == len(system.denominator) and system.numerator[0] < 0:
        candidate = -system.denominator[0] / system.numerator[0]
        if first <= candidate <= last:
            singular_gain = candidate
    roots = np.empty((grid.size, system.order), dtype=np.complex128)
    for index, gain in enumerate(grid):
        current = _roots_at(system, float(gain), singular_gain)
        if index and system.order:
            current = _link(roots[index - 1], current)
        else:
            current = np.sort_complex(current)
        roots[index] = current
    crossings, events = _boundary_events(system, first, last)
    degree_drops = () if singular_gain is None else (singular_gain,)
    if singular_gain is not None:
        singular = system.characteristic(singular_gain)
        singular[0] = 0.0
        if not np.any(singular):
            events += ("Equação característica identicamente nula no ganho singular.",)
    return RootLocusResult(grid.copy(), roots, crossings, degree_drops, events)
