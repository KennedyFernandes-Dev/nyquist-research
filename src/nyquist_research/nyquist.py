"""Nyquist contour in the right half plane and numerical winding."""

from dataclasses import dataclass
from math import ceil, pi

import numpy as np
from numpy.typing import NDArray

from .transfer_function import TransferFunction
from .winding_number import Winding, winding_number

_COMPLEX = NDArray[np.complex128]


@dataclass(frozen=True)
class NyquistResult:
    p: int
    n: int | None
    z: int | None
    status: str
    contour: _COMPLEX
    mapped: _COMPLEX
    radius: float
    min_distance: float | None
    reason: str
    max_angle: float | None = None
    refinements: int = 0


def _outer_radius(system: TransferFunction, poles: _COMPLEX) -> float | None:
    """Bound the exterior by a disk that excludes the critical point."""
    radius = max(1.0, 2 * float(np.max(np.abs(poles), initial=0.0)))
    a = np.asarray(system.denominator)
    b = np.pad(np.asarray(system.numerator) * system.gain, (len(a) - len(system.numerator), 0))
    feedthrough = float(b[0])
    if system.delay == 0:
        b = b - feedthrough * a
        distance = abs(1 + feedthrough)
    else:
        distance = 1.0
        if abs(feedthrough) >= distance:
            return None
    if distance == 0:
        return None
    powers = np.arange(len(a))
    for _ in range(60):
        inverse = (1 / radius) ** powers
        denominator_floor = 1 - np.dot(np.abs(a[1:]), inverse[1:])
        numerator_ceiling = np.dot(np.abs(b), inverse)
        if denominator_floor > 0 and numerator_ceiling < 0.85 * distance * denominator_floor:
            return radius
        radius *= 2
    return None


def _axis_centers(poles: _COMPLEX, tol: float) -> NDArray[np.float64]:
    on_axis = np.abs(poles.real) <= tol * np.maximum(1, np.abs(poles))
    candidates = sorted(poles.imag[on_axis])
    centers: list[float] = []
    for value in candidates:
        if not centers or abs(value - centers[-1]) > 10 * tol * max(1, abs(value)):
            centers.append(float(value))
    return np.asarray(centers)


def _indent_radii(
    axis: NDArray[np.float64], poles: _COMPLEX, radius: float, tol: float
) -> NDArray[np.float64]:
    radii = []
    for center in axis:
        distances = np.abs(poles - 1j * center)
        neighbors = distances[distances > tol * max(1.0, abs(center))]
        scale = min(radius / 16, 0.02 * max(1.0, abs(center)))
        if neighbors.size:
            scale = min(scale, float(neighbors.min()) / 4)
        radii.append(scale)
    return np.asarray(radii)


def _contour(
    system: TransferFunction,
    radius: float,
    points: int,
    axis: NDArray[np.float64],
    radii: NDArray[np.float64],
    markers: NDArray[np.float64],
) -> _COMPLEX:
    log_axis = np.geomspace(radius * 1e-13, radius, points)
    count = max(8, points // 8, ceil(8 * system.delay * radius))
    all_imag = np.unique(
        np.r_[-log_axis, 0, log_axis, np.linspace(-radius, radius, count), markers]
    )
    parts = []
    lower = -radius
    for center, rho in zip(axis, radii, strict=True):
        upper = center - rho
        inside = all_imag[(all_imag > lower) & (all_imag < upper)]
        parts.append(1j * np.r_[lower, inside, upper])
        theta = np.linspace(-pi / 2, pi / 2, max(24, points // 8))
        parts.append(1j * center + rho * np.exp(1j * theta[1:-1]))
        lower = center + rho
    inside = all_imag[(all_imag > lower) & (all_imag < radius)]
    parts.append(1j * np.r_[lower, inside, radius])
    theta = np.linspace(pi / 2, -pi / 2, max(32, points // 4))
    parts.append(radius * np.exp(1j * theta[1:-1]))
    parts.append(np.asarray([-1j * radius]))
    return np.concatenate(parts)


def _refine_mapping(system: TransferFunction, contour: _COMPLEX) -> tuple[_COMPLEX, _COMPLEX, bool]:
    """Bisect edges near the critical point or with rapidly turning images."""
    values = system.evaluate(contour)
    for _ in range(20):
        midpoints = contour[:-1] + np.diff(contour) / 2
        mid_values = system.evaluate(midpoints)
        left = 1 + values[:-1]
        mid = 1 + mid_values
        right = 1 + values[1:]
        with np.errstate(invalid="ignore", over="ignore"):
            am, bm, cm = np.abs(left), np.abs(mid), np.abs(right)
            first = np.angle(mid) - np.angle(left)
            second = np.angle(right) - np.angle(mid)
            angles = np.maximum(
                np.abs((first + pi) % (2 * pi) - pi),
                np.abs((second + pi) % (2 * pi) - pi),
            )
            curvature = np.abs(mid - (left / 2 + right / 2))
            distance = np.minimum(np.minimum(am, bm), cm)
            split = (angles > pi / 8) | (curvature > 0.25 * distance)
        if not np.any(split):
            return contour, values, True
        if len(contour) + int(split.sum()) > 150_000:
            break
        positions = np.cumsum(np.r_[0, 1 + split])
        next_contour = np.empty(positions[-1] + 1, dtype=np.complex128)
        next_values = np.empty_like(next_contour)
        next_contour[positions] = contour
        next_contour[positions[:-1][split] + 1] = midpoints[split]
        next_values[positions] = values
        next_values[positions[:-1][split] + 1] = mid_values[split]
        contour, values = next_contour, next_values
    return contour, values, False


def nyquist(
    system: TransferFunction,
    *,
    points: int = 192,
    refinements: int = 5,
    axis_tol: float = 1e-10,
    critical_tol: float = 1e-7,
) -> NyquistResult:
    """Analyze unity negative feedback using a clockwise closed contour.

    The exterior disk is bounded analytically. The interior winding uses
    sampling convergence and reports an indeterminate result when it fails.
    """
    from .bode import gain_frequencies

    if points < 16 or refinements < 2 or axis_tol <= 0 or critical_tol <= 0:
        raise ValueError("Resolução ou tolerância de Nyquist inválida.")
    poles = system.poles
    p = int(np.count_nonzero(poles.real > axis_tol * np.maximum(1, np.abs(poles))))
    empty = np.empty(0, dtype=np.complex128)
    radius = _outer_radius(system, poles)
    if radius is None:
        return NyquistResult(
            p,
            None,
            None,
            "indeterminate",
            empty,
            empty,
            0.0,
            None,
            "A cauda ou a realimentação algébrica requer análise específica.",
        )
    if system.delay * radius > 1e4:
        return NyquistResult(
            p,
            None,
            None,
            "indeterminate",
            empty,
            empty,
            radius,
            None,
            "O produto atraso–raio excede o limite da malha de certificação numérica.",
        )

    markers = np.asarray(gain_frequencies(system), dtype=np.float64)
    markers = markers[np.isfinite(markers) & (markers >= 0) & (markers < radius)]
    boundary_values = system.evaluate(1j * np.r_[0, markers])
    contact = np.any(np.abs(boundary_values + 1) <= critical_tol * 0.01)
    axis = _axis_centers(poles, axis_tol)

    hidden_axis = bool(axis.size) and (
        system.gain == 0
        or system.numerator == (0.0,)
        or np.any(
            np.abs(system.zeros[:, None] - 1j * axis) <= axis_tol * np.maximum(1.0, np.abs(axis))
        )
    )
    radii = _indent_radii(axis, poles, radius, axis_tol)
    contour = mapped = empty
    current: Winding | None = None
    previous: Winding | None = None
    previous_resolved = False
    for level in range(refinements):
        contour = _contour(
            system, radius, points * 2**level, axis, radii / 2**level, np.r_[-markers, markers]
        )
        if contact:
            mapped = system.evaluate(contour)
            return NyquistResult(
                p,
                None,
                None,
                "marginal",
                contour,
                mapped,
                radius,
                float(np.min(np.abs(boundary_values[np.isfinite(boundary_values)] + 1))),
                "A resposta cruza −1 no eixo imaginário dentro da tolerância de fronteira.",
                refinements=level + 1,
            )
        contour, mapped, resolved = _refine_mapping(system, contour)
        try:
            current = winding_number(mapped)
        except ValueError:
            previous = None
            continue
        if (
            previous is not None
            and resolved
            and previous_resolved
            and current.turns == previous.turns
            and max(current.max_angle, previous.max_angle) < pi / 3
            and min(current.min_distance, previous.min_distance) > critical_tol
            and current.residual < 1e-7
            and current.turns + p >= 0
        ):
            z = current.turns + p
            status = "unstable" if z else "stable"
            reason = "Contagem consistente após refinamento do contorno."
            if hidden_axis:
                status = "unstable" if z else "marginal"
                reason += " O modelo inclui um modo cancelado no eixo imaginário."
            return NyquistResult(
                p,
                current.turns,
                z,
                status,
                contour,
                mapped,
                radius,
                current.min_distance,
                reason,
                current.max_angle,
                level + 1,
            )
        previous = current
        previous_resolved = resolved
    return NyquistResult(
        p,
        None,
        None,
        "indeterminate",
        contour,
        mapped,
        radius,
        current.min_distance if current else None,
        "O contorno refinado não resolveu a passagem perto do ponto crítico.",
        current.max_angle if current else None,
        refinements,
    )
