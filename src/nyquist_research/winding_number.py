"""Geometric winding of a closed polygon, positive clockwise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Winding:
    turns: int
    residual: float
    max_angle: float
    min_distance: float


def winding_number(curve: ArrayLike, center: complex = -1 + 0j) -> Winding:
    """Sum signed principal increments and measure distance to polygon edges."""
    points = np.asarray(curve, dtype=np.complex128)
    if points.ndim != 1 or len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("São necessários pelo menos três vértices complexos finitos.")
    scale = float(np.max(np.abs(points - center)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("A curva não pode ser escalada ao redor do ponto crítico.")
    z = (points - center) / scale
    radii = np.abs(z)
    if np.any(radii == 0):
        raise ValueError("Um vértice coincide com o ponto crítico.")
    direction = z / radii
    angles = np.angle(np.conj(direction) * np.roll(direction, -1))
    turns = -float(np.sum(angles)) / (2 * np.pi)
    nearest = round(turns)

    segment = np.roll(z, -1) - z
    length = np.abs(segment)
    unit = np.divide(segment, length, out=np.zeros_like(segment), where=length != 0)
    projection = np.clip(-np.real(np.conj(unit) * z), 0, length)
    distance = scale * float(np.min(np.abs(z + projection * unit)))
    return Winding(nearest, abs(turns - nearest), float(max(abs(angles))), distance)
