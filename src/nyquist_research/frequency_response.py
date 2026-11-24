"""Direct evaluation of the complex frequency response."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transfer_function import TransferFunction


def frequency_response(system: TransferFunction, omega: ArrayLike) -> NDArray[np.complex128]:
    """Evaluate L(j*omega) for real omega expressed in rad/s."""
    if np.iscomplexobj(omega):
        raise ValueError("As frequências devem ser reais.")
    frequencies = np.asarray(omega, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("As frequências devem ser finitas.")
    return system.evaluate(1j * frequencies)


def frequency_grid(system: TransferFunction, points: int = 1200) -> NDArray[np.float64]:
    """Return a positive logarithmic grid spanning useful model scales."""
    if points < 2:
        raise ValueError("A malha deve ter pelo menos dois pontos.")
    scales = np.abs(np.r_[system.poles, system.zeros])
    if system.delay:
        scales = np.r_[scales, 1 / system.delay]
    values = scales[(scales > 0) & np.isfinite(scales)]
    low = float(np.min(values)) if values.size else 1.0
    high = float(np.max(values)) if values.size else 1.0
    low = max(low, np.finfo(float).tiny * 100)
    high = min(high, np.finfo(float).max / 100)
    return np.geomspace(low / 100, high * 100, points)
