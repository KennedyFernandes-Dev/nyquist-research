"""Real, proper, continuous SISO loop models."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _coefficients(values: ArrayLike, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} deve ter coeficientes reais.")
    coeffs = np.asarray(raw, dtype=np.float64)
    if coeffs.ndim != 1 or not coeffs.size or not np.all(np.isfinite(coeffs)):
        raise ValueError(f"{name} deve ser um vetor não vazio de coeficientes finitos.")
    trimmed = np.trim_zeros(coeffs, "f")
    return trimmed if trimmed.size else np.zeros(1)


def horner(coefficients: ArrayLike, points: ArrayLike) -> NDArray[np.complex128]:
    """Evaluate descending coefficients by Horner's recurrence."""
    s = np.asarray(points, dtype=np.complex128)
    value = np.zeros(s.shape, dtype=np.complex128)
    for coefficient in np.asarray(coefficients):
        value = value * s + coefficient
    return value


@dataclass(frozen=True, init=False)
class TransferFunction:
    """Loop L(s) = gain * B(s) / A(s) * exp(-delay*s)."""

    numerator: tuple[float, ...]
    denominator: tuple[float, ...]
    gain: float
    delay: float

    def __init__(
        self,
        numerator: ArrayLike,
        denominator: ArrayLike,
        gain: float = 1.0,
        delay: float = 0.0,
    ) -> None:
        b = _coefficients(numerator, "Numerador")
        a = _coefficients(denominator, "Denominador")
        k = float(gain)
        tau = float(delay)
        if a[0] == 0:
            raise ValueError("O denominador deve ser diferente de zero.")
        if b.size > a.size:
            raise ValueError("A função de transferência deve ser própria.")
        if not isfinite(k) or not isfinite(tau) or tau < 0:
            raise ValueError("Informe ganho finito e atraso finito não negativo.")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            b_normal = b / a[0]
            a_normal = a / a[0]
        if (
            not np.all(np.isfinite(b_normal))
            or not np.all(np.isfinite(a_normal))
            or (b[0] != 0 and b_normal[0] == 0)
        ):
            raise ValueError("Coeficientes fora da escala numérica suportada.")
        object.__setattr__(self, "numerator", tuple(map(float, b_normal)))
        object.__setattr__(self, "denominator", tuple(map(float, a_normal)))
        object.__setattr__(self, "gain", k)
        object.__setattr__(self, "delay", tau)

    @property
    def poles(self) -> NDArray[np.complex128]:
        return np.asarray(np.roots(self.denominator), dtype=np.complex128)

    @property
    def zeros(self) -> NDArray[np.complex128]:
        return np.asarray(np.roots(self.numerator), dtype=np.complex128)

    @property
    def order(self) -> int:
        return len(self.denominator) - 1

    def evaluate(self, s: ArrayLike) -> NDArray[np.complex128]:
        """Evaluate the complex loop, preserving NaN at unresolved poles."""
        points = np.asarray(s, dtype=np.complex128)
        if self.gain == 0 or self.numerator == (0.0,):
            return np.zeros(points.shape, dtype=np.complex128)
        flat = points.ravel()
        out = np.empty_like(flat)
        small = np.abs(flat) <= 1
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            direct = flat[small]
            out[small] = horner(self.numerator, direct) / horner(self.denominator, direct)
            inverse = 1 / flat[~small]
            degree = len(self.denominator) - len(self.numerator)
            out[~small] = (
                horner(self.numerator[::-1], inverse)
                / horner(self.denominator[::-1], inverse)
                * inverse**degree
            )
            out *= self.gain
            if self.delay:
                out *= np.exp(-self.delay * flat)
        return out.reshape(points.shape)

    def characteristic(self, gain: float | None = None) -> NDArray[np.float64]:
        """Form A + K B, preserving its leading zeros at a degree drop."""
        if self.delay:
            raise ValueError("O atraso exato tem equação característica quase polinomial.")
        k = self.gain if gain is None else float(gain)
        if not isfinite(k):
            raise ValueError("O ganho deve ser finito.")
        coeffs = np.asarray(self.denominator).copy()
        with np.errstate(over="ignore", invalid="ignore"):
            coeffs[-len(self.numerator) :] += k * np.asarray(self.numerator)
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("Coeficientes característicos fora da escala suportada.")
        return coeffs
