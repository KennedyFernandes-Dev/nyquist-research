"""Measure algorithm cost and precision on orders 5, 10 and 20."""

import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections.abc import Callable

import mpmath as mp
import numpy as np

from nyquist_research.bode import bode
from nyquist_research.nyquist import nyquist
from nyquist_research.root_locus import root_locus
from nyquist_research.routh_hurwitz import routh_hurwitz
from nyquist_research.transfer_function import TransferFunction
from validation.references import root_error, symbolic_roots

from .common import CONFIG, complex_pairs, csv_result, json_result


def _designed_polynomial(order: int, lower: str, upper: str) -> tuple[np.ndarray, np.ndarray]:
    """Build coefficients in high precision before rounding once to float64."""
    with mp.workdps(80):
        roots = [-(mp.mpf(lower) + i * (mp.mpf(upper) - mp.mpf(lower)) / (order - 1))
                 for i in range(order)]
        coefficients = [mp.mpf(1)]
        for root in roots:
            coefficients = [*coefficients, mp.mpf(0)]
            for j in range(len(coefficients) - 1, 0, -1):
                coefficients[j] -= root * coefficients[j - 1]
        return np.asarray(coefficients, dtype=float), np.asarray(roots, dtype=float)


def _roots_of_rounded(coefficients: np.ndarray) -> tuple[np.ndarray | None, str]:
    with mp.workdps(60):
        initial = np.roots(coefficients)
        try:
            roots = mp.polyroots([mp.mpf(float(c)) for c in coefficients], maxsteps=450,
                                 extraprec=165, roots_init=list(map(complex, initial)))
            return np.asarray(list(map(complex, roots))), "Mpmath 60 dígitos, +165 bits"
        except mp.libmp.NoConvergence:
            try:
                return symbolic_roots(coefficients), "SymPy nroots, 35 dígitos"
            except (mp.libmp.NoConvergence, ValueError) as error:
                return None, f"Inconclusivo: {type(error).__name__}"


def _precision(order: int, cfg: dict, rng: np.random.Generator) -> list[dict]:
    rows = []
    for family, lower, upper in (("dispersa", "0.4", "3.2"), ("agrupada", "0.8", "1.3")):
        coefficients, designed_roots = _designed_polynomial(order, lower, upper)
        model = TransferFunction([1], coefficients)
        observed = model.poles
        reference, solver = _roots_of_rounded(coefficients)
        perturbed = coefficients * (1 + cfg["coefficient_relative_perturbation"] *
                                    rng.choice([-1, 1], len(coefficients)))
        perturbed[0] = coefficients[0]
        perturbed_roots = TransferFunction([1], perturbed).poles
        shift = root_error(observed, perturbed_roots)
        absolute_error = root_error(observed, designed_roots)
        relative_error = absolute_error / float(np.max(abs(designed_roots)))
        normalized_residual = np.max(abs(np.polyval(coefficients, observed)) /
                                     np.polyval(abs(coefficients), abs(observed)))
        with mp.workdps(80):
            # Evaluate the condition of the known factor roots in high precision.
            m_roots = [-(mp.mpf(lower) +
                         i * (mp.mpf(upper) - mp.mpf(lower)) / (order - 1))
                       for i in range(order)]
            weighted_coefficients = [mp.mpf(float(x)) for x in coefficients]
            condition = [
                mp.polyval(weighted_coefficients, abs(root)) /
                abs(root * mp.fprod(root - other for other in m_roots if root != other))
                for root in m_roots
            ]
        rows.append({
            "order": order, "family": family, "coefficients": coefficients.tolist(),
            "perturbed_coefficients": perturbed.tolist(),
            "designed_roots": designed_roots.tolist(), "observed_roots": complex_pairs(observed),
            "reference_rounded_roots": complex_pairs(reference) if reference is not None else None,
            "perturbed_roots": complex_pairs(perturbed_roots), "reference_solver": solver,
            "root_absolute_error_to_design": absolute_error,
            "relative_error_to_design": relative_error,
            "root_rounding_error_to_design": root_error(reference, designed_roots)
            if reference is not None else None,
            "root_algorithm_error_to_rounded_poly": root_error(observed, reference)
            if reference is not None else None,
            "relative_coefficient_perturbation": cfg["coefficient_relative_perturbation"],
            "max_pole_shift_after_perturbation": shift,
            "max_relative_condition": float(max(condition)),
            "max_normalized_residual": float(normalized_residual),
        })
    return rows


def _environment(cfg: dict) -> dict:
    processor = platform.processor()
    if platform.system() == "Darwin":
        finder = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                check=False, capture_output=True, text=True)
        if finder.returncode == 0:
            processor = finder.stdout.strip()
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "processor": processor,
        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "control", "sympy", "mpmath", "matplotlib")
        },
        "thread_environment": {
            name: os.getenv(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS")
        },
        "configuration": cfg,
    }


def _methods(order: int, cfg: dict) -> tuple[dict[str, Callable], TransferFunction]:
    coefficients, _ = _designed_polynomial(order, "0.4", "3.2")
    model = TransferFunction([cfg["loop_gain_at_dc"] * coefficients[-1]], coefficients)
    closed = model.characteristic()
    omega = np.geomspace(1e-3, 1e2, cfg["bode_frequency_count"])
    gains = np.r_[0, np.geomspace(0.01, 8, cfg["root_locus_gain_count"] - 1)]
    return {
        "Nyquist": lambda: nyquist(model),
        "Routh-Hurwitz": lambda: routh_hurwitz(closed),
        "Lugar das Raízes": lambda: root_locus(model, gains),
        "Bode": lambda: bode(model, omega),
    }, model


def _measurement(method: Callable) -> tuple[float, float]:
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    method()
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.process_time_ns() - cpu_start
    return wall_ns / 1e6, cpu_ns / 1e6


def _check_model(methods: dict, model: TransferFunction) -> dict:
    nyquist_result = methods["Nyquist"]()
    routh_result = methods["Routh-Hurwitz"]()
    bode_result = methods["Bode"]()
    locus_result = methods["Lugar das Raízes"]()
    # On the whole imaginary axis, |L(jw)| <= L(0)=0.4 by factorization.
    return {
        "expected_closed_rhp": 0, "nyquist_z": nyquist_result.z,
        "nyquist_status": nyquist_result.status, "routh_z": routh_result.rhp,
        "routh_status": routh_result.status, "max_open_loop_magnitude_analytic": float(
            abs(model.evaluate(0))),
        "bode_gain_crossings": len(bode_result.gain_crossings),
        "locus_gain_samples": len(locus_result.gains),
        "locus_poles_per_sample": int(locus_result.roots.shape[1]),
    }


def run() -> dict:
    cfg = json.loads((CONFIG / "benchmark.json").read_text())
    if cfg["repetitions"] < 2:
        raise ValueError("O benchmark deve registrar pelo menos duas repetições.")
    rng = np.random.default_rng(cfg["seed"])
    raw = []
    summary = []
    precision = []
    classification = []
    for order in cfg["orders"]:
        methods, model = _methods(order, cfg)
        classification.append({"order": order, **_check_model(methods, model)})
        precision.extend(_precision(order, cfg, rng))
        for name, method in methods.items():
            for _ in range(cfg["warmups"]):
                method()
            samples = [_measurement(method) for _ in range(cfg["repetitions"])]
            tracemalloc.start()
            try:
                method()
                _, memory = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            raw.extend({
                "order": order, "method": name, "repetition": i + 1,
                "wall_ms": wall, "cpu_ms": cpu,
            } for i, (wall, cpu) in enumerate(samples))
            wall_values, cpu_values = zip(*samples, strict=True)
            summary.append({
                "order": order, "method": name, "repetitions": len(samples),
                "mean_wall_ms": statistics.mean(wall_values),
                "stdev_wall_ms": statistics.stdev(wall_values),
                "median_wall_ms": statistics.median(wall_values),
                "mean_cpu_ms": statistics.mean(cpu_values),
                "stdev_cpu_ms": statistics.stdev(cpu_values),
                "median_cpu_ms": statistics.median(cpu_values),
                "min_wall_ms": min(wall_values), "max_wall_ms": max(wall_values),
                "min_cpu_ms": min(cpu_values), "max_cpu_ms": max(cpu_values),
                "peak_tracemalloc_bytes": memory,
            })
    csv_result("raw/benchmark-timings.csv", raw)
    csv_result("tables/benchmark.csv", summary)
    csv_result("tables/precision.csv", [
        {key: value for key, value in row.items() if not isinstance(value, list)}
        for row in precision
    ])
    json_result("raw/precision.json", precision)
    json_result("raw/environment.json", _environment(cfg))
    json_result("processed/benchmark-summary.json", {
        "timings": summary, "classification": classification,
        "precision": [
            {key: value for key, value in row.items() if not isinstance(value, list)}
            for row in precision
        ],
    })
    return {"timings": len(raw), "precision_cases": len(precision),
            "orders": cfg["orders"], "classification": classification}


if __name__ == "__main__":
    print(run())
