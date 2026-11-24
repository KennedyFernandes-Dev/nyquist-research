"""Run the four mandatory plant cases and the delay approximation study."""

from dataclasses import replace
from math import atan, degrees, pi

import control as ct
import numpy as np
from scipy import optimize, signal

from nyquist_research.bode import bode
from nyquist_research.frequency_response import frequency_response
from nyquist_research.nyquist import nyquist
from nyquist_research.pade import approximate_delay, pade
from nyquist_research.root_locus import root_locus
from nyquist_research.routh_hurwitz import routh_hurwitz
from nyquist_research.step_response import step_response
from nyquist_research.time_delay import critical_delay
from nyquist_research.transfer_function import TransferFunction
from validation.references import (
    d_delay_analytic,
    d_lambert_roots,
    d_rhp_analytic,
    frequency_reference,
    pade_reference,
    rational_model,
    root_error,
    symbolic_roots,
)

from .common import complex_pairs, csv_result, json_result, systems

FREQUENCIES = np.geomspace(1e-3, 1e2, 1001)
STEP_TIMES = np.linspace(0, 15, 301)
LOCUS_GAINS = np.r_[0, np.geomspace(0.01, 20, 100)]
PADE_ORDERS = (1, 2, 3, 5)
STATUS_TOL = 1e-8


def classify_roots(poles: np.ndarray) -> tuple[int, str]:
    rhp = int(np.count_nonzero(poles.real > STATUS_TOL))
    axis = bool(np.any(abs(poles.real) <= STATUS_TOL))
    return rhp, "unstable" if rhp else ("marginal" if axis else "stable")


def check(name: str, observed: float, limit: float) -> dict:
    return {
        "check": name,
        "value": float(observed),
        "limit": float(limit),
        "pass": bool(observed <= limit),
    }


def _plant_points(plants: dict[str, TransferFunction]) -> list[tuple]:
    delay, _ = d_delay_analytic()
    extra = {
        "A": (("fronteira", 10, "marginal"), ("instável", 15, "unstable")),
        "B": (("instável", 1, "unstable"), ("fronteira", 2, "marginal")),
        "C": (("fronteira", 3, "marginal"), ("instável", 5, "unstable")),
    }
    points = []
    for label, model in plants.items():
        points.append((label, "nominal", model, "stable"))
        for tag, gain, expected in extra.get(label, ()):
            points.append((label, tag, replace(model, gain=gain), expected))
    points.extend(
        ("D", tag, replace(plants["D"], delay=value), expected)
        for tag, value, expected in (
            ("sem atraso", 0, "stable"), ("fronteira", delay, "marginal"),
            ("instável", 1.6, "unstable"), ("dois pares", 5.0, "unstable"),
        )
    )
    return points


def _stability_points(plants: dict[str, TransferFunction]) -> tuple[list[dict], list[dict]]:
    rows = []
    checks = []
    for label, tag, model, expected in _plant_points(plants):
        initial = nyquist(model)
        extended_case = label == "D" and tag == "dois pares"
        resolution = 1024 if extended_case else None
        winding = nyquist(model, points=resolution) if resolution else initial
        if not model.delay:
            denominator = model.characteristic()
            routh = routh_hurwitz(denominator)
            roots = symbolic_roots(denominator)
            count_ref, root_status = classify_roots(roots)
            reference_source = "SymPy nroots, 35 dígitos"
            reference_winding = None
            if root_status != "marginal":
                reference_winding = int(ct.nyquist_response(
                    rational_model(model), omega_num=6000, warn_encirclements=False,
                    warn_nyquist=False,
                ).count)
                checks.append(check(f"{label}/{tag}: N Python-control",
                                    abs(float(winding.n) - reference_winding)
                                    if winding.n is not None else 1, 0))
            checks.extend([
                check(f"{label}/{tag}: Routh", routh.status != expected, 0),
                check(f"{label}/{tag}: Z Routh", routh.rhp != count_ref, 0),
            ])
        else:
            routh = None
            roots = d_lambert_roots(model.delay)
            count_ref, root_status = d_rhp_analytic(model.delay)
            reference_source = "Cruzamentos analíticos + Lambert W, 65 dígitos"
            reference_winding = None
            checks.append(check(f"{label}/{tag}: Lambert W",
                                classify_roots(roots) != (count_ref, root_status), 0))
        checks.extend([
            check(f"{label}/{tag}: estado de referência", root_status != expected, 0),
            check(f"{label}/{tag}: Nyquist", winding.status != expected, 0),
            check(f"{label}/{tag}: Z Nyquist", winding.z != count_ref, 0)
            if root_status != "marginal" else
            check(f"{label}/{tag}: fronteira Nyquist", winding.status != "marginal", 0),
        ])
        if extended_case:
            checks.append(check(
                "D/dois pares: malha padrão converge após refinamento local",
                initial.status != expected or initial.z != count_ref, 0
            ))
        rows.append({
            "system": label, "point": tag, "gain": model.gain, "delay_s": model.delay,
            "expected": expected, "p": winding.p, "n": winding.n, "z": winding.z,
            "nyquist_status": winding.status, "reference_n": reference_winding,
            "reference_z": count_ref, "reference_status": root_status,
            "reference_method": reference_source,
            "routh_rhp": routh.rhp if routh else None,
            "routh_status": routh.status if routh else "not_applicable",
            "nyquist_radius": winding.radius,
            "nyquist_distance": winding.min_distance,
            "nyquist_reason": winding.reason,
            "nyquist_requested_points": resolution,
            "nyquist_contour_samples": int(winding.contour.size),
            "nyquist_initial_status": initial.status,
            "nyquist_initial_z": initial.z,
            "nyquist_initial_max_angle_rad": initial.max_angle,
            "nyquist_initial_contour_samples": int(initial.contour.size),
            "reference_poles": complex_pairs(roots),
        })
    return rows, checks


def _rational_margins(model: TransferFunction) -> tuple[list, list]:
    gm, pm, _, wpc, wgc, _ = ct.stability_margins(rational_model(model), returnall=True)
    return sorted(zip(wgc, pm, strict=True)), sorted(zip(wpc, gm, strict=True))


def _d_margins(delay: float) -> tuple[list, list]:
    _, wc = d_delay_analytic()
    wpc = optimize.brentq(lambda w: atan(w) + delay * w - pi, 0, pi / delay)
    return [(wc, 120 - degrees(delay * wc))], [(wpc, np.hypot(1, wpc) / 2)]


def _array_margins(results: tuple, references: list, name: str) -> list[dict]:
    actual = np.asarray(sorted(results)).reshape(-1, 2)
    expected = np.asarray(sorted(references)).reshape(-1, 2)
    if actual.shape != expected.shape:
        return [check(name + " cardinalidade", abs(len(actual) - len(expected)), 0)]
    delta = abs(actual - expected)
    return [
        check(name + " frequência", np.max(delta[:, 0], initial=0.0), 2e-6),
        check(name + " margem", np.max(delta[:, 1], initial=0.0), 2e-5),
    ]


def _rational_approximation(model: TransferFunction, order: int) -> ct.TransferFunction:
    rational = rational_model(model)
    if not model.delay:
        return rational
    numerator, denominator = ct.pade(model.delay, order)
    return rational * ct.tf(numerator, denominator)


def _frequency_data(plants: dict[str, TransferFunction]) -> tuple[list[dict], list[dict]]:
    nominal = []
    checks = []
    frequency_rows = []
    winding_rows = []
    locus_rows = []
    step_rows = []
    for label, model in plants.items():
        response = frequency_response(model, FREQUENCIES)
        reference = frequency_reference(model, FREQUENCIES)
        error = abs(response - reference)
        relative = error / np.maximum(abs(reference), np.finfo(float).tiny)
        diag = bode(model, FREQUENCIES)
        winding = nyquist(model)
        step = step_response(model, STEP_TIMES, pade_order=3)
        rational = _rational_approximation(model, 3)
        closed = ct.feedback(rational)
        _, step_ref = signal.step((closed.num[0][0], closed.den[0][0]), T=STEP_TIMES)

        algebraic = approximate_delay(model, 3) if model.delay else model
        locus = root_locus(algebraic, LOCUS_GAINS)
        reference_map = ct.root_locus_map(rational / model.gain, gains=LOCUS_GAINS)
        locus_error = max(
            root_error(left, right)
            for left, right in zip(locus.roots, reference_map.loci, strict=True)
        )
        rational_bare = rational_model(model)
        checks.extend([
            check(f"{label}: resposta complexa SciPy absoluta", error.max(), 1e-10),
            check(f"{label}: resposta complexa SciPy relativa", relative.max(), 1e-10),
            check(f"{label}: polos Python-control", root_error(model.poles, ct.poles(rational_bare)),
                  1e-9),
            check(f"{label}: zeros Python-control", root_error(model.zeros, ct.zeros(rational_bare)),
                  1e-9),
            check(f"{label}: Lugar das Raízes Python-control", locus_error, 2e-6),
            check(f"{label}: degrau SciPy", np.max(abs(step - step_ref)), 2e-6),
        ])
        expected_crossings = {
            "A": (10, np.sqrt(11)),
            "B": (2, 0),
            "C": (3, np.sqrt(5)),
        }
        if label in expected_crossings:
            expected_gain, expected_omega = expected_crossings[label]
            distances = [abs(gain - expected_gain) + abs(abs(pole.imag) - expected_omega)
                         for gain, pole in locus.crossings]
            checks.append(check(
                f"{label}: fronteira analítica do Lugar das Raízes",
                min(distances, default=1.0), 2e-7,
            ))
        if label == "C":
            checks.append(check("C: resposta inicialmente inversa", step[1] >= 0, 0))
        gain_crossings = tuple((float(a), float(b)) for a, b in diag.gain_crossings)
        phase_crossings = tuple((float(a), float(b)) for a, b in diag.phase_crossings)
        reference_gain, reference_phase = (_d_margins(model.delay) if model.delay else
                                            _rational_margins(model))
        # The rational control function omits DC for unit-magnitude crossover.
        comparable_gain = tuple(pair for pair in gain_crossings if pair[0] > 1e-9)
        comparable_phase = tuple(pair for pair in phase_crossings if pair[0] > 1e-9)
        comparable_ref_phase = [pair for pair in reference_phase if pair[0] > 1e-9]
        if model.delay:
            comparable_phase = comparable_phase[:1]
        checks.extend(_array_margins(comparable_gain, reference_gain, f"{label}: MF"))
        checks.extend(_array_margins(comparable_phase, comparable_ref_phase, f"{label}: MG"))
        if label == "A":
            checks.append(check("A: ganho unitário em DC, solução direta", abs(
                float(abs(model.evaluate(0))) - 1), 1e-12))
        if label == "B":
            checks.append(check("B: MG em DC, solução direta", abs(
                1 / float(abs(model.evaluate(0))) - reference_phase[0][1]), 1e-12))
        dc_gain = next((margin for freq, margin in gain_crossings if freq == 0), None)
        routh = routh_hurwitz(algebraic.characteristic())
        nominal.append({
            "system": label, "gain": model.gain, "delay_s": model.delay,
            "numerator": list(model.numerator), "denominator": list(model.denominator),
            "open_poles": complex_pairs(model.poles), "open_zeros": complex_pairs(model.zeros),
            "p": winding.p, "n": winding.n, "z": winding.z, "nyquist_status": winding.status,
            "routh_status": routh.status, "routh_delay_model": "Padé [3/3]" if model.delay else None,
            "root_locus_crossings": [[float(g), float(s.real), float(s.imag)]
                                      for g, s in locus.crossings],
            "gain_crossings_rad_s_pm_deg": gain_crossings,
            "phase_crossings_rad_s_gm": phase_crossings,
            "reference_gain_crossings_rad_s_pm_deg": reference_gain,
            "reference_phase_crossings_rad_s_gm": reference_phase,
            "gain_crossover_at_dc": dc_gain is not None,
            "max_frequency_absolute_error": float(error.max()),
            "max_frequency_relative_error": float(relative.max()),
            "max_locus_error": locus_error,
            "max_step_error": float(np.max(abs(step - step_ref))),
            "step_minimum": float(np.min(step)),
            "step_final_sample": float(step[-1]),
        })
        for row in zip(FREQUENCIES, response, reference, diag.magnitude_db, diag.phase_deg,
                       strict=True):
            w, actual, expected, db, phase = row
            frequency_rows.append({
                "system": label, "omega_rad_s": w, "real": actual.real, "imag": actual.imag,
                "reference_real": expected.real, "reference_imag": expected.imag,
                "magnitude_db": db, "phase_deg": phase,
            })
        for index, (s, mapped) in enumerate(zip(winding.contour, winding.mapped, strict=True)):
            winding_rows.append({
                "system": label, "sample": index, "s_real": s.real, "s_imag": s.imag,
                "mapped_real": mapped.real, "mapped_imag": mapped.imag,
            })
        for index, (gain, roots) in enumerate(zip(locus.gains, locus.roots, strict=True)):
            for branch, pole in enumerate(roots):
                locus_rows.append({
                    "system": label, "sample": index, "branch": branch, "gain": gain,
                    "pole_real": pole.real, "pole_imag": pole.imag,
                })
        for ts, sample, expected in zip(STEP_TIMES, step, step_ref, strict=True):
            step_rows.append({
                "system": label, "time_s": ts, "response": sample, "reference": expected,
                "pade_order": 3 if model.delay else None,
            })
    csv_result("raw/frequency.csv", frequency_rows)
    csv_result("raw/nyquist-contours.csv", winding_rows)
    csv_result("raw/root-locus.csv", locus_rows)
    csv_result("raw/step.csv", step_rows)
    return nominal, checks


def _pade_critical(model: TransferFunction, order: int) -> float:
    """Numerically find the first crossing in a bracket justified for system D."""
    def spectral_abscissa(delay: float) -> float:
        approximation = approximate_delay(replace(model, delay=delay), order)
        return float(np.max(np.roots(approximation.characteristic()).real))

    critical, _ = d_delay_analytic()
    sign_bracket = (0.8 * critical, 2.2)
    return float(optimize.brentq(spectral_abscissa, *sign_bracket, xtol=1e-12))


def _reference_pade_critical(model: TransferFunction, order: int) -> float:
    def spectral_abscissa(delay: float) -> float:
        rational = _rational_approximation(replace(model, delay=delay), order)
        return float(np.max(ct.poles(ct.feedback(rational)).real))

    return float(optimize.brentq(spectral_abscissa, 0.95, 2.2, xtol=1e-12))


def _delay_data(model: TransferFunction) -> tuple[dict, list[dict]]:
    target, frequency = d_delay_analytic()
    estimate = critical_delay(model)
    observed = float(estimate.delay) if estimate.delay is not None else None
    checks = [check("D: atraso crítico, referência analítica",
                    abs(observed - target) if observed is not None else 1, 1e-8)]
    response_rows = []
    convergence_rows = []
    summary = []
    omega = np.geomspace(0.01, 30, 801)
    model_response = frequency_reference(model, omega)
    delay_sweep = np.linspace(0.5, 2.1, 65)
    for order in PADE_ORDERS:
        numerator, denominator = pade(model.delay, order)
        reference_num, reference_den = pade_reference(model.delay, order)
        # Cross multiplication avoids conventions for a monic or unit-constant denominator.
        product = np.polysub(np.polymul(numerator, reference_den),
                             np.polymul(denominator, reference_num))
        coefficient_error = float(np.max(abs(product), initial=0.0)) / float(abs(denominator[-1]))
        approximation = approximate_delay(model, order)
        approx_response = frequency_response(approximation, omega)
        complex_error = abs(approx_response - model_response)
        phase_error = np.rad2deg(np.unwrap(np.angle(approx_response / model_response)))
        threshold = _pade_critical(model, order)
        threshold_ref = _reference_pade_critical(model, order)
        checks.extend([
            check(f"D: coeficientes Padé [{order}/{order}] Mpmath", coefficient_error, 1e-10),
            check(f"D: atraso crítico Padé [{order}/{order}] Python-control",
                  abs(threshold - threshold_ref), 1e-8),
        ])
        sample_delay = 1.01 * target
        boundary_approx = approximate_delay(replace(model, delay=sample_delay), order)
        near_boundary = routh_hurwitz(boundary_approx.characteristic())
        band = omega <= 5
        summary.append({
            "order": order, "critical_delay_s": threshold,
            "critical_reference_control_s": threshold_ref,
            "critical_error_s": abs(threshold - target),
            "critical_relative_error": abs(threshold - target) / target,
            "max_inband_complex_error": float(np.max(complex_error[band])),
            "max_inband_phase_error_deg": float(np.max(abs(phase_error[band]))),
            "status_at_1p01_exact": near_boundary.status,
        })
        for w, err, dp in zip(omega, complex_error, phase_error, strict=True):
            response_rows.append({
                "order": order, "omega_rad_s": w, "absolute_complex_error": err,
                "phase_error_deg": dp, "approximation_delay_s": model.delay,
            })
        for tau in delay_sweep:
            rational = approximate_delay(replace(model, delay=float(tau)), order)
            rhp_rational = routh_hurwitz(rational.characteristic()).rhp
            rhp_exact, _ = d_rhp_analytic(float(tau))
            convergence_rows.append({
                "order": order, "delay_s": tau, "rhp_exact": rhp_exact, "rhp_pade": rhp_rational,
            })
    csv_result("raw/pade-frequency-error.csv", response_rows)
    csv_result("raw/pade-delay-sweep.csv", convergence_rows)
    csv_result("tables/pade.csv", summary)
    return {
        "critical_exact_analytic_s": target, "critical_exact_computed_s": observed,
        "critical_frequency_rad_s": frequency, "critical_estimator_status": estimate.status,
        "inband_upper_rad_s": 5, "model_delay_s": model.delay,
        "orders": summary,
    }, checks


def run() -> dict:
    plants = systems()
    points, point_checks = _stability_points(plants)
    nominal, nominal_checks = _frequency_data(plants)
    delay, delay_checks = _delay_data(plants["D"])
    csv_result("tables/stability.csv", [
        {key: value for key, value in row.items() if key != "reference_poles"} for row in points
    ])
    csv_result("tables/plants.csv", [
        {key: row[key] for key in ("system", "gain", "delay_s", "p", "n", "z", "nyquist_status",
                                    "routh_status", "max_frequency_relative_error",
                                    "step_minimum", "step_final_sample")}
        for row in nominal
    ])
    checks = point_checks + nominal_checks + delay_checks
    failures = [row for row in checks if not row["pass"]]
    json_result("raw/stability-points.json", points)
    json_result("processed/plant-summary.json", nominal)
    json_result("processed/pade-summary.json", delay)
    json_result("processed/validation.json", {
        "total": len(checks), "passed": len(checks) - len(failures),
        "failures": failures, "checks": checks,
        "conventions": [
            "Routh e Lugar das Raízes usam Padé [3/3] na planta D nominal.",
            "Margens usuais do Python-control omitem o cruzamento unitário em DC.",
            "Para D, confronta-se a primeira margem de ganho positiva em frequência.",
            "A classificação na fronteira admite Z indisponível quando N não está definido.",
            "O caso D a 5 s confronta as malhas padrão e de 1024 pontos de base.",
        ],
    })
    return {"checks": len(checks), "failures": failures, "stability_points": len(points)}


if __name__ == "__main__":
    print(run())
