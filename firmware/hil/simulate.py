"""Executa a simulação de bancada com piloto discreto e latência de comandos."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from nyquist_research.nyquist import nyquist
from nyquist_research.time_delay import critical_delay
from nyquist_research.transfer_function import TransferFunction

from .pid import FILTER_TIME_S, KD, KI, KP, SAMPLE_TIME_S, FilteredPID

SEED = 20261124
DURATION_S = 90.0


@dataclass(frozen=True)
class Scenario:
    name: str
    sample_time_s: float
    nominal_delay_s: float
    jitter_half_width_s: float


def continuous_loop(delay=0.0):
    """Forma C(s)P(s), sendo P(s)=1/(s+1) e D filtrada em C(s)."""
    return TransferFunction(
        [KP * FILTER_TIME_S + KD, KP + KI * FILTER_TIME_S, KI],
        [FILTER_TIME_S, FILTER_TIME_S + 1, 1, 0],
        delay=delay,
    )


def simulate(scenario, duration_s=DURATION_S, seed=SEED):
    step = scenario.sample_time_s
    if not all(
        math.isfinite(v)
        for v in (step, scenario.nominal_delay_s, scenario.jitter_half_width_s, duration_s)
    ):
        raise ValueError("Parâmetros da simulação devem ser finitos.")
    if (
        step <= 0
        or scenario.nominal_delay_s < 0
        or scenario.jitter_half_width_s < 0
        or duration_s < step
    ):
        raise ValueError("Passo, duração, atraso ou jitter fora da faixa física.")
    sample_count = math.floor(duration_s / step)
    times = np.arange(sample_count + 1) * step
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-scenario.jitter_half_width_s, scenario.jitter_half_width_s, len(times))
    requested = np.maximum(0.0, scenario.nominal_delay_s + jitter)
    rounding = np.ceil(requested / step - 1e-10).astype(int)
    plant_decay = math.exp(-step)
    plant_gain = -math.expm1(-step)
    pid = FilteredPID()
    messages = []
    state, applied = 0.0, 0.0
    last_sequence = -1
    records = []
    for sequence, time in enumerate(times):
        error = 1 - state
        emitted = pid.update(error, step)
        heapq.heappush(messages, (sequence + rounding[sequence], sequence, emitted))
        delivered = 0
        dropped = 0
        applied_delay = None
        while messages and messages[0][0] <= sequence:
            _, source, command = heapq.heappop(messages)
            if source <= last_sequence:
                dropped += 1
                continue
            applied = command
            last_sequence = source
            delivered += 1
            applied_delay = (sequence - source) * step
        records.append(
            {
                "scenario": scenario.name,
                "seq": sequence,
                "time_s": float(time),
                "output": state,
                "error": error,
                "command_emitted": emitted,
                "command_applied": applied,
                "requested_delay_s": float(requested[sequence]),
                "queued_delay_s": float(rounding[sequence] * step),
                "applied_delay_s": applied_delay,
                "delivered_at_step": delivered,
                "reordered_drops_at_step": dropped,
            }
        )
        state = plant_decay * state + plant_gain * applied
    return records


def summarize(scenario, records, runtime_s):
    step = scenario.sample_time_s
    errors = np.array([row["error"] for row in records])
    requested = np.array([row["requested_delay_s"] for row in records])
    queued = np.array([row["queued_delay_s"] for row in records])
    applied = np.array(
        [row["applied_delay_s"] for row in records if row["applied_delay_s"] is not None]
    )
    window = max(1, min(round(10 / step), len(errors) // 2))
    earlier_errors, final_errors = errors[-2 * window : -window], errors[-window:]
    earlier_rms, final_rms = (float(np.sqrt(np.mean(x**2))) for x in (earlier_errors, final_errors))
    if final_rms < 1e-7:
        observed = "settled"
    elif final_rms < 0.8 * earlier_rms:
        observed = "decaying"
    elif final_rms > 1.2 * earlier_rms:
        observed = "growing"
    else:
        observed = "persistent"
    comparison = nyquist(continuous_loop(scenario.nominal_delay_s))
    return {
        **asdict(scenario),
        "sample_frequency_hz": 1 / step,
        "simulation_wall_time_s": runtime_s,
        "samples": len(errors),
        "constant_nominal_nyquist": comparison.status,
        "constant_nominal_nyquist_rhp": comparison.z,
        "requested_delay_mean_s": float(np.mean(requested)),
        "queued_delay_mean_s": float(np.mean(queued)),
        "queued_delay_std_s": float(np.std(queued, ddof=1)) if np.ptp(queued) else 0.0,
        "queued_delay_range_s": [float(np.min(queued)), float(np.max(queued))],
        "applied_delay_mean_s": float(np.mean(applied)) if applied.size else None,
        "reordered_drops": sum(row["reordered_drops_at_step"] for row in records),
        "max_abs_error": float(np.max(np.abs(errors))),
        "earlier_rms_error": earlier_rms,
        "final_rms_error": final_rms,
        "observed": observed,
    }


def write_figure(records_by_case, output):
    labels = {
        "constant_below": "Constante abaixo",
        "constant_above": "Constante acima",
        "jitter_below": "Variável abaixo",
        "jitter_above": "Variável acima",
    }
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 4.1), sharex=True, layout="constrained")
    for name, records in records_by_case.items():
        if name not in labels:
            continue
        t = np.array([row["time_s"] for row in records])
        err = np.abs([row["error"] for row in records])
        dt = t[1] - t[0]
        width = round(3 / dt)
        window_t = t[: len(t) // width * width].reshape(-1, width)
        envelope = err[: window_t.size].reshape(window_t.shape)
        axis = axes[0] if name.startswith("constant") else axes[1]
        axis.semilogy(
            np.mean(window_t, axis=1),
            np.maximum(np.max(envelope, axis=1), 1e-14),
            label=labels[name],
        )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="upper left")
        axis.set_ylabel("Pico de |erro| em 3 s")
    axes[1].set_xlabel("Tempo simulado (s)")
    fig.savefig(output, metadata={"Title": "PID em bancada simulada com atraso"})
    plt.close(fig)


def run(root):
    prediction = critical_delay(continuous_loop())
    if prediction.delay is None:
        raise RuntimeError("A previsão analítica do cenário PID ficou indeterminada.")
    boundary = prediction.delay
    cases = [
        Scenario("constant_below", SAMPLE_TIME_S, 0.75 * boundary, 0),
        Scenario("constant_above", SAMPLE_TIME_S, 1.23 * boundary, 0),
        Scenario("jitter_below", SAMPLE_TIME_S, 0.75 * boundary, 0.10),
        Scenario("jitter_above", SAMPLE_TIME_S, 1.23 * boundary, 0.10),
        Scenario("fine_below", SAMPLE_TIME_S / 2, 0.75 * boundary, 0),
        Scenario("fine_above", SAMPLE_TIME_S / 2, 1.23 * boundary, 0),
    ]
    raw, processed, figures = (root / directory for directory in ("raw", "processed", "figures"))
    for directory in (raw, processed, figures):
        directory.mkdir(parents=True, exist_ok=True)
    records_by_case, summary = {}, []
    with (raw / "hil-timeseries.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = None
        for case in cases:
            start = perf_counter()
            records = simulate(case)
            elapsed = perf_counter() - start
            results = summarize(case, records, elapsed)
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(records[0]), lineterminator="\n")
                writer.writeheader()
            writer.writerows(records)
            records_by_case[case.name] = records
            summary.append(results)
    write_figure(records_by_case, figures / "hil-pid.pdf")
    result = {
        "mode": "software_simulation",
        "hardware_connected": False,
        "seed": SEED,
        "simulation_duration_s": DURATION_S,
        "design": {"kp": KP, "ki": KI, "kd": KD, "filter_time_s": FILTER_TIME_S},
        "continuous_delay_prediction": asdict(prediction),
        "cases": summary,
    }
    (processed / "hil-summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
