"""Ensaio HIL opcional. Usa um PID externo via UART e uma planta virtual no computador."""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from pathlib import Path
from random import Random
from time import perf_counter, sleep

from .pid import SAMPLE_TIME_S

FIELDS = (
    "seq",
    "scheduled_time_s",
    "sample_time_s",
    "sample_period_s",
    "output",
    "command_received",
    "command_applied",
    "roundtrip_s",
    "injected_delay_s",
    "delivered_at_sample",
    "reordered_drops_at_sample",
)


def parse_response(frame, expected_sequence):
    """Da resposta do ESP32, aceita somente um comando completo da amostra pedida."""
    if not frame.endswith(b"\n"):
        return None
    try:
        sequence_text, command_text = frame.decode("ascii").strip().split(",")
        sequence = int(sequence_text)
        command = float(command_text)
    except (UnicodeError, ValueError):
        return None
    return command if sequence == expected_sequence and math.isfinite(command) else None


def plant_zoh(output, command, duration):
    decay = math.exp(-duration)
    return decay * output - math.expm1(-duration) * command


def run_bench(port, log, delay, jitter, duration, seed):
    """Executa a aquisição em tempo de parede e escreve evidências CSV da bancada."""
    # A latência experimental deve permanecer reproduzível.
    rng = Random(seed)  # nosec B311
    origin = perf_counter()
    previous_sample = origin
    last_sequence = -1
    output, applied = 0.0, 0.0
    messages = []
    log.writeheader()
    for sequence in range(math.floor(duration / SAMPLE_TIME_S) + 1):
        scheduled = origin + sequence * SAMPLE_TIME_S
        remaining = scheduled - perf_counter()
        if remaining > 0:
            sleep(remaining)
        sample_at = perf_counter()
        state_at = previous_sample
        delivered = 0
        dropped = 0
        while messages and messages[0][0] <= sample_at:
            due, source, candidate = heapq.heappop(messages)
            if source <= last_sequence:
                dropped += 1
                continue
            output = plant_zoh(output, applied, due - state_at)
            applied, state_at, last_sequence = candidate, due, source
            delivered += 1
        output = plant_zoh(output, applied, sample_at - state_at)

        started = perf_counter()
        port.write(f"{sequence},1,{output:.10g}\n".encode("ascii"))
        deadline = started + 0.75 * SAMPLE_TIME_S
        command, roundtrip, injected = None, None, None
        while perf_counter() < deadline:
            port.timeout = max(0.0, deadline - perf_counter())
            frame = port.read_until(size=97)
            received = perf_counter()
            command = parse_response(frame, sequence)
            if command is not None:
                roundtrip = received - started
                injected = max(0.0, delay + rng.uniform(-jitter, jitter))
                heapq.heappush(messages, (received + injected, sequence, command))
                break
        log.writerow(
            {
                "seq": sequence,
                "scheduled_time_s": scheduled - origin,
                "sample_time_s": sample_at - origin,
                "sample_period_s": sample_at - previous_sample if sequence else None,
                "output": output,
                "command_received": command,
                "command_applied": applied,
                "roundtrip_s": roundtrip,
                "injected_delay_s": injected,
                "delivered_at_sample": delivered,
                "reordered_drops_at_sample": dropped,
            }
        )
        previous_sample = sample_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if (
        not all(math.isfinite(value) for value in (args.delay, args.jitter, args.duration))
        or min(args.delay, args.jitter) < 0
        or args.duration <= 0
    ):
        parser.error("Use atraso e jitter finitos não negativos e duração finita positiva.")
    try:
        from serial import Serial
    except ImportError:
        parser.error("Instale pyserial==3.5 no ambiente opcional de bancada.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Serial(args.port, 115200, timeout=0, write_timeout=SAMPLE_TIME_S / 2) as port:
        port.reset_input_buffer()
        with args.output.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            run_bench(port, writer, args.delay, args.jitter, args.duration, args.seed)


if __name__ == "__main__":
    main()
