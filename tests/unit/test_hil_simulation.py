import math

import numpy as np
import pytest

from firmware.hil.main import process_line
from firmware.hil.pid import FilteredPID
from firmware.hil.serial_host import parse_response, plant_zoh
from firmware.hil.simulate import Scenario, continuous_loop, simulate
from nyquist_research.time_delay import critical_delay


def test_filtered_pid_updates_held_error_states():
    controller = FilteredPID(kp=2, ki=1, kd=0.1, filter_time=0.04)
    first = controller.update(1, 0.02)
    second = controller.update(1, 0.02)
    assert first == pytest.approx(4.5)
    assert second == pytest.approx(2 + 0.02 + 2.5 * math.exp(-0.5))


def test_microcontroller_protocol_returns_command_with_matching_sequence():
    controller = FilteredPID()
    sequence, frame = process_line(bytearray(b"0,1,0"), controller, 0.02, -1)
    assert sequence == 0 and frame == b"0,4.5\n"
    with pytest.raises(ValueError):
        process_line(bytearray(b"0,1,0"), controller, 0.02, sequence)
    with pytest.raises(ValueError):
        process_line(bytearray(b"1,inf,0"), controller, 0.02, sequence)


def test_delay_queue_and_plant_zoh_are_causal():
    results = simulate(Scenario("check", 0.02, 0.04, 0), duration_s=0.1)
    assert all(row["command_applied"] == 0 for row in results[:2])
    assert results[2]["applied_delay_s"] == pytest.approx(0.04)
    assert results[2]["output"] == pytest.approx(0)
    assert results[3]["output"] == pytest.approx(4.5 * (1 - math.exp(-0.02)))


def test_host_discards_stale_or_incomplete_frames_and_advances_plant():
    assert parse_response(b"7,4.5\n", 7) == pytest.approx(4.5)
    assert parse_response(b"6,4.5\n", 7) is None
    assert parse_response(b"7,4.5", 7) is None
    assert parse_response(b"7,inf\n", 7) is None
    assert plant_zoh(0.25, 4.5, 0.02) == pytest.approx(
        math.exp(-0.02) * 0.25 + (1 - math.exp(-0.02)) * 4.5
    )


def test_jitter_is_seeded_and_can_reorder_arrivals():
    case = Scenario("jitter", 0.02, 0.5, 0.1)
    results = simulate(case, duration_s=3, seed=17)
    assert results == simulate(case, duration_s=3, seed=17)
    assert np.std([row["queued_delay_s"] for row in results]) > 0.025
    assert sum(row["reordered_drops_at_step"] for row in results) > 0


def test_pid_delay_boundary_satisfies_characteristic_equation():
    result = critical_delay(continuous_loop())
    assert result.status == "finite"
    assert result.delay == pytest.approx(1.05905567399771)
    assert result.frequency == pytest.approx(1.787887825679066)
    response = continuous_loop(result.delay).evaluate(1j * result.frequency)
    assert response == pytest.approx(-1 + 0j, abs=1e-12)
