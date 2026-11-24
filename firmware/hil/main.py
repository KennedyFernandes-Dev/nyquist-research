"""Firmware ESP32/MicroPython para o ensaio de PID no UART."""

import math


def process_line(line, controller, sample_time, last_sequence):
    """Protocolo ASCII: número de sequência, referência, medição."""
    if len(line) > 96:
        raise ValueError("Quadro longo demais.")
    sequence_text, reference_text, measured_text = line.decode().strip().split(",")
    sequence = int(sequence_text)
    reference, measured = float(reference_text), float(measured_text)
    if sequence <= last_sequence or not math.isfinite(reference) or not math.isfinite(measured):
        raise ValueError("Quadro fora de sequência ou não finito.")
    elapsed_steps = sequence - last_sequence if last_sequence >= 0 else 1
    command = controller.update(reference - measured, sample_time * elapsed_steps)
    return sequence, f"{sequence:d},{command:.10g}\n".encode()


def run():
    from time import sleep_ms

    from machine import UART
    from pid import SAMPLE_TIME_S, FilteredPID

    uart = UART(2, 115200, tx=17, rx=16, timeout=0, rxbuf=256, txbuf=256)
    controller = FilteredPID()
    last_sequence = -1
    line = bytearray()
    oversize = False
    while True:
        new_data = uart.read(64)
        if not new_data:
            sleep_ms(1)
            continue
        for value in new_data:
            if value == 10:
                try:
                    if oversize:
                        raise ValueError("Quadro longo demais.")
                    last_sequence, response = process_line(
                        line, controller, SAMPLE_TIME_S, last_sequence
                    )
                except ValueError:
                    response = b"ERR,FRAME\n"
                uart.write(response)
                line = bytearray()
                oversize = False
            elif len(line) < 96:
                line.append(value)
            else:
                oversize = True


if __name__ == "__main__":
    run()
