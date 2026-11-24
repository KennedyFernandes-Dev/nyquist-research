"""PID com derivada filtrada, compartilhado pelo firmware e pela simulação."""

import math

KP = 2.0
KI = 1.0
KD = 0.1
FILTER_TIME_S = 0.04
SAMPLE_TIME_S = 0.02


class FilteredPID:
    def __init__(self, kp=KP, ki=KI, kd=KD, filter_time=FILTER_TIME_S):
        if not all(math.isfinite(value) for value in (kp, ki, kd, filter_time)):
            raise ValueError("Parâmetros do PID devem ser finitos.")
        if filter_time <= 0:
            raise ValueError("O tempo do filtro deve ser positivo.")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.filter_time = filter_time
        self.integral = 0.0
        self.filtered_error = 0.0

    def update(self, error, dt):
        """Emite a ação usando o estado atual e avança o estado sob entrada constante."""
        if not math.isfinite(error) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Erro e passo devem ser finitos, com passo positivo.")
        derivative = self.kd * (error - self.filtered_error) / self.filter_time
        command = self.kp * error + self.integral + derivative
        self.integral += self.ki * dt * error
        self.filtered_error += (error - self.filtered_error) * -math.expm1(-dt / self.filter_time)
        return command
