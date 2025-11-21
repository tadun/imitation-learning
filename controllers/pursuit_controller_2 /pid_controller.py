import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, target, output_min, output_max):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic()

    def update(self, current_value):
        current_time = time.monotonic()
        dt = current_time - self._last_time
        if dt == 0:
            return self.output_min
        error = current_value - self.target
        p = self.Kp * error
        self._integral += error * dt
        i = self.Ki * self._integral
        d = self.Kd * ((error - self._last_error) / dt)
        output = p + i + d
        self._last_error = error
        self._last_time = current_time
        clamped = max(min(output, self.output_max), self.output_min)
        if output != clamped:
            self._integral -= error * dt
        return clamped
