import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, target, output_min, output_max):
        self.Kp = Kp  #proportional
        self.Ki = Ki  #integral
        self.Kd = Kd  #derivative

        self.target = target  #value to achieve
        
        #output clamping
        self.output_min = output_min
        self.output_max = output_max

        #internal state
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic() #monotonic so that e.g. daylight savings does not affect time steps

    def update(self, current_value):
        current_time = time.monotonic()
        dt = current_time - self._last_time
        
        #avoid division by zero on first run
        if dt == 0:
            return self.output_min #or 0.0, depends on use case
      
        #1-calculate error
        error = current_value - self.target
        
        #2-proportional term
        p_term = self.Kp * error
        
        #3-integral term -- add the error over time
        self._integral += error * dt
        i_term = self.Ki * self._integral
        
        #4-derivative term -- how fast error is changing?
        derivative = (error - self._last_error) / dt
        d_term = self.Kd * derivative

        #combine terms
        output = p_term + i_term + d_term
        
        #update state for next loop
        self._last_error = error
        self._last_time = current_time

        #output clamping
        clamped_output = max(min(output, self.output_max), self.output_min)
        
        #anti-windup for integral
        #if output clamped, stop integral from growing
        if output != clamped_output:
            self._integral -= error * dt

        return clamped_output