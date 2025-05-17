class PIDController:
    def __init__(self, Kp=1.6, Ki=0.01, Kd=0.8):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def control(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

