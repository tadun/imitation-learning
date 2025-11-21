class PioneerController:
    def __init__(self, robot):
        self.W = 0.381
        self.r = 0.0975
        self.left_motor = robot.getDevice('left wheel')
        self.right_motor = robot.getDevice('right wheel')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.max_motor_vel_rads = self.left_motor.getMaxVelocity()
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def set_robot_velocity(self, v, w):
        vl = v - (w * self.W / 2.0)
        vr = v + (w * self.W / 2.0)
        w_motor_l = vl / self.r
        w_motor_r = vr / self.r
        w_motor_l = max(min(w_motor_l, self.max_motor_vel_rads), -self.max_motor_vel_rads)
        w_motor_r = max(min(w_motor_r, self.max_motor_vel_rads), -self.max_motor_vel_rads)
        self.left_motor.setVelocity(w_motor_l)
        self.right_motor.setVelocity(w_motor_r)
