class PioneerController:
    def __init__(self, robot):
        self.W = 0.381  # track width in metres
        self.r = 0.0975 # wheel radius in metres

        #get motor devices
        self.left_motor = robot.getDevice('left wheel')
        self.right_motor = robot.getDevice('right wheel')

        #set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        #get true max velocity from the motor
        self.max_motor_vel_rads = self.left_motor.getMaxVelocity()

        #initialize motors to be stopped
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
    def set_robot_velocity(self, v, w):
        #calculate linear wheel speeds (m/s) using IK formulae
        vl = v - (w * self.W / 2.0)
        vr = v + (w * self.W / 2.0)

        #convert linear m/s to angular rad/s
        w_motor_l = vl / self.r
        w_motor_r = vr / self.r

        #safety clamping-make sure command sent to wheel is below limit
        w_motor_l = max(min(w_motor_l, self.max_motor_vel_rads), -self.max_motor_vel_rads)
        w_motor_r = max(min(w_motor_r, self.max_motor_vel_rads), -self.max_motor_vel_rads)

        #set velocity
        self.left_motor.setVelocity(w_motor_l) 
        self.right_motor.setVelocity(w_motor_r)