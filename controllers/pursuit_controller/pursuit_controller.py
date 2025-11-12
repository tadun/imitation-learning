from controller import Robot
from pioneer_controller import PioneerController

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Create controller instance and pass in robot object
controller = PioneerController(robot)

# placeholder basic PID logic
v_desired = 0  # m/s
w_desired = -1  # rad/s

while robot.step(timestep) != -1:
    # call pioneer_controller.py method
    controller.set_robot_velocity(v_desired, w_desired)