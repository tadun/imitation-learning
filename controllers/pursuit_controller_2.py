from controller import Supervisor
from pioneer_controller import PioneerController
from pid_controller import PIDController
from pure_pursuit import PurePursuit
from pf_state_estimator import PFEstimator
import math
import numpy as np

# config
TEACHER_DEF_NAME = "PIONEER_3DX_TEACHER"
VISION_NOISE_STD = 0.10
HORIZON_STEPS = 40
TARGET_DISTANCE = 1.0
LOOKAHEAD_DISTANCE = 1.0
VMAX = 1.2


def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
controller = PioneerController(robot)

teacher_node = robot.getFromDef(TEACHER_DEF_NAME)
student_node = robot.getSelf()

if teacher_node is None:
    print(f"Error: teacher node '{TEACHER_DEF_NAME}' not found")
    exit()

pid_speed = PIDController(Kp=0.5, Ki=0.01, Kd=0.1, target=TARGET_DISTANCE, output_min=0.0, output_max=VMAX)
pp_steer = PurePursuit(L_d=LOOKAHEAD_DISTANCE)
pf_estimator = PFEstimator(N=200)

pos = student_node.getPosition()
orn = student_node.getOrientation()
yaw = math.atan2(orn[3], orn[0])
last_student_pose = np.array([pos[0], pos[1], yaw])

while robot.step(timestep) != -1:
    current_pos = student_node.getPosition()
    current_orn = student_node.getOrientation()
    current_yaw = math.atan2(current_orn[3], current_orn[0])

    student_pose = np.array([current_pos[0], current_pos[1], current_yaw])
    teacher_pos = teacher_node.getPosition()

    dx = student_pose[0] - last_student_pose[0]
    dy = student_pose[1] - last_student_pose[1]
    dd_local = math.hypot(dx, dy)
    dtheta_local = normalize_angle(current_yaw - last_student_pose[2])

    heading = np.array([math.cos(last_student_pose[2]), math.sin(last_student_pose[2])])
    disp = np.array([dx, dy])
    if np.dot(heading, disp) < 0:
        dd_local = -dd_local

    dxg = teacher_pos[0] - current_pos[0]
    dyg = teacher_pos[1] - current_pos[1]
    cos_y = math.cos(current_yaw)
    sin_y = math.sin(current_yaw)
    x_local = dxg * cos_y + dyg * sin_y
    y_local = -dxg * sin_y + dyg * cos_y

    measured_x = x_local + np.random.normal(0.0, VISION_NOISE_STD)
    measured_y = y_local + np.random.normal(0.0, VISION_NOISE_STD)

    est_x, est_y, est_theta, path = pf_estimator.update_state(dd_local, dtheta_local, measured_x, measured_y, horizon_steps=HORIZON_STEPS)

    dist = math.hypot(est_x, est_y)
    v_desired = pid_speed.update(dist)
    w_desired = pp_steer.update(v_desired, path)

    controller.set_robot_velocity(v_desired, w_desired)
    last_student_pose = student_pose