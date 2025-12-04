from controller import Supervisor
from pioneer_controller import PioneerController
from pid_controller import PIDController
from pure_pursuit import calculate_steering
from pf_state_estimator import PFEstimator
from visualizer import PFVisualizer
from perception.api import PerceptionAPI
import math
import numpy as np

#constants
TEACHER_DEF_NAME = "PIONEER_3DX_TEACHER" 
HORIZON_STEPS = 40       
TARGET_DISTANCE = 0.3    
LOOKAHEAD_DISTANCE = 1.0 
VMAX = 1.2
PLOT_INTERVAL = 10 

#normalise angle to range (-pi,pi)
def normalise_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

#init robot + perception
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
controller = PioneerController(robot)
perception_stack = PerceptionAPI(cfg_path="perception/marker_config.yaml")

#get robot nodes
teacher_node = robot.getFromDef(TEACHER_DEF_NAME)
student_node = robot.getSelf() 

if teacher_node is None:
    print(f"error: could not find teacher robot node '{TEACHER_DEF_NAME}'.")
    exit()

#setup pid and purepursuit controllers
pid_speed = PIDController(
    Kp=0.5, Ki=0.01, Kd=0.1,
    target=TARGET_DISTANCE,
    output_min=-VMAX,           
    output_max=VMAX           
)

pf_estimator = PFEstimator(N=200)
viz = PFVisualizer()

#get starting pos to track student movement
initial_pos = student_node.getPosition()
initial_orn = student_node.getOrientation()
initial_yaw = math.atan2(initial_orn[3], initial_orn[0])
last_student_pose = np.array([initial_pos[0], initial_pos[1], initial_yaw])

#data needed for student recovery
last_valid_bearing = 0.0 
last_cmd_v = 0.0 #needed for smooth braking
step_counter = 0 #needed for graph plotting

#MAINLOOOP
while robot.step(timestep) != -1:
    step_counter += 1
    
    #treu current pos used for particle filter prediction
    current_pos_global = student_node.getPosition()
    current_orn_global = student_node.getOrientation()
    current_yaw_global = math.atan2(current_orn_global[3], current_orn_global[0])
    
    current_student_pose = np.array([current_pos_global[0], current_pos_global[1], current_yaw_global])
    
    #how much student moved since last step?
    dx_diff = current_student_pose[0] - last_student_pose[0]
    dy_diff = current_student_pose[1] - last_student_pose[1]
    
    dd_local = math.sqrt(dx_diff**2 + dy_diff**2)
    dtheta_local = normalise_angle(current_yaw_global - last_student_pose[2])
    
    #make sure not going backwards
    heading_vec = np.array([math.cos(last_student_pose[2]), math.sin(last_student_pose[2])])
    disp_vec = np.array([dx_diff, dy_diff])
    if np.dot(heading_vec, disp_vec) < 0:
        dd_local = -dd_local

    #vision step setup
    measured_x = None
    measured_y = None
    obs = None
    
    #get camera img
    raw_img = controller.camera.getImage()
    
    if raw_img:
        np_img = np.frombuffer(raw_img, np.uint8).reshape((controller.camera.getHeight(), controller.camera.getWidth(), 4))
        bgr_img = np_img[:, :, :3] 
        #pass img to perception stack to find teeacher
        sim_time = robot.getTime()
        obs = perception_stack.process(sim_time, bgr_img)
        
        #if teacher marker found then convert to x, y coords (standard coord transformation)
        if obs.visible and obs.range_m is not None:
            b_rad = -obs.bearing_rad
            r_m = obs.range_m
            
            #store where teacher last seen for recovery
            last_valid_bearing = b_rad 

            measured_x = r_m * math.cos(b_rad)
            measured_y = r_m * math.sin(b_rad)

    #feed all current data into PF
    est_x, est_y, est_theta, current_path = pf_estimator.update_state(
        dd_local, dtheta_local, measured_x, measured_y, horizon_steps=HORIZON_STEPS
    )

    #PF GRAPH PLOT UPDATING (NOT EVERY TIME STEP)
    if step_counter % PLOT_INTERVAL == 0:
        plot_mx = measured_x if measured_x is not None else 0
        plot_my = measured_y if measured_y is not None else 0
        
        viz.draw(
            pf_estimator.particles, 
            [est_x, est_y, est_theta], 
            [plot_mx, plot_my]
        )

    #actual control logic
    #init parameters
    v_desired = 0.0
    w_desired = 0.0

    #is teacher marker visible right now
    is_visible = (obs is not None) and obs.visible and (obs.range_m is not None)

    if is_visible:
        #normal teacher follow mode - it is visible
        current_distance = math.sqrt(est_x**2 + est_y**2)
        v_desired = pid_speed.update(current_distance)
        w_desired = calculate_steering(abs(v_desired), current_path, LOOKAHEAD_DISTANCE)
    else:
        #recovery mode - time to sweep until found again
        v_desired = last_cmd_v * 0.9 #multiply previous speed by 0.9 to slow smoothly
        if v_desired < 0.05: v_desired = 0.0
        
        #sweep step, only sweep when slow enough
        if v_desired < 0.1:
            rotation_speed = 0.8 #high value to find marker again asap
            if last_valid_bearing >= 0:#pivot towards where marker last seen
                w_desired = rotation_speed #spin right
            else:
                w_desired = -rotation_speed #spin left
        else:
            #moving too fast to spin -continu slowing down
            w_desired = 0.0

    #send final speeds to motors and save pose for next step
    last_cmd_v = v_desired
    controller.set_robot_velocity(v_desired, w_desired)
    last_student_pose = current_student_pose