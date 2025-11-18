from controller import Robot
from pioneer_controller import PioneerController
from pid_controller import PIDController
from pure_pursuit import PurePursuit


#FIX COMMENTS ON PUREPURSUIT SCRIPT
#FIND A WAY TO USE TECHER ROBOT OWN ODOMETRY TO GENERATE CURRENT-PATH
#CURRENT-PATH CAN BE A 1 ELEMENT LIST
#USING THAT TEST THE WHOLE SYSTEM.

robot = Robot()
timestep = int(robot.getBasicTimeStep())
controller = PioneerController(robot)

VMAX = 1.2
TARGET_DISTANCE = 1.0  #(placeholder)
LOOKAHEAD_DISTANCE = 1 #(placeholder) TUNE

pid_speed = PIDController(
    Kp=0.5, Ki=0.01, Kd=0.1,  # TUNING NEEDED
    target=TARGET_DISTANCE,
    output_min=0.0,           
    output_max=VMAX           
)

pp_steer = PurePursuit(L_d=LOOKAHEAD_DISTANCE)

#MAINLOOP
while robot.step(timestep) != -1:
    
    #placeholder for particle filter output
    current_distance = 2.5 
    
    #placeholder particle filter output (RANDOM)
    current_path = [(0.1, 0.01), (0.2, 0.02), (0.3, 0.05),
                    (0.4, 0.08), (0.5, 0.1), (0.6, 0.12),
                    (0.7, 0.13), (0.8, 0.14), (0.9, 0.15)]

    #calculate required v
    v_desired = pid_speed.update(current_distance)
    
    #calculate required w
    w_desired = pp_steer.update(v_desired, current_path)
    
    #command robot
    controller.set_robot_velocity(v_desired, w_desired)