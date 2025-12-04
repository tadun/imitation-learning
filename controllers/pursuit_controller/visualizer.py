import matplotlib.pyplot as plt
import numpy as np

class PFVisualizer:
    def __init__(self):
        #interactive mode
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        self.ax.set_title("Driver POV")
        self.ax.set_xlabel("(Left <--- 0 ---> Right)")
        self.ax.set_ylabel("distance forward (meters)")
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        #set fixed limits
        # y axis: -1m back to 4m forward
        self.ax.set_ylim(-1, 4) 
        # x axis: 2.5m left and right
        self.ax.set_xlim(2.5, -2.5) #flip these numbers because positive y is actually left in this case
        
        #init empty plot objects
        self.particles_plot, = self.ax.plot([], [], 'r.', markersize=2, alpha=0.5, label='particles')
        self.measurement_plot, = self.ax.plot([], [], 'gx', markersize=10, markeredgewidth=2, label='vision est.')
        self.estimate_plot, = self.ax.plot([], [], 'bo', markersize=8, label='PF estimate')
        
        #student always at 0,0 facing up
        self.student_plot, = self.ax.plot([0], [0], 'k^', markersize=12, label='Student (You)') 

        self.ax.legend(loc='upper right')

    def draw(self, particles, est_pose, measured_pos):
        #in odometry, x is forward/back and y is left/right
        #therefore plotted axis need to be swapped to be in the correct frame
        self.particles_plot.set_data(particles[1, :], particles[0, :])
        self.estimate_plot.set_data([est_pose[1]], [est_pose[0]])
        if measured_pos is not None:
            self.measurement_plot.set_data([measured_pos[1]], [measured_pos[0]])
        #refresh
        plt.pause(0.001)