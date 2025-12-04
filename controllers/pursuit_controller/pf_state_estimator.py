import numpy as np
import math

class PFEstimator:
    #estimates x, y, theta of teacher using particle filter
    #student's own odomety is used for prediction
    def __init__(self, N=100, process_std=[0.05, 0.05], measurement_std=0.5):
        self._N = int(N)
        #state vector (x, y, theta) - Init with noise centred around 0, 0
        self.particles = np.random.normal(scale=1.0, size=(3, self._N))
        
        #init orientation with random vals in -pi -> pi range
        self.particles[2, :] = np.random.uniform(-math.pi, math.pi, self._N) 
        
        #init all weights equally
        self.weights = np.ones((self._N,)) / self._N
        
        #standard deviation for motion noise (linear and angular)
        self.process_std = np.array(process_std) 
        
        #standard deviation for vision measurement noise
        self.measurement_std = measurement_std 
        
        #initial state estimate
        self.est = np.array([0.0, 0.0, 0.0])

    def resample(self):
        #resample particles based on weights, higher weights more likely to be chosen
        indices = np.random.choice(self._N, size=self._N, p=self.weights)
        self.particles = self.particles[:, indices]
        
        #reset weights uniformly after resampling
        self.weights.fill(1.0 / self._N)

    #KINEMATIC MODEL -prediction step
    def predict(self, dd, dtheta):
        #update particles based on the students motion
        #apply inverse transformation to particles because
        #particles must move relative to the motion of the student

        #add noise to the student's odometry to simulate uncertainty
        dd_noisy = dd + np.random.normal(scale=self.process_std[0], size=self._N)
        dtheta_noisy = dtheta + np.random.normal(scale=self.process_std[1], size=self._N)

        #translation - student moves forward, particles move backward relative to student
        #assume student moves along its own x axis
        self.particles[0, :] -= dd_noisy

        #rotation: if student rotates by +dtheta, coordinate frame rotates by +dtheta
        #therefore any given particle rotates by -dtheta
        cos_rot = np.cos(-dtheta_noisy)
        sin_rot = np.sin(-dtheta_noisy)
        
        #store previous positions to ensure accurate rotation
        x_prev = self.particles[0, :].copy()
        y_prev = self.particles[1, :].copy()
        
        #apply rotation matrix        
        self.particles[0, :] = x_prev * cos_rot - y_prev * sin_rot
        self.particles[1, :] = x_prev * sin_rot + y_prev * cos_rot
        
        #update heading of particles
        self.particles[2, :] -= dtheta_noisy
        
        #normalise angles to -pi to pi range
        self.particles[2, :] = np.arctan2(np.sin(self.particles[2, :]), np.cos(self.particles[2, :]))
        
    def update_weights(self, measured_x, measured_y):
        #update importance weight of each particle based on vision measurement
        
        #calculate diff between each particle and measured position
        diffs_x = self.particles[0, :] - measured_x
        diffs_y = self.particles[1, :] - measured_y
        
        #calculate squared euclidean dist
        sq = diffs_x**2 + diffs_y**2

        #calculate gaussian likelihood
        coeff = 1.0 / (self.measurement_std * np.sqrt(2 * np.pi))
        
        #update weights using gaussian funct
        #add 1e-12 to avoid division by 0 or 0-weights
        un = coeff * np.exp(-0.5 * sq / (self.measurement_std ** 2)) + 1e-12 
        self.weights *= un
        #normalise weights such that they sum to 1
        self.weights /= np.sum(self.weights)

    def predict_trajectory(self, horizon_steps, est_state):
        #generate short path forward from current estimate
        #used by controller to steer student
        trajectory = []
        current_x, current_y, current_theta = est_state 
        
        #distance to project each step forward
        projected_dd = 0.05
        
        for _ in range(horizon_steps):
            current_x += projected_dd * math.cos(current_theta)
            current_y += projected_dd * math.sin(current_theta)
            trajectory.append((current_x, current_y))
            
        return trajectory

   
    def update_state(self, dd, dtheta, measured_x, measured_y, horizon_steps=40):
        #main update loop for estimator
        #prediction step
        self.predict(dd, dtheta)

        #correction step (only run if teacher detected)
        if measured_x is not None and measured_y is not None:
            self.update_weights(measured_x, measured_y)
            
            #resample only if weights updated
            neff = 1.0 / np.sum(self.weights ** 2)
            if neff < (self._N / 2.0):
                self.resample()

        #state estimation step - weighted ave of all particles
        self.est = np.average(self.particles, axis=1, weights=self.weights)
        est_x, est_y, est_theta = self.est

        #trajectory generation
        trajectory = self.predict_trajectory(horizon_steps, self.est)
        
        return est_x, est_y, est_theta, trajectory