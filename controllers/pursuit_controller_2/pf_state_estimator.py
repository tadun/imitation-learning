import numpy as np
import math

class PFEstimator:
    def __init__(self, N=100, process_std=[0.05, 0.05], measurement_std=0.5):
        self._N = int(N)
        self.particles = np.random.normal(scale=1.0, size=(3, self._N))
        self.particles[2, :] = np.random.uniform(-math.pi, math.pi, self._N)
        self.weights = np.ones((self._N,)) / self._N
        self.process_std = np.array(process_std)
        self.measurement_std = measurement_std
        self.est = np.array([0.0, 0.0, 0.0])

    def resample(self):
        indices = np.random.choice(self._N, size=self._N, p=self.weights)
        self.particles = self.particles[:, indices]
        self.weights.fill(1.0 / self._N)

    def predict(self, dd, dtheta):
        dd_noisy = dd + np.random.normal(scale=self.process_std[0], size=self._N)
        dtheta_noisy = dtheta + np.random.normal(scale=self.process_std[1], size=self._N)

        self.particles[0, :] -= dd_noisy

        cos_r = np.cos(-dtheta_noisy)
        sin_r = np.sin(-dtheta_noisy)
        x_prev = self.particles[0, :].copy()
        y_prev = self.particles[1, :].copy()
        self.particles[0, :] = x_prev * cos_r - y_prev * sin_r
        self.particles[1, :] = x_prev * sin_r + y_prev * cos_r

        self.particles[2, :] -= dtheta_noisy
        self.particles[2, :] = np.arctan2(np.sin(self.particles[2, :]), np.cos(self.particles[2, :]))

    def update_weights(self, measured_x, measured_y):
        dx = self.particles[0, :] - measured_x
        dy = self.particles[1, :] - measured_y
        sq = dx**2 + dy**2
        coeff = 1.0 / (self.measurement_std * np.sqrt(2 * np.pi))
        un = coeff * np.exp(-0.5 * sq / (self.measurement_std ** 2)) + 1e-12
        self.weights *= un
        self.weights /= np.sum(self.weights)

    def predict_trajectory(self, horizon_steps, est_state):
        traj = []
        x, y, th = est_state
        step = 0.05
        for _ in range(horizon_steps):
            x += step * math.cos(th)
            y += step * math.sin(th)
            traj.append((x, y))
        return traj

    def update_state(self, dd, dtheta, measured_x, measured_y, horizon_steps=40):
        self.predict(dd, dtheta)
        self.update_weights(measured_x, measured_y)
        self.est = np.average(self.particles, axis=1, weights=self.weights)
        est_x, est_y, est_theta = self.est
        neff = 1.0 / np.sum(self.weights ** 2)
        if neff < (self._N / 2.0):
            self.resample()
        trajectory = self.predict_trajectory(horizon_steps, self.est)
        return est_x, est_y, est_theta, trajectory