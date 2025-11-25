# perception/particle_filter.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class PFState:
    x: float
    y: float
    theta: float  # kept for extensibility; not used in tonight's bearing-only update

def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi

class VisionPF:
    """
    Minimal vision-only particle filter over relative pose (x,y,theta).
    Tonight: bearing-only likelihood (range optional later).
    """
    def __init__(self, N=500, box=((-1, 1), (-1, 1), (-np.pi, np.pi)),
                 sigma_proc_xy=0.03, sigma_proc_theta=0.05,
                 sigma_bearing=0.10):
        self.N = int(N)
        self.box = box
        self.sxy = float(sigma_proc_xy)
        self.sth = float(sigma_proc_theta)
        self.sb  = float(sigma_bearing)
        self.x = np.zeros((self.N, 3), np.float32)  # [x, y, theta]
        self.w = np.ones(self.N, np.float32) / self.N
        self.init_particles()

    def init_particles(self):
        (xl, xh), (yl, yh), (tl, th) = self.box
        self.x[:, 0] = np.random.uniform(xl, xh, self.N)
        self.x[:, 1] = np.random.uniform(yl, yh, self.N)
        self.x[:, 2] = np.random.uniform(tl, th, self.N)
        self.w.fill(1.0 / self.N)

    def predict(self, dt: float):
        # Constant "nearly static" with process noise (we don't use robot kinematics tonight)
        self.x[:, 0] += np.random.normal(0, self.sxy, size=self.N)
        self.x[:, 1] += np.random.normal(0, self.sxy, size=self.N)
        self.x[:, 2] = _wrap_pi(self.x[:, 2] + np.random.normal(0, self.sth, size=self.N))

    @staticmethod
    def _bearing_from_xy(xy: np.ndarray) -> np.ndarray:
        # +ve = target to the LEFT of the optical axis, consistent with measurement model
        return np.arctan2(xy[:, 1], xy[:, 0])

    def update_bearing(self, bearing_obs: Optional[float]):
        if bearing_obs is None:
            return
        b_pred = self._bearing_from_xy(self.x[:, :2])
        err = _wrap_pi(b_pred - bearing_obs)
        # Gaussian likelihood
        ll = np.exp(-0.5 * (err / self.sb) ** 2) + 1e-9
        self.w *= ll.astype(np.float32)
        s = float(self.w.sum())
        if s <= 0:
            self.w.fill(1.0 / self.N)
        else:
            self.w /= s

    def resample(self):
        neff = 1.0 / np.sum(self.w ** 2)
        if neff < 0.5 * self.N:
            idx = np.random.choice(self.N, size=self.N, p=self.w)
            self.x = self.x[idx]
            self.w.fill(1.0 / self.N)

    def estimate(self) -> PFState:
        m = np.average(self.x, axis=0, weights=self.w)
        return PFState(float(m[0]), float(m[1]), float(m[2]))
