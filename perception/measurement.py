import numpy as np, yaml
from messages.types import PerceptionObs

class MeasurementModel:
    def __init__(self, cfg_path="perception/marker_config.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.fx, self.fy = cfg["fx"], cfg["fy"]
        self.cx, self.cy = cfg["cx"], cfg["cy"]
        self.L = cfg["marker_size_m"]
        self.pix_noise = cfg["pix_noise"]

    def from_blob(self, t, u, v, area_px, visible) -> PerceptionObs:
        if not visible:
            return PerceptionObs(t, 0.0, None, 1e3, None, False)
        bearing = np.arctan2((u - self.cx)/self.fx, 1.0)
        s_px = np.sqrt(max(area_px, 1))
        rng = max((self.fx * self.L) / s_px, 0.05)
        bearing_var = (self.pix_noise / self.fx)**2
        range_var = (self.pix_noise * self.fx * self.L / (s_px**2))**2
        return PerceptionObs(float(t), float(bearing), float(rng),
                             float(bearing_var), float(range_var), True)
