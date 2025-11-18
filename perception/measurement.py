import numpy as np, yaml
from messages.types import PerceptionObs

class MeasurementModel:
    """Pixel centroid -> bearing (+optional range) using pinhole intrinsics."""
    def __init__(self, cfg_path="perception/marker_config.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.fx = float(cfg["fx"]); self.fy = float(cfg["fy"])
        self.cx = float(cfg["cx"]); self.cy = float(cfg["cy"])
        self.L  = float(cfg.get("marker_size_m", 0.08))
        self.pix_noise = float(cfg.get("pix_noise", 1.5))
        self.area_min_for_range = int(cfg.get("min_area", 60)) * 2

    def from_blob(self, t, u, v, area_px, visible) -> PerceptionObs:
        if not visible:
            return PerceptionObs(float(t), 0.0, None, 1e3, None, False)

        # +bearing means target is to the LEFT of the optical axis
        bearing = float(np.arctan2((u - self.cx) / self.fx, 1.0))

        if not area_px or area_px < self.area_min_for_range:
            rng, rvar = None, None
        else:
            s_px = float(max(area_px, 1)) ** 0.5
            rng  = max((self.fx * self.L) / s_px, 0.05)
            rvar = (self.pix_noise * self.fx * self.L / (s_px ** 2)) ** 2

        bvar = (self.pix_noise / self.fx) ** 2
        return PerceptionObs(
            t=float(t), bearing_rad=bearing,
            range_m=None if rng  is None else float(rng),
            bearing_var=float(bvar),
            range_var=None if rvar is None else float(rvar),
            visible=True
        )
