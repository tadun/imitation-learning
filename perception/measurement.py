import numpy as np, yaml
from messages.types import PerceptionObs

class MeasurementModel:
    """Map pixel centroid -> bearing (+optional range) using pinhole intrinsics."""
    def __init__(self, cfg_path="perception/marker_config.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.fx = float(cfg["fx"]); self.fy = float(cfg["fy"])
        self.cx = float(cfg["cx"]); self.cy = float(cfg["cy"])
        self.L  = float(cfg.get("marker_size_m", 0.08))         # marker physical size (m)
        self.pix_noise = float(cfg.get("pix_noise", 1.5))
        self.area_min_for_range = int(cfg.get("min_area", 60)) * 2
        # Ablation: features.level (1=bearing only)
        feats = (cfg.get("features") or {})
        self.feature_level = int(feats.get("level", 2))

    def from_blob(self, t, u, v, area_px, visible) -> PerceptionObs:
        if not visible:
            return PerceptionObs(t=float(t), bearing_rad=0.0, range_m=None,
                                 bearing_var=1e3, range_var=None, visible=False)
        # +bearing = target to the left of optical axis
        bearing = float(np.arctan2((u - self.cx) / self.fx, 1.0))

        rng, rvar = None, None
        if self.feature_level >= 2:
            # crude range from apparent size; gate tiny blobs
            if area_px and area_px >= self.area_min_for_range:
                s_px = float(max(area_px, 1)) ** 0.5
                rng  = max((self.fx * self.L) / s_px, 0.05)
                rvar = (self.pix_noise * self.fx * self.L / (s_px ** 2)) ** 2

        bvar = (self.pix_noise / self.fx) ** 2
        return PerceptionObs(t=float(t), bearing_rad=bearing,
                             range_m=(None if rng  is None else float(rng)),
                             bearing_var=float(bvar),
                             range_var=(None if rvar is None else float(rvar)),
                             visible=True)
