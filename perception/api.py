# perception/api.py
from pathlib import Path
import yaml
from messages.types import PerceptionObs
from perception.detector import MarkerDetector
from perception.measurement import MeasurementModel
from perception.tracker import EmaTracker

class PerceptionAPI:
    """Detector -> Measurement -> Tracker. Returns PerceptionObs suitable for control."""
    def __init__(self, cfg_path=str(Path(__file__).with_name("marker_config.yaml"))):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f) or {}
        self.det = MarkerDetector(cfg_path)
        self.meas = MeasurementModel(cfg_path)
        tcfg = (self.cfg.get("tracker") or {})
        self.trk = EmaTracker(
            alpha=float(tcfg.get("alpha", 0.35)),
            lose_after=int(tcfg.get("lose_after", 6)),
            reacquire_k=float(tcfg.get("reacquire_k", 2.0)),
        )

    def process(self, t: float, bgr) -> PerceptionObs:
        blob = self.det.detect(bgr)
        obs = self.meas.from_blob(t, blob.u, blob.v, blob.area, blob.visible)
        st = self.trk.update(obs)
        # Return same dataclass type (smoothed where available)
        return PerceptionObs(
            t=obs.t,
            bearing_rad=st.bearing if st.bearing is not None else obs.bearing_rad,
            range_m=st.range_m if st.range_m is not None else obs.range_m,
            bearing_var=obs.bearing_var,
            range_var=obs.range_var,
            visible=st.visible,
        )
