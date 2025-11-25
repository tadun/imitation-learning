# perception/api.py
from pathlib import Path
import yaml
from messages.types import PerceptionObs
from perception.detector import MarkerDetector
from perception.measurement import MeasurementModel
from perception.tracker import EmaTracker
from perception.particle_filter import VisionPF, PFState

class PerceptionAPI:
    """Detector -> Measurement -> Tracker (+ optional PF). Control consumes PerceptionObs / PFState."""
    def __init__(self, cfg_path=str(Path(__file__).with_name("marker_config.yaml"))):
        self.cfg_path = cfg_path
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.det  = MarkerDetector(cfg_path)
        self.meas = MeasurementModel(cfg_path)
        tcfg = self.cfg.get("tracker", {}) or {}
        self.trk  = EmaTracker(
            alpha=float(tcfg.get("alpha", 0.35)),
            lose_after=int(tcfg.get("lose_after", 6)),
            reacquire_k=float(tcfg.get("reacquire_k", 2.0)),
        )
        self._pf = None  # lazy init

    @property
    def pf(self) -> VisionPF:
        if self._pf is None:
            pfc = self.cfg.get("pf", {}) or {}
            self._pf = VisionPF(
                N=int(pfc.get("N", 500)),
                sigma_proc_xy=float(pfc.get("sigma_proc_xy", 0.03)),
                sigma_proc_theta=float(pfc.get("sigma_proc_theta", 0.05)),
                sigma_bearing=float(pfc.get("sigma_bearing", 0.10)),
            )
        return self._pf

    # Classic perception hand-off (no PF)
    def process(self, t: float, bgr) -> PerceptionObs:
        blob = self.det.detect(bgr)
        obs = self.meas.from_blob(t, blob.u, blob.v, blob.area, blob.visible)
        st  = self.trk.update(obs)
        # return smoothed values where present
        return PerceptionObs(
            t=obs.t,
            bearing_rad=st.bearing if st.bearing is not None else obs.bearing_rad,
            range_m=st.range_m if st.range_m is not None else obs.range_m,
            bearing_var=obs.bearing_var,
            range_var=obs.range_var,
            visible=st.visible,
        )

    # PF-enabled path (tonight: bearing-only)
    def process_pf(self, t: float, bgr) -> tuple[PerceptionObs, PFState]:
        blob = self.det.detect(bgr)
        obs = self.meas.from_blob(t, blob.u, blob.v, blob.area, blob.visible)
        st  = self.trk.update(obs)

        # Predict with dt derived from obs_rate_hz (rough but fine for tonight)
        obs_rate = float(self.cfg.get("obs_rate_hz", 10))
        dt = 1.0 / max(1.0, obs_rate)
        self.pf.predict(dt)

        # Update with smoothed bearing (fallback to raw)
        b = st.bearing if st.bearing is not None else obs.bearing_rad
        self.pf.update_bearing(b)
        self.pf.resample()
        est = self.pf.estimate()

        # Return obs (for control) and PF state (for logging/optionally control)
        return PerceptionObs(
            t=obs.t,
            bearing_rad=b,
            range_m=st.range_m if st.range_m is not None else obs.range_m,
            bearing_var=obs.bearing_var,
            range_var=obs.range_var,
            visible=st.visible,
        ), est
