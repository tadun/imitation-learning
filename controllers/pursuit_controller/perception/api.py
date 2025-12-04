from pathlib import Path
import yaml
from perception.types import PerceptionObs
from perception.detector import MarkerDetector
from perception.measurement import MeasurementModel
from perception.tracker import EmaTracker

class PerceptionAPI:
    def __init__(self, cfg_path=str(Path(__file__).with_name("marker_config.yaml"))):
        #save path to settings file
        self.cfg_path = cfg_path
        self.cfg = {}
        
        #load yaml safely
        if yaml:
            try:
                with open(cfg_path, "r") as f:
                    #load empty dict if failed
                    self.cfg = yaml.safe_load(f) or {}
            except FileNotFoundError:
                print(f"Warning: Config file {cfg_path} not found. Using defaults.")

        #init main stages of perception pipeline
        self.det  = MarkerDetector(cfg_path) #find blob
        self.meas = MeasurementModel(cfg_path) #calculations (pixels to metres)
        
        #set up tracker (noise smoothed)
        tcfg = self.cfg.get("tracker", {}) or {}
        self.trk  = EmaTracker(
            alpha=float(tcfg.get("alpha", 0.35)),
            lose_after=int(tcfg.get("lose_after", 6)),
            reacquire_k=float(tcfg.get("reacquire_k", 2.0)),
        )

    def process(self, t: float, bgr) -> PerceptionObs:
        #locate purple box
        blob = self.det.detect(bgr)
        #convert pixels to bearing + range
        obs = self.meas.from_blob(t, blob.u, blob.v, blob.area, blob.visible)
        #smooth vals
        st  = self.trk.update(obs)
        #return final obs, prefer smoothed vals        
        return PerceptionObs(
            t=obs.t,
            bearing_rad=st.bearing if st.bearing is not None else obs.bearing_rad,
            range_m=st.range_m if st.range_m is not None else obs.range_m,
            bearing_var=obs.bearing_var,
            range_var=obs.range_var,
            visible=st.visible,
        )