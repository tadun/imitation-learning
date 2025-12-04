import numpy as np, yaml
from perception.types import PerceptionObs
#Measurement model class: turn raw image into distance/bearing
#using pinhole camera model
class MeasurementModel:
    def __init__(self, cfg_path="perception/marker_config.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        #load focal length + centre
        self.fx = float(cfg["fx"]); self.fy = float(cfg["fy"])
        self.cx = float(cfg["cx"]); self.cy = float(cfg["cy"])
        self.L  = float(cfg.get("marker_size_m", 0.08)) #marker physical size (m)
        self.pix_noise = float(cfg.get("pix_noise", 1.5)) #noise level
        self.area_min_for_range = int(cfg.get("min_area", 60)) * 2 #ignore tiny blobs
        
        #ablation functionality as specified in proposal 
        #feature level 1=bearing only
        feats = (cfg.get("features") or {})
        self.feature_level = int(feats.get("level", 2))

    def from_blob(self, t, u, v, area_px, visible) -> PerceptionObs:
        #if nothing detected return none with high variance
        if not visible:
            return PerceptionObs(t=float(t), bearing_rad=0.0, range_m=None,
                                 bearing_var=1e3, range_var=None, visible=False)
                                 
        #calculate bearing based on pixel, using basic trig
        bearing = float(np.arctan2((u - self.cx) / self.fx, 1.0))

        rng, rvar = None, None
        
        #only do the following if bearing-only mode NOT enabled
        if self.feature_level >= 2:
            #only guess distance if the blob is larger than min threshold
            if area_px and area_px >= self.area_min_for_range:
                #area -> width
                s_px = float(max(area_px, 1)) ** 0.5
                
                #Pinhole maths: distance = (focal_length * real_size) / pixel_size
                #i.e. bigger it looks, closer it is
                raw_rng = (self.fx * self.L) / s_px
                
                #saftey check, min rng is 0.2, lower vals could break maths
                rng  = max(raw_rng, 0.2) 
                
                #variance increases with distance squared
                #close = sure, far = unsure
                rvar = (self.pix_noise * raw_rng**2 / (self.fx * self.L)) ** 2

        #variance for bearing depends on pix noise and focal length
        bvar = (self.pix_noise / self.fx) ** 2
        
        return PerceptionObs(t=float(t), bearing_rad=bearing,
                             range_m=(None if rng  is None else float(rng)),
                             bearing_var=float(bvar),
                             range_var=(None if rvar is None else float(rvar)),
                             visible=True)