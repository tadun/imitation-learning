import cv2, numpy as np, yaml
from dataclasses import dataclass

@dataclass
class Blob:
    u:int; v:int; area:int; visible:bool

class MarkerDetector:
    def __init__(self, cfg_path="perception/marker_config.yaml"):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def _auto_hsv_bounds(self, hsv):
        h,s,v = cv2.split(hsv)
        vmin, vmax = np.percentile(v, [self.cfg["v_p_low"], self.cfg["v_p_high"]])
        smin = np.percentile(s, self.cfg["s_p_low"])
        lo = np.array([self.cfg["h_lo"], max(5, smin), max(5, vmin)], np.uint8)
        hi = np.array([self.cfg["h_hi"], 255, min(255, vmax)], np.uint8)
        return lo, hi

    def detect(self, bgr) -> Blob:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo, hi = self._auto_hsv_bounds(hsv)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return Blob(0,0,0,False)
        c = max(cnts, key=cv2.contourArea)
        area = int(cv2.contourArea(c))
        if area < self.cfg["min_area"]: return Blob(0,0,0,False)
        M = cv2.moments(c); u = int(M["m10"]/M["m00"]); v = int(M["m01"]/M["m00"])
        return Blob(u,v,area,True)
