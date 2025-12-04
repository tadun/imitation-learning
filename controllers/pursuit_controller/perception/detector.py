import cv2, numpy as np, yaml
from dataclasses import dataclass

@dataclass #simple class to hold what is detected
class Blob:
    u: int
    v: int
    area: int
    visible: bool

class MarkerDetector:
    def __init__(self, cfg_path="perception/marker_config.yaml"):
        #load settings for marker yaml file
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        #create kernels for image cleanup
        #open and close commands remove small noise and fill in holes
        self.k_open  = np.ones((3,3), np.uint8)
        self.k_close = np.ones((7,7), np.uint8)

    def detect(self, bgr) -> Blob:
        #switch to hsv colour space because easier to filter colours that way
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        #autobounding based on image percentiles does not work when marker fills screen
        #so use fixed numbers to prevent breaking        
        lo = np.array([
            self.cfg["h_lo"], 
            self.cfg["s_p_low"], 
            self.cfg["v_p_low"]
        ], np.uint8)
        
        hi = np.array([
            self.cfg["h_hi"], 
            255, 
            255 #always allow bright things through
        ], np.uint8)
        #create a black and white mask - white signifies where the marker is
        mask = cv2.inRange(hsv, lo, hi)
        
        #remove static and noise in mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        
        #show mask on new window
        cv2.imshow("mask obj detection view", mask)
        cv2.waitKey(1)

        #find outline of shape in mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #return nothing if nothing found
        if not cnts: 
            return Blob(0, 0, 0, False)
            
        #assume biggest contour is marker
        #this is why exotic colour chosen for marker, prevent misidentifying
        c = max(cnts, key=cv2.contourArea)
        area = int(cv2.contourArea(c))
        
        #ignore if too small, most likely floor noise in this case
        if area < int(self.cfg["min_area"]): 
            return Blob(0, 0, 0, False)
        
        #calculate centre point of assumed marker blob    
        M = cv2.moments(c)
        if M["m00"] == 0: 
            return Blob(0, 0, 0, False)
            
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        
        return Blob(u, v, area, True)