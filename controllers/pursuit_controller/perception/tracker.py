from dataclasses import dataclass
from typing import Optional

@dataclass
class TrackState: #simple class to hold teacher info
    t: float = 0.0
    bearing: Optional[float] = None 
    range_m: Optional[float] = None
    visible: bool = False

class EmaTracker:
    #class smooths camera data to prevent jitter
    #using exponential moving average (ema) to blend old and new data
    def __init__(self, alpha=0.35, lose_after=6, reacquire_k=2.0): #following are explained in yaml file
        self.alpha = float(alpha) #how much to trust new data vs old
        self.lose_after = int(lose_after) #how many dropped frames tolerated before teacher declared not visible
        self.reacquire_k = float(reacquire_k) #avoid false positives when teacher reacquired
        self.miss = 0
        self.state = TrackState()

    def update(self, obs):
        s = self.state
        s.t = float(obs.t)

        if obs.visible:
            #reset miss counter if teacher seen
            self.miss = 0
            #update bearing (ema)
            if s.bearing is None: #first time seen, take raw val
                s.bearing = obs.bearing_rad
            else: #if lost before, reduce alpha, otherwise use normal alpha
                a = self.alpha / self.reacquire_k if s.visible is False else self.alpha
                #standard smoothing
                s.bearing = (1 - a) * s.bearing + a * obs.bearing_rad

            #update range (ema) if avail, use same smoothing
            if obs.range_m is not None:
                s.range_m = obs.range_m if s.range_m is None else (1 - self.alpha) * s.range_m + self.alpha * obs.range_m

            s.visible = True
        else: #nothing seen this frame
            self.miss += 1
            if self.miss >= self.lose_after: #if missed too many frames consecutively, mark as lost
                s.visible = False

        return s
