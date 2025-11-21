# perception/tracker.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrackState:
    t: float = 0.0
    bearing: Optional[float] = None
    range_m: Optional[float] = None
    visible: bool = False
    quality: float = 0.0  # 0..1

class EmaTracker:
    """
    Exponential moving-average tracker for bearing/range.
    Keeps state through brief dropouts (lost after N misses).
    """
    def __init__(self, alpha=0.35, lose_after=6, reacquire_k=2.0):
        self.alpha = float(alpha)
        self.lose_after = int(lose_after)
        self.reacquire_k = float(reacquire_k)
        self.miss = 0
        self.state = TrackState()

    def update(self, obs):
        s = self.state
        s.t = float(obs.t)

        if obs.visible:
            self.miss = 0
            # EMA on bearing
            if s.bearing is None:
                s.bearing = obs.bearing_rad
            else:
                a = self.alpha / self.reacquire_k if s.visible is False else self.alpha
                s.bearing = (1 - a) * s.bearing + a * obs.bearing_rad

            # EMA on range (if available)
            if obs.range_m is not None:
                s.range_m = obs.range_m if s.range_m is None else (1 - self.alpha) * s.range_m + self.alpha * obs.range_m

            # very simple quality (visible → good, decays otherwise)
            s.quality = min(1.0, 0.85 * s.quality + 0.3)
            s.visible = True
        else:
            self.miss += 1
            if self.miss >= self.lose_after:
                s.visible = False
            s.quality *= 0.85  # decay

        return s
