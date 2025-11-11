from dataclasses import dataclass
from typing import Optional

@dataclass
class PerceptionObs:
    t: float                   # seconds (monotonic)
    bearing_rad: float         # +left / -right from camera axis
    range_m: Optional[float]   # None when unreliable
    bearing_var: float         # variance (rad^2)
    range_var: Optional[float] # variance (m^2) or None
    visible: bool              # marker currently seen
