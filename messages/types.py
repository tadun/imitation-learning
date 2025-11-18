from dataclasses import dataclass
from typing import Optional

@dataclass
class PerceptionObs:
    t: float
    bearing_rad: float
    range_m: Optional[float]
    bearing_var: float
    range_var: Optional[float]
    visible: bool
