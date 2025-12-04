from dataclasses import dataclass
from typing import Optional

@dataclass #dataclass is better than a normal class here, no need for init function
class PerceptionObs:
    t: float #timestamp of measurement
    bearing_rad: float #angle to teacher
    range_m: Optional[float] #distance to teacher (optional)
    bearing_var: float #bearing variance (bearing uncertainy)
    range_var: Optional[float] #distance variance
    visible: bool #simple yes/no flag