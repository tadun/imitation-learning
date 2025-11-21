# perception/recovery.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class RecoveryHint:
    suggest_sweep: bool
    sweep_center_rad: float   # where to center the yaw sweep (e.g., last known bearing)
    sweep_width_rad: float    # total width to sweep (e.g., ±width/2)
    note: str = ""

def compute_recovery_hint(last_bearing: Optional[float], lost: bool,
                          default_width_rad=0.9) -> RecoveryHint:
    if not lost:
        return RecoveryHint(False, 0.0, 0.0, "target visible")
    center = float(last_bearing) if last_bearing is not None else 0.0
    return RecoveryHint(True, center, float(default_width_rad), "lost; perform yaw sweep")
