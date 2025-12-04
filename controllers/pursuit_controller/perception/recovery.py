from dataclasses import dataclass
from typing import Optional

@dataclass #simple class for recovery
class RecoveryHint:
    suggest_sweep: bool
    sweep_center_rad: float   # sweep centre based on last known teacher bearing
    sweep_width_rad: float    
    note: str = ""

def compute_recovery_hint(last_bearing: Optional[float], lost: bool,
                          default_width_rad=0.9) -> RecoveryHint:
    #do nothing if teacher visible
    if not lost:
        return RecoveryHint(False, 0.0, 0.0, "target visible")
    
    #if this part of method reached, teacher is lost        
    #set last known teacher bearing
    center = float(last_bearing) if last_bearing is not None else 0.0    
    
    #return recovery hint with necessary information to relocate teacher
    return RecoveryHint(True, center, float(default_width_rad), "lost; perform yaw sweep")
