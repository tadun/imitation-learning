import cv2

def draw_overlay(bgr, blob, bearing_rad=None):
    out = bgr.copy()
    if getattr(blob, "visible", False):
        cv2.circle(out, (blob.u, blob.v), 6, (0,255,0), 2)
    if bearing_rad is not None:
        cv2.putText(out, f"bearing {bearing_rad:+.3f} rad", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out
