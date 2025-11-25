# controllers/perception_debug/perception_debug.py
# Visualise + log perception with PerceptionAPI (detector→measurement→tracker→PF).

# ---------- repo root on sys.path ----------
import sys, csv, time
from pathlib import Path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print("[perception_debug] REPO_ROOT=", REPO_ROOT)
# ------------------------------------------

import cv2, numpy as np, yaml
from controller import Robot
from perception.api import PerceptionAPI
from perception.viz import draw_overlay

def get_any_camera(robot):
    for name in ["camera", "camera1", "kinect color", "kinect", "color"]:
        try:
            dev = robot.getDevice(name)
            if dev is not None:
                return dev, name
        except Exception:
            pass
    raise RuntimeError("No camera device found.")

def to_bgr(cam):
    w, h = cam.getWidth(), cam.getHeight()
    img = np.frombuffer(cam.getImage(), dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Config (for throttling)
    cfg = yaml.safe_load(open(REPO_ROOT / "perception" / "marker_config.yaml", "r")) or {}
    obs_rate = int(cfg.get("obs_rate_hz", 10))
    step_div = max(1, int((1000 / max(1, timestep)) / obs_rate))  # frames to skip

    # Camera
    cam, cam_name = get_any_camera(robot)
    print("[perception_debug] using camera device:", cam_name)
    cam.enable(timestep)

    # API
    api = PerceptionAPI(str(REPO_ROOT / "perception" / "marker_config.yaml"))

    # CSV logging
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / f"perception_{int(time.time())}.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["t","u","v","area","visible","bearing_rad","range_m","x_pf","y_pf","theta_pf"])
    print("[perception_debug] Logging to", csv_path)

    t0 = robot.getTime()
    tick = 0
    while robot.step(timestep) != -1:
        # Throttle processing to obs_rate_hz
        if tick % step_div != 0:
            tick += 1
            continue

        t = robot.getTime() - t0
        bgr = to_bgr(cam)

        # PF-enabled perception
        obs, pf = api.process_pf(t, bgr)

        # Draw overlay using detector centroid (fast)
        blob = api.det.detect(bgr)

        # HSV mask
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo, hi = api.det._auto_hsv_bounds(hsv)  # ok for debug
        mask = cv2.inRange(hsv, lo, hi)
        cv2.imshow("HSV mask", mask)

        frame = draw_overlay(bgr, blob, obs.bearing_rad if obs.visible else None)
        cv2.imshow("Perception Debug (Webots)", frame)

        # Console (optional)
        print(f"[dbg] vis={obs.visible} u={blob.u} v={blob.v} area={blob.area} "
              f"bearing={obs.bearing_rad:+.3f} "
              f"range={'' if obs.range_m is None else f'{obs.range_m:.2f}'} "
              f"pf=({pf.x:+.2f},{pf.y:+.2f},{pf.theta:+.2f})")

        writer.writerow([
            f"{t:.3f}", blob.u, blob.v, blob.area, int(bool(blob.visible)),
            f"{obs.bearing_rad:.6f}",
            "" if obs.range_m is None else f"{float(obs.range_m):.3f}",
            f"{pf.x:.3f}", f"{pf.y:.3f}", f"{pf.theta:.3f}"
        ])

        tick += 1
        if cv2.waitKey(1) == 27:  # ESC
            break

    f.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
