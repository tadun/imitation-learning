# controllers/perception_debug/perception_debug.py
# Webots controller to visualise + log perception (HSV mask + overlay + CSV).

# --- make repo root importable BEFORE any local imports ---
import sys, csv, time
from pathlib import Path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # .../imitation-learning
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print("[perception_debug] REPO_ROOT=", REPO_ROOT)

# --- std imports ---
import cv2
import numpy as np
from controller import Robot

# --- project imports (now that sys.path includes repo root) ---
from perception.detector import MarkerDetector
from perception.measurement import MeasurementModel
from perception.viz import draw_overlay


def get_any_camera(robot) -> tuple:
    """Try common device names; return (camera, name) or raise."""
    candidates = ["camera", "camera1", "kinect color", "kinect", "color"]
    for name in candidates:
        try:
            dev = robot.getDevice(name)
            if dev is not None:
                return dev, name
        except Exception:
            pass
    raise RuntimeError("No camera device found. Tried: " + ", ".join(candidates))


def to_bgr(camera) -> np.ndarray:
    """Convert Webots BGRA byte buffer to OpenCV BGR ndarray."""
    w, h = camera.getWidth(), camera.getHeight()
    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    cam, cam_name = get_any_camera(robot)
    print("[perception_debug] using camera device:", cam_name)
    cam.enable(timestep)

    cfg_path = str(REPO_ROOT / "perception" / "marker_config.yaml")
    det = MarkerDetector(cfg_path)
    meas = MeasurementModel(cfg_path)

    # CSV logging
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / f"perception_{int(time.time())}.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["t", "u", "v", "area", "visible", "bearing_rad", "range_m"])
    print("[perception_debug] Logging to", csv_path)

    t0 = robot.getTime()
    tick = 0
    while robot.step(timestep) != -1:
        t = robot.getTime() - t0
        bgr = to_bgr(cam)

        # Detect marker → blob
        blob = det.detect(bgr)

        # Pixels → measurement (bearing + optional range)
        obs = meas.from_blob(t, blob.u, blob.v, blob.area, blob.visible)

        # Debug HSV mask (uses detector's adaptive bounds)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo, hi = det._auto_hsv_bounds(hsv)  # acceptable for debugging
        mask = cv2.inRange(hsv, lo, hi)
        cv2.imshow("HSV mask", mask)

        # Overlay (green dot + bearing text)
        frame = draw_overlay(bgr, blob, obs.bearing_rad if obs.visible else None)
        cv2.imshow("Perception Debug (Webots)", frame)

        # Periodic console telemetry (≈5 Hz)
        if tick % max(1, int(1000 / max(1, timestep) / 5)) == 0:
            print(f"[dbg] vis={obs.visible} u={blob.u} v={blob.v} area={blob.area} "
                  f"bearing={obs.bearing_rad:+.3f} range={'' if obs.range_m is None else f'{obs.range_m:.2f}'}")

        # Log to CSV
        writer.writerow([
            f"{t:.3f}", blob.u, blob.v, blob.area, int(blob.visible),
            f"{obs.bearing_rad:.6f}",
            "" if obs.range_m is None else f"{obs.range_m:.3f}"
        ])

        tick += 1
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    f.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
