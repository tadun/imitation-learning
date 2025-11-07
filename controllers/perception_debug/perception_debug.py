
import sys, csv, time
from pathlib import Path

# Make repo root importable
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.append(str(REPO_ROOT))

import cv2
import numpy as np
from controller import Robot, Camera

import sys
from pathlib import Path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # .../imitation-learning
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"[perception_debug] REPO_ROOT={REPO_ROOT}") 

from perception.detector import MarkerDetector
from perception.measurement import MeasurementModel
from perception.viz import draw_overlay


def to_bgr(camera: Camera) -> np.ndarray:
    """Convert Webots BGRA image to OpenCV BGR."""
    w, h = camera.getWidth(), camera.getHeight()
    # Webots returns BGRA bytes (height, width, 4)
    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    cam = robot.getDevice("camera")
    assert cam is not None, "No device named 'camera'. Check the robot's device list in the world."
    cam.enable(timestep)

    cfg = str(REPO_ROOT / "perception" / "marker_config.yaml")
    det = MarkerDetector(cfg_path=cfg)
    meas = MeasurementModel(cfg_path=cfg)

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / f"perception_{int(time.time())}.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["t", "u", "v", "area", "visible", "bearing_rad", "range_m"])

    print(f"[perception_debug] Logging to {csv_path}")

    t0 = robot.getTime()
    while robot.step(timestep) != -1:
        t = robot.getTime() - t0

        bgr = to_bgr(cam)
        blob = det.detect(bgr)
        obs = meas.from_blob(t, blob.u, blob.v, blob.area, blob.visible)

        writer.writerow([
            f"{t:.3f}", blob.u, blob.v, blob.area, int(blob.visible),
            f"{obs.bearing_rad:.6f}",
            "" if obs.range_m is None else f"{obs.range_m:.3f}"
        ])

        # Overlay window (close with ESC). Comment out if headless.
        frame = draw_overlay(bgr, blob, obs.bearing_rad if obs.visible else None)
        cv2.imshow("Perception Debug (Webots)", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    f.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
