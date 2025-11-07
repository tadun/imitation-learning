import numpy as np, cv2
from perception.detector import MarkerDetector

def test_detector_runs_on_blank():
    det = MarkerDetector()
    frame = np.zeros((480,640,3), np.uint8)
    blob = det.detect(frame)
    assert blob.visible is False

def test_detector_finds_coloured_blob():
    det = MarkerDetector()
    frame = np.zeros((480,640,3), np.uint8)
    # Draw a magenta-ish circle (BGR: 255,0,255)
    cv2.circle(frame, (320,240), 25, (255,0,255), -1)
    blob = det.detect(frame)
    assert blob.visible
    assert abs(blob.u-320) < 10 and abs(blob.v-240) < 10
