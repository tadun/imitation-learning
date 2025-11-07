import numpy as np, cv2
from perception.detector import MarkerDetector

def test_detector_runs_on_blank_frame():
    det = MarkerDetector()
    blank = np.zeros((480,640,3), dtype=np.uint8)
    blob = det.detect(blank)
    assert blob.visible is False
