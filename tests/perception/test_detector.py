import numpy as np, cv2
from perception.detector import MarkerDetector

def make_img(h=240, w=320, color=(255,0,255)):
    img = np.zeros((h,w,3), np.uint8)
    cv2.rectangle(img, (120,80), (200,160), color, -1)  # magenta block
    return img

def test_blank_frame_has_no_blob(tmp_path, monkeypatch):
    d = MarkerDetector()
    blank = np.zeros((240,320,3), np.uint8)
    b = d.detect(blank)
    assert b.visible is False and b.area == 0

def test_magenta_is_detected(tmp_path):
    d = MarkerDetector()
    img = make_img()
    b = d.detect(img)
    assert b.visible is True and b.area > 0
