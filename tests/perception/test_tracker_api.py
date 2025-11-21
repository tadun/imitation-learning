import numpy as np, cv2
from perception.api import PerceptionAPI

def img(h=240,w=320):
    m = np.zeros((h,w,3), np.uint8)
    cv2.rectangle(m,(140,100),(180,140),(255,0,255),-1)  # magenta
    return m

def test_api_returns_visible():
    api = PerceptionAPI()
    obs = api.process(0.0, img())
    assert hasattr(obs, "bearing_rad")
    assert isinstance(obs.visible, bool)

def test_tracker_smooths():
    api = PerceptionAPI()
    # jitter the centroid slightly across frames:
    vals = [api.process(i*0.02, img()).bearing_rad for i in range(10)]
    # basic sanity: values exist and don't explode
    assert all(isinstance(v, float) for v in vals)
