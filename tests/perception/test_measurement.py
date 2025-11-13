from perception.measurement import MeasurementModel

def test_measurement_invisible():
    m = MeasurementModel()
    obs = m.from_blob(0.0, 0, 0, 0, False)
    assert obs.visible is False and obs.range_m is None
