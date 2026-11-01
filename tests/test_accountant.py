import math

from coadaptive.accounting import accountant, calibrate


def test_review_arithmetic():
    alpha = 2.1
    delta_sensitivity = 1.70
    sigma = 15.3
    rdp = alpha * delta_sensitivity**2 / (2 * sigma**2)
    conversion = math.log(1 / 1e-5) / (alpha - 1)
    assert abs(rdp - 0.01296296) < 1e-8
    assert abs(conversion - 10.46629588) < 1e-8


def test_calibration_hits_target():
    z = calibrate(events=3, target=2.0, delta=1e-5)
    best, _ = accountant(events=3, z=z, delta=1e-5)
    assert best["epsilon"] <= 2.0 + 1e-10
    assert best["epsilon"] > 1.999
