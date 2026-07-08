import pytest
from src.risk_bands import get_risk_band

def test_risk_bands_boundaries():
    amber_thresh = 0.25
    red_thresh = 0.50

    # 0.0
    band, action = get_risk_band(0.0, amber_thresh, red_thresh)
    assert band == "green"
    assert action == "Normal operation"
    
    # 0.249
    band, action = get_risk_band(0.249, amber_thresh, red_thresh)
    assert band == "green"
    assert action == "Normal operation"
    
    # 0.25
    band, action = get_risk_band(0.25, amber_thresh, red_thresh)
    assert band == "amber"
    assert action == "Schedule inspection within 2 days"
    
    # 0.499
    band, action = get_risk_band(0.499, amber_thresh, red_thresh)
    assert band == "amber"
    assert action == "Schedule inspection within 2 days"
    
    # 0.50
    band, action = get_risk_band(0.50, amber_thresh, red_thresh)
    assert band == "red"
    assert action == "Immediate inspection, stop the machine"
    
    # 1.0
    band, action = get_risk_band(1.0, amber_thresh, red_thresh)
    assert band == "red"
    assert action == "Immediate inspection, stop the machine"
