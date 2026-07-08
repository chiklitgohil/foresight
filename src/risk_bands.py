from src import config

def get_risk_band(probability: float, amber_threshold: float, red_threshold: float) -> tuple[str, str]:
    """
    Classifies a failure probability into a risk band using dynamic thresholds.
    Returns (Band Name, Recommended Action).
    """
    if probability < amber_threshold:
        return "green", "Normal operation"
    elif probability < red_threshold:
        return "amber", "Schedule inspection within 2 days"
    else:
        return "red", "Immediate inspection, stop the machine"
