import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
'\nCode extracted from THREE_APPROACHES_COMPARISON.md\n'
expected_power = power_curve(wind_speed)
deviation = actual_power - expected_power
power_ratio = actual_power / expected_power
label = 1 if mean(power_ratio) < 0.8 else 0