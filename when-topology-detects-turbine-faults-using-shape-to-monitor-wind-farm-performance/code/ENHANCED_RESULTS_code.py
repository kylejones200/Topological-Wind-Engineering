import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
'\nCode extracted from ENHANCED_RESULTS.md\n'
capacity_factor = average_power / rated_power
label = 1 if capacity_factor > 0.35 else 0