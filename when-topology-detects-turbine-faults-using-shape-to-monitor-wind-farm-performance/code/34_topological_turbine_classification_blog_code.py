"""
Code extracted from 34_topological_turbine_classification_blog.md
"""
'\nVerify that all required packages are installed and working.\nRun this before executing the main analysis.\n'
import sys
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors

def check_import(module_name, package_name=None):
    """Try to import a module and report success or failure."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        logger.info(f'✓ {package_name}')
        return True
    except ImportError as e:
        logger.error(f'✗ {package_name} - {str(e)}')
        return False

def main():
    """Check all required dependencies."""Check all required dependencies."""
    logger.info('Checking dependencies...\n')
    required = [('numpy', 'numpy'), ('pandas', 'pandas'), ('matplotlib', 'matplotlib'), ('sklearn', 'scikit-learn'), ('ripser', 'ripser'), ('persim', 'persim'), ('openoa', 'openoa')]
    all_good = True
    for module, package in required:
        if not check_import(module, package):
            all_good = False
    logger.info()
    if all_good:
        logger.info('✓ All dependencies are installed!')
        logger.info('\nYou can now run: python turbine_tda_analysis.py')
        return 0
    else:
        logger.error('✗ Some dependencies are missing.')
        logger.info('\nInstall them with: pip install -r requirements.txt')
        return 1
if __name__ == '__main__':
    sys.exit(main())