"""
Topological turbine classification dependency check. Run from repo root: python path/to/34_topological_turbine_classification_blog_code.py [--config path/to/config.yaml]
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
for _ in range(15):
    if (_REPO_ROOT / "config" / "default.yaml").is_file() or (_REPO_ROOT / "pyproject.toml").is_file():
        break
    _REPO_ROOT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR.parent))

import logging
from config.load import load_config
from tda_utils import setup_tufte_plot, TufteColors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

def main(config_path=None):
    """Check all required dependencies. Optionally load config for consistency."""
    load_config(config_path)
    logger.info("Checking dependencies...\n")
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
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blog code: dependency check")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    sys.exit(main(config_path=args.config))