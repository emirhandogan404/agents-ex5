from pathlib import Path
import tomllib
import rootpath

PROJECT_DIR = Path(rootpath.detect(pattern="pyproject.toml"))
DATA_DIR = PROJECT_DIR / "data"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
PLOTS_DIR = PROJECT_DIR / "plots"
SRC_DIR = PROJECT_DIR / "src"

MODEL_CHKPT_DIR=Path("/sc/projects/sci-lippert/intelligent-agents/model_checkpoints")

CONFIG = tomllib.load(open(PROJECT_DIR / "cfg.toml", "rb"))