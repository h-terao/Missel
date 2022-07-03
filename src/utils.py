import logging
import os
import shutil
import hashlib
from pathlib import Path
import warnings

from omegaconf import DictConfig, OmegaConf


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # # this ensures all logging levels get marked with the rank zero decorator
    # # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_hash_from_file(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def setup_main(config: DictConfig) -> None:
    """Setup for the experiment.
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - move to the previous experiment directory and remove the current directory
      if found and autoload=True.
    - save config.yaml if autoload=True.

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    log = get_logger(__name__)

    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    OmegaConf.save(config, "./config.yaml", resolve=True)

    cwd_path = Path(os.getcwd())
    if "multiruns" in str(cwd_path) or not config.autoload or config.get("debug_mode"):
        # If use optuna, avoid to search previous experiment
        return

    log_dir_path = Path(config.work_dir, "logs")

    # Get hash of current configure.
    current_hash = get_hash_from_file("./config.yaml")

    # Compare hash of current configure with hashes of all experiments in logs/runs/.
    for config_path in log_dir_path.glob("**/config.yaml"):
        prev_hash = get_hash_from_file(config_path)
        if current_hash == prev_hash and config_path != (cwd_path / "config.yaml"):
            log.info(f"The previous experiment is found. Move to {str(config_path.parent)}.")
            os.chdir(config_path.parent)  # Move to the previous experiment.
            shutil.rmtree(cwd_path)  # Remove current directory. There is no experiment.
            break
