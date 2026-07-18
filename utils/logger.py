import os
import sys
import logging


def setup_logger(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Sets up a standardized logger that writes to both console and a log file.

    Args:
        log_dir (str): Directory where logs should be saved.
        experiment_name (str): Identifier for the active run/experiment.

    Returns:
        logging.Logger: Configured logger object.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicating log messages when re-running in environments like Jupyter
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for log lines
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File output handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
