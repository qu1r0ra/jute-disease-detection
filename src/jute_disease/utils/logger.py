import logging
import sys


def setup_logging(level=logging.INFO):
    """Set up standardized logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Check if handlers already exist to avoid duplicate logs
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Silence noisy loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def get_logger(name, level=logging.INFO):
    """Get a logger with the specified name and ensure logging is set up."""
    setup_logging(level)
    return logging.getLogger(name)
