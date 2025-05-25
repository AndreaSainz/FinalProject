"""
Logger configuration utility for consistent project-wide logging.

This module provides a utility function to create and configure Python loggers
that write logs to both a file and, optionally, the console. It ensures that
duplicate handlers are not added and applies a consistent formatting style.

Typical usage involves importing `configure_logger` and initializing a logger
at the top of each module or script.

Functions:
    configure_logger(name, log_file, debug=False): Returns a configured logger instance.
"""

import logging

def configure_logger(name, log_file, debug= False) :
    """
    Creates and configures a logger that writes logs to a file and optionally to the console.

    The logger uses a consistent format and avoids duplicate handlers, making it suitable
    for module-level or project-wide logging.

    Args:
        name (str): Name of the logger instance, typically `__name__`.
        log_file (str): Path to the log file where outputs will be saved in append mode.
        debug (bool): If True, also prints logs to the console via stdout.

    Returns:
        logging.Logger: A configured logger instance with file and optional stream handlers.

    Example:
        >>> from ct_reconstruction.utils.logger import configure_logger
        >>> logger = configure_logger(__name__, "logs/run.log", debug=True)
        >>> logger.info("Logger initialized.")
    """
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler (always)
        file_handler = logging.FileHandler(log_file, mode='a')  # append
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler (only if debug is True)
        if debug:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger