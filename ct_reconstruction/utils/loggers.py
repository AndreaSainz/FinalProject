import logging

def configure_logger(name, log_file, debug= False) :
        """
        Configures and returns a logger that writes logs to a file and optionally to the console.

        This logger is useful for consistent logging across modules. It ensures that duplicate
        handlers are not added if the logger is called multiple times.

        Args:
            name (str): Logger name, typically set to `__name__` for module-level logging.
            log_file (str): Path to the file where logs will be saved (in append mode).
            debug (bool): If True, logs will also be printed to the console (stdout).

        Returns:
            logging.Logger: Configured logger instance ready for use.
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