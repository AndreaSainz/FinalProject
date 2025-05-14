import logging

def configure_logger(name, log_file, debug= False) :
        """
        Set up a logger that logs to both file and optionally console.

        Args:
            name (str): Name of the logger (usually __name__).
            log_file (str): Path to the log file.
            debug (bool): If True, also logs to the console.

        Returns:
            logging.Logger: Configured logger instance.
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