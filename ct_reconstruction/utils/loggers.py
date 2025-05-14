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
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')

        # Avoid duplicate handlers
        if not logger.handlers:
            # File handler (always)
            fh = logging.FileHandler(log_file, mode='w')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            # Console handler (only if debug)
            if debug:
                sh = logging.StreamHandler()
                sh.setFormatter(formatter)
                logger.addHandler(sh)

        return logger