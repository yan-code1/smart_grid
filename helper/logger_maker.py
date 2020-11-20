import logging
import os, sys

def get_logger(logger_name, log_file = None):
    if log_file is None:
        log_file = os.path.join("logs", "%s.log" % (logger_name))

    logger = logging.getLogger(logger_name)
    for h in logger.root.handlers:
        logger.root.removeHandler(h)

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(
            "{asctime}.{msecs:03.0f} [{levelname[0]}:{lineno}]: {message}",
            "%Y%m%d %H:%M:%S",
            "{"
        )
    )
    file_handler.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter(
            "{asctime}.{msecs:03.0f} [{levelname[0]}]: {message}",
            "%H:%M:%S",
            "{"
        )
    )
    stdout_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.debug("=" * 12 + " Start logging " + "=" * 12)

    return logger