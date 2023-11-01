import logging


def set_log_level(log_level: str):
    """Sets the logging level for the application.

    Args:
        log_level (str): The desired logging level. Valid options are 'critical', 'error', 'warning', 'info', 'debug', and 'notset'.
    """
    log_levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET,
    }
    logging.basicConfig(level=log_levels[log_level])
