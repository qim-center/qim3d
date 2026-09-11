"""Logging configuration."""

import logging

logger = logging.getLogger("qim3d")


def set_detailed_output():
    """
    Configures the logging output to display detailed information.

    This function sets up a logging formatter with a specific format that
    includes the log level, filename, line number, and log message.

    Example:
        >>> set_detailed_output()

    """

    formatter = logging.Formatter(
        "%(levelname)-10s%(filename)s:%(lineno)-5s%(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)


def set_simple_output():
    """
    Configures the logging output to display simple messages.

    This function sets up a logging formatter with a format that includes only
    the log message.

    Example:
        >>> set_simple_output()

    """
    formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)


def initialize_logger(level: str | int, detailed: bool = False):
    """
    Initializes the logger with the specified log level and output format.

    Args:
        level (str or int): The log level to set for the logger. It can be a
            string (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") or
            an integer corresponding to the log level.
            Note that `logging.INFO` is an int, so those enum-like values can be used.
        detailed (bool, optional): If True, sets the output format to detailed.
    """

    logger.setLevel(level)
    if detailed:
        set_detailed_output()
    else:
        set_simple_output()
