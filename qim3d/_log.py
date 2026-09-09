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


# Set up logging configuration
set_detailed_output()
logger.setLevel(logging.INFO)
