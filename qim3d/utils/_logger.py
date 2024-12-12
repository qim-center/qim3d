"""Logging configuration."""

import logging


def set_detailed_output():
    """Configures the logging output to display detailed information.

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
    logger = logging.getLogger("qim3d")
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
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)


def set_level_debug():
    """Sets the logging level of the "qim3d" logger to DEBUG.

    Example:
        >>> set_level_debug()
    """

    logging.getLogger("qim3d").setLevel(logging.DEBUG)


def set_level_info():
    """Sets the logging level of the "qim3d" logger to INFO.

    Example:
        >>> set_level_info()
    """

    logging.getLogger("qim3d").setLevel(logging.INFO)


def set_level_warning():
    """Sets the logging level of the "qim3d" logger to WARNING.

    Example:
        >>> set_level_warning()
    """

    logging.getLogger("qim3d").setLevel(logging.WARNING)


def set_level_error():
    """Sets the logging level of the "qim3d" logger to ERROR.

    Example:
        >>> set_level_error()
    """
    logging.getLogger("qim3d").setLevel(logging.ERROR)


def set_level_critical():
    """Sets the logging level of the "qim3d" logger to CRITICAL.

    Example:
        >>> set_level_critical()
    """
    logging.getLogger("qim3d").setLevel(logging.CRITICAL)


def level(log_level):
    """Set the logging level based on the specified level.

    Args:
        log_level (str or int): The logging level to set. It can be one of:
            - "DEBUG" or "debug": Set the logging level to DEBUG.
            - "INFO" or "info": Set the logging level to INFO.
            - "WARNING" or "warning": Set the logging level to WARNING.
            - "ERROR" or "error": Set the logging level to ERROR.
            - "CRITICAL" or "critical": Set the logging level to CRITICAL.
            - int: Set level to numeric value (e.g., logging.DEBUG).

    Raises:
        ValueError: If the specified level is not a valid logging level.

    """
    if log_level in ["DEBUG", "debug"]:
        set_level_debug()

    elif log_level in ["INFO", "info"]:
        set_level_info()

    elif log_level in ["WARNING", "warning"]:
        set_level_warning()

    elif log_level in ["ERROR", "error"]:
        set_level_error()

    elif log_level in ["CRITICAL", "critical"]:
        set_level_critical()

    elif isinstance(log_level, int):
        logging.getLogger("qim3d").setLevel(log_level)

    else:
        raise ValueError(
            f"Invalid logging level: '{log_level}'. Please use"
            "'debug', 'info', 'warning', 'error', 'critical' or an int."
        )


# create the logger
log = logging.getLogger("qim3d")
set_level_info()
set_simple_output()
