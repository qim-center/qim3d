import logging


def set_detailed_output():
    formatter = logging.Formatter(
        "%(levelname)-10s%(filename)s:%(lineno)-5s%(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger("qim3d")
    logger.handlers = []
    logger.addHandler(handler)


def set_simple_output():
    formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger("qim3d")
    logger.handlers = []
    logger.addHandler(handler)


def set_level_DEBUG():
    logging.getLogger("qim3d").setLevel(logging.DEBUG)


def set_level_INFO():
    logging.getLogger("qim3d").setLevel(logging.INFO)


def set_level_WARNING():
    logging.getLogger("qim3d").setLevel(logging.WARNING)


def set_level_ERROR():
    logging.getLogger("qim3d").setLevel(logging.ERROR)


def set_level_CRITICAL():
    logging.getLogger("qim3d").setLevel(logging.CRITICAL)
