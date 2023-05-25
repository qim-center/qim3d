import logging

log = logging.getLogger()


def set_detailed_output(logger):
    formatter = logging.Formatter(
        "%(levelname)-10s%(filename)s:%(lineno)-5s%(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.handlers = [handler]


def set_simple_output(logger):
    formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.handlers = [handler]


def set_level_DEBUG():
    log.setLevel(logging.DEBUG)


def set_level_INFO():
    log.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s")


def set_level_WARNING():
    log.setLevel(logging.WARNING)
    logging.basicConfig(format="%(message)s")


def set_level_ERROR():
    log.setLevel(logging.ERROR)
    logging.basicConfig(format="%(message)s")


def set_level_CRITICAL():
    log.setLevel(logging.CRITICAL)
    logging.basicConfig(format="%(message)s")
