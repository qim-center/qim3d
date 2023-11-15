"""Provides tools for obtaining information about the system."""
import psutil
from qim3d.utils.internal_tools import sizeof
from qim3d.io.logger import log


class Memory:
    """Class for obtaining current memory information

    Attributes:
        total (int): Total system memory in bytes
        free (int): Free system memory in bytes
        used (int): Used system memory in bytes
        used_pct (float): Used system memory in percentage
    """

    def __init__(self):
        mem = psutil.virtual_memory()

        self.total = mem.total
        self.free = mem.available
        self.free_pct = (mem.available / mem.total) * 100
        self.used = mem.total - mem.available
        self.used_pct = mem.percent

    def report(self):
        log.info(
            "System memory:\n • Total.: %s\n • Used..: %s (%s%%)\n • Free..: %s (%s%%)",
            sizeof(self.total),
            sizeof(self.used),
            round(self.used_pct, 1),
            sizeof(self.free),
            round(self.free_pct, 1),
        )
