"""Provides tools for obtaining information about the system."""
import os
import time
import psutil
from qim3d.utils.misc import sizeof
from qim3d.utils.logger import log
import numpy as np


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

def _test_disk_speed(file_size_bytes=1024, ntimes=10):
    '''
    Test the write and read speed of the disk by writing a file of a given size
    and then reading it back.

    Args:
        file_size_bytes (int): Size of the file to write in bytes

    Returns:
        write_speed (float): Write speed in GB/s
        read_speed (float): Read speed in GB/s

    Example:
        ```python
        import qim3d
        file_size = 1024 * 1024 * 5000  # 5 GB
        write_speed, read_speed = qim3d.utils.system.test_disk_speed(file_size)
        print(f"Write speed: {write_speed:.2f} GB/s")
        print(f"Read speed: {read_speed:.2f} GB/s")
        ```
    '''

    write_speeds = []
    read_speeds = []

    for _ in range(ntimes):
        # Generate random data for the file
        data = os.urandom(file_size_bytes)
        
        # Write data to a temporary file
        with open('temp_file.bin', 'wb') as f:
            start_write = time.time()
            f.write(data)
            end_write = time.time()
        
        # Read data from the temporary file
        with open('temp_file.bin', 'rb') as f:
            start_read = time.time()
            f.read()
            end_read = time.time()
        
        # Calculate read and write speed (GB/s)
        write_speed = file_size_bytes / (end_write - start_write) / (1024**3)
        read_speed = file_size_bytes / (end_read - start_read) / (1024**3)

        write_speeds.append(write_speed)
        read_speeds.append(read_speed)
        
        # Clean up temporary file
        os.remove('temp_file.bin')

    avg_write_speed = np.mean(write_speeds)
    write_speed_std = np.std(write_speeds)
    avg_read_speed = np.mean(read_speeds)
    read_speed_std = np.std(read_speeds)
    
    return avg_write_speed, write_speed_std, avg_read_speed, read_speed_std


def disk_report(file_size=1024 * 1024 * 100, ntimes=10):
    '''
    Report the average write and read speed of the disk.

    Args:
        file_size (int): Size of the file to write in bytes

    Example:
        ```python
        import qim3d
        qim3d.io.logger.level("info")
        qim3d.utils.system.disk_report()
        ```
    '''

    # Test disk speed
    avg_write_speed, write_speed_std, avg_read_speed, read_speed_std = _test_disk_speed(file_size_bytes=file_size, ntimes=ntimes)
    
    # Print disk information
    log.info(
        "Disk:\n • Write speed..: %.2f GB/s (± %.2f GB/s)\n • Read speed...: %.2f GB/s (± %.2f GB/s)",
        avg_write_speed,
        write_speed_std,
        avg_read_speed,
        read_speed_std
    )
