from threading import Timer
import psutil
import sys

from tqdm.auto import tqdm

from qim3d.utils.internal_tools import get_file_size


class RepeatTimer(Timer):
    """
    If the memory check is set as a normal thread, there is no garuantee it will switch
        resulting in not enough memory checks to create smooth progress bar or to make it
        work at all.
    Thus we have to use timer, which runs the function at (approximately) the given time. With this subclass
    from https://stackoverflow.com/a/48741004/11514359
    we don't have to guess how many timers we will need and create multiple timers. 
    """
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class ProgressBar:
    def __init__(self, filename:str, repeat_time:float = 0.5, *args, **kwargs):
        """
        Creates class for 'with' statement to track progress during loading a file into memory

        Parameters:
        ------------
        - filename (str): to get size of the file
        - repeat_time (float, optional): How often the timer checks how many bytes were loaded. Even if very small, 
            it doesn't make the progress bar smoother as there are only few visible changes in number of read_chars.
            Defaults to 0.25
        """
        self.timer = RepeatTimer(repeat_time, self.memory_check)
        self.pbar = tqdm(total = get_file_size(filename), 
                         desc = "Loading: ", 
                         unit = "B", 
                         file = sys.stdout,
                         unit_scale = True, 
                         unit_divisor = 1024,
                         bar_format = '{l_bar}{bar}| {n_fmt}{unit}/{total_fmt}{unit}  [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]')
        self.last_memory = 0
        self.process = psutil.Process()
    
    def memory_check(self):
        counters = self.process.io_counters()
        try:
            memory = counters.read_chars
        except AttributeError:
            memory = counters.read_bytes + counters.other_bytes


        try:
            self.pbar.update(memory - self.last_memory)
        except AttributeError: # When we leave the context manager, we delete the pbar so it can not be updated anymore
                                # It's because it takes quite a long time for the timer to end and might update the pbar
                                # one more time before ending which messes up the whole thing
            pass


        self.last_memory = memory

    def __enter__(self):
        self.timer.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.timer.cancel()
        self.pbar.clear()
        self.pbar.n = self.pbar.total
        self.pbar.display()
        del self.pbar # So the update process can not update it anymore

