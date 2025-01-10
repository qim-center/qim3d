from threading import Timer
import psutil
import sys
import os
from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from qim3d.utils._misc import get_file_size


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

class ProgressBar(ABC):
    def __init__(self, tqdm_kwargs: dict, repeat_time: float,  *args, **kwargs):
        """
        Context manager for ('with' statement) to track progress during a long progress over 
        which we don't have control (like loading a file) and thus can not insert the tqdm
        updates into loop
        Thus we have to run parallel thread with forced activation to check the state


        Parameters:
        ------------
        - tqdm_kwargs (dict): Passed directly to tqdm constructor
        - repeat_time (float): How often the timer runs the function (in seconds)
        """
        self.timer = RepeatTimer(repeat_time, self.update_pbar)
        self.pbar = tqdm(**tqdm_kwargs)
        self.last_update = 0

    def update_pbar(self):
        new_update = self.get_new_update()
        update = new_update - self.last_update

        try:
            self.pbar.update(update)
        except (
            AttributeError
        ):  # When we leave the context manager, we delete the pbar so it can not be updated anymore
            # It's because it takes quite a long time for the timer to end and might update the pbar
            # one more time before ending which messes up the whole thing
            pass

        self.last_update = new_update


    @abstractmethod
    def get_new_update(self):
        pass

    def __enter__(self):
        self.timer.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.timer.cancel()
        self.pbar.clear()
        self.pbar.n = self.pbar.total
        self.pbar.display()
        del self.pbar  # So the update process can not update it anymore



class FileLoadingProgressBar(ProgressBar):
    def __init__(self, filename: str, repeat_time: float = 0.5, *args, **kwargs):
        """
        Context manager ('with' statement) to track progress during loading a file into memory

        Parameters:
        ------------
        - filename (str): to get size of the file
        - repeat_time (float, optional): How often the timer checks how many bytes were loaded. Even if very small,
            it doesn't make the progress bar smoother as there are only few visible changes in number of read_chars.
            Defaults to 0.5
        """
        tqdm_kwargs = dict(
            total=get_file_size(filename),
            desc="Loading: ",
            unit="B",
            file=sys.stdout,
            unit_scale=True,
            unit_divisor=1024,
            bar_format="{l_bar}{bar}| {n_fmt}{unit}/{total_fmt}{unit}  [{elapsed}<{remaining}, "
            "{rate_fmt}{postfix}]",
        )
        super().__init__( tqdm_kwargs, repeat_time)
        self.process = psutil.Process()

    def get_new_update(self) -> int:
        counters = self.process.io_counters()
        try:
            memory = counters.read_chars
        except AttributeError:
            memory = counters.read_bytes + counters.other_bytes
        return memory

class OmeZarrExportProgressBar(ProgressBar):
    def __init__(self,path: str, n_chunks: int, reapeat_time: str = "auto"):
        """
        Context manager to track the exporting of OmeZarr files.

        Parameters
        ----------
        path : str
            The folder path where the files will be saved.
        n_chunks : int
            The total number of chunks to track.
        repeat_time : int or float, optional
            The interval (in seconds) for updating the progress bar. Defaults to "auto", which 
            sets the update frequency based on the number of chunks.
        """



        # Calculate the repeat time for the progress bar
        if reapeat_time == "auto":
            # Approximate the repeat time based on the number of chunks
            # This ratio is based on reading the HOA dataset over the network:
            # 620,000 files took 300 seconds to read
            # The ratio is little smaller than needed to avoid disk stress

            reapeat_time = n_chunks / 1500

        else:
            reapeat_time = float(reapeat_time)

        # We don't want to update the progress bar too often anyway
        if reapeat_time < 0.5:
            reapeat_time = 0.5

        self.path = path
        tqdm_kwargs = dict(
            total = n_chunks,
            unit = "Chunks",
            desc = "Saving",
            unit_scale = True

        )
        super().__init__(tqdm_kwargs, reapeat_time)
        self.last_update = 0

    def get_new_update(self):
        def file_count(folder_path: str) -> int:
            """
            Goes recursively through the folders and counts how many files are there, 
            Doesn't count metadata json files
            """
            count = 0
            for path in os.listdir(folder_path):
                new_path = os.path.join(folder_path, path)
                if os.path.isfile(new_path):
                    filename = os.path.basename(os.path.normpath(new_path))
                    if not filename.startswith("."):
                        count += 1
                else:
                    count += file_count(new_path)
            return count

        return file_count(self.path)
