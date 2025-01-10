""" Dataset synchronization tasks """
import os
import subprocess
import outputformat as ouf
from qim3d.utils import log
from pathlib import Path


class Sync:
    """Class for dataset synchronization tasks"""

    def __init__(self):
        # Checks if rsync is available
        if not self._check_rsync():
            raise RuntimeError(
                "Could not find rsync, please check if it is installed in your system."
            )

    def _check_rsync(self):
        """Check if rsync is available"""
        try:
            subprocess.run(["rsync", "--version"], capture_output=True, check=True)
            return True

        except Exception as error:
            log.error("rsync is not available")
            log.error(error)

            return False

    def check_destination(self, source: str, destination: str, checksum: bool = False, verbose: bool = True) -> list[str]:
        """Check if all files from 'source' are in 'destination'

        This function compares the files in the 'source' directory to those in
        the 'destination' directory and reports any differences or missing files.

        Args:
            source (str or Path): The source directory path.
            destination (str or Path): The destination directory path.
            checksum (bool, optional): If True, use checksums to compare files (slower but more accurate).
                Default is False.
            verbose (bool, optional): If True, display a list of differing or missing files in the log.
                Default is True.

        Returns:
            list: A list of differing or missing file paths in the destination directory.

        """

        source = Path(source)
        destination = Path(destination)

        if checksum:
            rsync_args = "-avrc"
        else:
            rsync_args = "-avr"

        command = [
            "rsync",
            "-n",
            rsync_args,
            str(source) + os.path.sep,
            str(destination) + os.path.sep,
        ]

        out = subprocess.run(
            command,
            capture_output=True,
            check=True,
        )

        diff_files_and_folders = out.stdout.decode().splitlines()[1:-3]
        diff_files = [f for f in diff_files_and_folders if not f.endswith("/")]

        if len(diff_files) > 0 and verbose:
            title = "Source files differing or missing in destination"
            log.info(
                ouf.showlist(diff_files, style="line", return_str=True, title=title)
            )

        return diff_files

    def compare_dirs(self, source: str, destination: str, checksum: bool = False, verbose: bool = True) -> None:
        """Checks whether 'source' and 'destination' directories are synchronized.

        This function compares the contents of two directories
        ('source' and 'destination') and reports any differences.
        It checks for files that exist in one directory but not the other and
        files that are present in both but not equal.

        If no differences are found between the directories,
        it logs a message indicating that they are synchronized.
        If differences are found, it logs detailed information about the differing files.

        Args:
            source (str or Path): The source directory path.
            destination (str or Path): The destination directory path.
            checksum (bool, optional): If True, use checksums to compare files (slower but more accurate).
                Default is False.
            verbose (bool, optional): If True, display information about the comparison in the log.
                Default is True.

        Returns:
            None: This function does not return a value.

        """
        if verbose:
            s_files, s_dirs = self.count_files_and_dirs(source)
            d_files, d_dirs = self.count_files_and_dirs(destination)
            log.info("\n")

        s_d = self.check_destination(
            source, destination, checksum=checksum, verbose=False
        )
        d_s = self.check_destination(
            destination, source, checksum=checksum, verbose=False
        )

        if len(s_d) == 0 and len(d_s) == 0:
            # No differences
            if verbose:
                log.info(
                    "Source and destination are synchronized, no differences found."
                )
            return

        union = list(set(s_d + d_s))
        log.info(
            ouf.showlist(
                union,
                style="line",
                return_str=True,
                title=f"{len(union)} files are not in sync",
            )
        )

        intersection = list(set(s_d) & set(d_s))
        if len(intersection) > 0:
            log.info(
                ouf.showlist(
                    intersection,
                    style="line",
                    return_str=True,
                    title=f"{len(intersection)} files present on both, but not equal",
                )
            )

        s_exclusive = list(set(s_d).symmetric_difference(set(intersection)))
        if len(s_exclusive) > 0:
            log.info(
                ouf.showlist(
                    s_exclusive,
                    style="line",
                    return_str=True,
                    title=f"{len(s_exclusive)} files present only on {source}",
                )
            )

        d_exclusive = list(set(d_s).symmetric_difference(set(intersection)))
        if len(d_exclusive) > 0:
            log.info(
                ouf.showlist(
                    d_exclusive,
                    style="line",
                    return_str=True,
                    title=f"{len(d_exclusive)} files present only on {destination}",
                )
            )
        return

    def count_files_and_dirs(self, path: str|os.PathLike, verbose: bool = True) -> tuple[int, int]:
        """Count the number of files and directories in the given path.

        This function recursively counts the number of files and
        directories in the specified directory 'path'.

        If 'verbose' is True, the function logs the total count
        of files and directories in the specified path.


        Args:
            path (str or Path): The directory path to count files and directories in.
            verbose (bool, optional): If True, display the total count in the log.
                Default is True.

        Returns:
            tuple: A tuple containing two values:
                - The count of files in the directory and its subdirectories.
                - The count of directories in the directory and its subdirectories.

        """
        path = Path(path)
        files = 0
        dirs = 0
        for p in os.scandir(path):
            if p.is_file():
                files += 1
            elif p.is_dir():
                dirs += 1
                file_count, dirs_count = self.count_files_and_dirs(p, verbose=False)
                files += file_count
                dirs += dirs_count

        if verbose:
            log.info(f"Total of {files} files and {dirs} directories on {path}")

        return files, dirs
