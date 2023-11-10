"""Provides functionality for saving data to various file formats."""

import os
import tifffile
import numpy as np
from qim3d.io.logger import log
from qim3d.utils.internal_tools import sizeof, stringify_path


class DataSaver:
    """Utility class for saving data to different file formats.

    Attributes:
        replace (bool): Specifies if an existing file with identical path is replaced.
        compression (bool): Specifies if the file is saved with Deflate compression (lossless).
        basename (str): Specifies the basename for a TIFF stack saved as several files
            (only relevant for TIFF stacks).
        sliced_dim (int): Specifies the dimension that is sliced in case a TIFF stack is saved
            as several files (only relevant for TIFF stacks)

    Methods:
        save_tiff(path,data): Save data to a TIFF file to the given path.
        load(path,data): Save data to the given path.

    Example:
        image = qim3d.examples.blobs_256x256
        saver = qim3d.io.DataSaver(compression=True)
        saver.save_tiff("image.tif",image)
    """

    def __init__(self, **kwargs):
        """Initializes a new instance of the DataSaver class.

        Args:
            replace (bool, optional): Specifies if an existing file with identical path should be replaced.
                Default is False.
            compression (bool, optional): Specifies if the file should be saved with Deflate compression.
                Default is False.
            basename (str, optional): Specifies the basename for a TIFF stack saved as several files
                (only relevant for TIFF stacks). Default is None
            sliced_dim (int, optional): Specifies the dimension that is sliced in case a TIFF stack is saved
                as several files (only relevant for TIFF stacks). Default is 0, i.e., the first dimension.
        """
        self.replace = kwargs.get("replace", False)
        self.compression = kwargs.get("compression", False)
        self.basename = kwargs.get("basename", None)
        self.sliced_dim = kwargs.get("sliced_dim", 0)

    def save_tiff(self, path, data):
        """Save data to a TIFF file to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        """
        tifffile.imwrite(path, data, compression=self.compression)

    def save_tiff_stack(self, path, data):
        """Save data as a TIFF stack containing slices in separate files to the given path.
        The slices will be named according to the basename plus a suffix with a zero-filled
        value corresponding to the slice number

        Args:
            path (str): The directory to save files to
            data (numpy.ndarray): The data to be saved
        """
        extension = ".tif"

        if data.ndim <= 2:
            path = os.path.join(path, self.basename, ".tif")
            self.save_tiff(path, data)
        else:
            # get number of total slices
            no_slices = data.shape[self.sliced_dim]
            # Get number of zero-fill values as the number of digits in the total number of slices
            zfill_val = len(str(no_slices))

            # Create index list
            idx = [slice(None)] * data.ndim

            # Iterate through each slice and save
            for i in range(no_slices):
                idx[self.sliced_dim] = i
                sliced = data[tuple(idx)]
                filename = self.basename + str(i).zfill(zfill_val) + extension
                filepath = os.path.join(path, filename)
                self.save_tiff(filepath, sliced)

            pattern_string = filepath[:-(len(extension)+zfill_val)] + "-"*zfill_val + extension

            log.info(f"Total of {no_slices} files saved following the pattern '{pattern_string}'")

    def save(self, path, data):
        """Save data to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved

        Raises:
            ValueError: If the provided path is an existing directory and self.basename is not provided
            ValueError: If the file format is not supported.
            ValueError: If the provided path does not exist and self.basename is not provided
            ValueError: If a file extension is not provided.
            ValueError: if a file with the specified path already exists and replace=False.

        Example:
            image = qim3d.examples.blobs_256x256
            saver = qim3d.io.DataSaver(compression=True)
            saver.save("image.tif",image)

        """

        path = stringify_path(path)
        isdir = os.path.isdir(path)
        _, ext = os.path.splitext(path)

        # If path is an existing directory
        if isdir:
            # If basename is provided
            if self.basename:
                # Save as tiff stack
                return self.save_tiff_stack(path, data)
            # If basename is not provided
            else:
                raise ValueError(
                    f"To save a stack as several TIFF files to the directory '{path}', please provide the keyword argument 'basename'. "
                    + "Otherwise, to save a single file, please provide a full path with a filename and valid extension."
                )

        # If path is not an existing directory
        else:
            # If there is no file extension in path and basename is provided
            if not ext and self.basename:
                # Make directory and save as tiff stack
                os.mkdir(path)
                log.info("Created directory '%s'!", path)
                return self.save_tiff_stack(path, data)

            # Check if a parent directory exists
            parentdir = os.path.dirname(path) or "."
            if os.path.isdir(parentdir):
                # If there is a file extension in the path
                if ext:
                    # If there is a basename
                    if self.basename:
                        # It will be unused and the user is informed accordingly
                        log.info("'basename' argument is unused")
                    # Check if a file with the given path already exists
                    if os.path.isfile(path) and not self.replace:
                        raise ValueError(
                            "A file with the provided path already exists. To replace it set 'replace=True'"
                        )

                    if path.endswith((".tif", ".tiff")):
                        return self.save_tiff(path, data)
                    else:
                        raise ValueError("Unsupported file format")
                # If there is no file extension in the path
                else:
                    raise ValueError(
                        "Please provide a file extension if you want to save as a single file."
                        + " Otherwise, please provide a basename to save as a TIFF stack"
                    )
            else:
                raise ValueError(
                    f"The directory '{parentdir}' does not exist.\n"
                    + "Please provide a valid directory or specify a basename if you want to save a tiff stack as several files to a folder that does not yet exist"
                )


def save(
    path, data, replace=False, compression=False, basename=None, sliced_dim=0, **kwargs
):
    """Save data to a specified file path.

    Args:
        path (str): The path to save file to
        data (numpy.ndarray): The data to be saved
        replace (bool, optional): Specifies if an existing file with identical path should be replaced.
            Default is False.
        compression (bool, optional): Specifies if the file should be saved with Deflate compression (lossless).
            Default is False.
        basename (str, optional): Specifies the basename for a TIFF stack saved as several files
            (only relevant for TIFF stacks). Default is None
        sliced_dim (int, optional): Specifies the dimension that is sliced in case a TIFF stack is saved
            as several files (only relevant for TIFF stacks). Default is 0, i.e., the first dimension.
        **kwargs: Additional keyword arguments to be passed to the DataSaver constructor

    Example:
        image = qim3d.examples.blobs_256x256
        qim3d.io.save("image.tif",image,compression=True)
    """

    DataSaver(
        replace=replace,
        compression=compression,
        basename=basename,
        sliced_dim=sliced_dim,
        **kwargs,
    ).save(path, data)
