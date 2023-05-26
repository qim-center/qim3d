"""Provides functionality for loading data from various file formats."""

import os
import sys
import difflib
import tifffile
import h5py
from qim3d.io.logger import log
from qim3d.tools.internal_tools import sizeof


class DataLoader:
    """Utility class for loading data from different file formats.

    Args:
        virtual_stack (bool, optional): Specifies whether to use virtual stack
        when loading TIFF files. Default is False.

    Attributes:
        virtual_stack (bool): Specifies whether virtual stack is enabled.

    Methods:
        load_tiff(path): Load a TIFF file from the specified path.
        load_h5(path): Load an HDF5 file from the specified path.
        load(path): Load a file or directory based on the given path.

    Raises:
        ValueError: If the file format is not supported or the path is invalid.

    Example:
        loader = DataLoader(virtual_stack=True)
        data = loader.load_tiff("image.tif")
    """

    def __init__(self, **kwargs):
        """Initializes a new instance of the DataLoader class.

        Args:
            virtual_stack (bool, optional): Specifies whether to use virtual
            stack when loading TIFF files. Default is False.
        """
        # Virtual stack is False by default
        self.virtual_stack = kwargs.get("virtual_stack", False)

    def load_tiff(self, path):
        """Load a TIFF file from the specified path.

        Args:
            path (str): The path to the TIFF file.

        Returns:
            numpy.ndarray: The loaded volume as a NumPy array.

        Raises:
            FileNotFoundError: If the file does not exist.

        Example:
            loader = DataLoader()
            data = loader.load_tiff("image.tif")
        """
        if self.virtual_stack:
            log.info("Using virtual stack")
            vol = tifffile.memmap(path)
        else:
            vol = tifffile.imread(path)

        log.info("Loaded shape: %s", vol.shape)
        log.info("Using %s of memory", sizeof(sys.getsizeof(vol)))

        return vol

    def load_h5(self, path):
        """Load an HDF5 file from the specified path.

        Args:
            path (str): The path to the HDF5 file.

        Returns:
            numpy.ndarray: The loaded volume as a NumPy array.

        Raises:
            FileNotFoundError: If the file does not exist.

        Example:
            loader = DataLoader()
            data = loader.load_h5("data.h5")
        """
        with h5py.File(path, "r") as f:
            vol = f["data"][:]
        return vol

    def load(self, path):
        """
        Load a file or directory based on the given path.

        Args:
            path (str): The path to the file or directory.

        Returns:
            numpy.ndarray: The loaded volume as a NumPy array.

        Raises:
            ValueError: If the format is not supported or the path is invalid.
            FileNotFoundError: If the file or directory does not exist.

        Example:
            loader = DataLoader()
            data = loader.load("image.tif")
        """
        # Load a file
        if os.path.isfile(path):
            # Choose the loader based on the file extension
            if path.endswith(".tif") or path.endswith(".tiff"):
                return self.load_tiff(path)
            elif path.endswith(".h5"):
                return self.load_h5(path)
            else:
                raise ValueError("Unsupported file format")

        # Load a directory
        elif os.path.isdir(path):
            raise NotImplementedError("Loading from directory is not implemented yet")

        # Fails
        else:
            # Find the closest matching path to warn the user
            parent_dir = os.path.dirname(path)
            parent_files = os.listdir(parent_dir)
            valid_paths = [os.path.join(parent_dir, file) for file in parent_files]
            similar_paths = difflib.get_close_matches(path, valid_paths)
            if similar_paths:
                suggestion = similar_paths[0]  # Get the closest match
                message = f"Invalid path.\nDid you mean '{suggestion}'?"
                raise ValueError(message)
            else:
                raise ValueError("Invalid path")


def load(path, virtual_stack=False, **kwargs):
    """
    Load data from the specified file or directory.

    Args:
        path (str): The path to the file or directory.
        virtual_stack (bool, optional): Specifies whether to use virtual
        stack when loading TIFF files. Default is False.

        **kwargs: Additional keyword arguments to be passed
        to the DataLoader constructor.

    Returns:
        numpy.ndarray: The loaded volume as a NumPy array.

    Raises:
        ValueError: If the file format is not supported or the path is invalid.
        NotImplementedError: If loading from a directory is not implemented yet.
        FileNotFoundError: If the file or directory does not exist.

    Example:
        data = load("image.tif", virtual_stack=True)
    """
    loader = DataLoader(virtual_stack=virtual_stack, **kwargs)

    return loader.load(path)
