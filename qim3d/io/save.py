"""Provides functionality for saving data to various file formats."""

import os
import tifffile
import numpy as np
from qim3d.io.logger import log

class DataSaver:
    """Utility class for saving data to different file formats.

    Attributes:
        replace (bool): Specifies if an existing file with identical path is replaced.
        compression (bool): Specifies if the file is with Deflate compression.

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
        """
        self.replace = kwargs.get("replace",False)
        self.compression = kwargs.get("compression",False)

    def save_tiff(self,path,data):
        """Save data to a TIFF file to the given path.

        Args: 
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved    
        """
        tifffile.imwrite(path,data,compression=self.compression)

    def save(self, path, data):
        """Save data to the given path.

        Args: 
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        
        Raises:
            ValueError: If the file format is not supported.
            ValueError: If the specified folder does not exist.
            ValueError: If a file extension is not provided.
            ValueError: if a file with the specified path already exists and replace=False.

        Example:
            image = qim3d.examples.blobs_256x256
            saver = qim3d.io.DataSaver(compression=True)
            saver.save("image.tif",image)
        """

        folder = os.path.dirname(path) or '.'
        # Check if directory exists
        if os.path.isdir(folder):
            _, ext = os.path.splitext(path)
            # Check if provided path contains file extension
            if ext:
                # Check if a file with the given path already exists
                if os.path.isfile(path) and not self.replace:
                    raise ValueError("A file with the provided path already exists. To replace it set 'replace=True'")
                
                if path.endswith((".tif",".tiff")):
                    return self.save_tiff(path,data)
                else:
                    raise ValueError("Unsupported file format")
            else:
                raise ValueError('Please provide a file extension')
        else:
            raise ValueError(f'The directory {folder} does not exist. Please provide a valid directory')
        
def save(path,
        data,
        replace=False,
        compression=False,
        **kwargs
        ):
    """Save data to a specified file path.

    Args:
        path (str): The path to save file to
        data (numpy.ndarray): The data to be saved  
        replace (bool, optional): Specifies if an existing file with identical path should be replaced. 
            Default is False.
        compression (bool, optional): Specifies if the file should be saved with Deflate compression (lossless).
            Default is False.
        **kwargs: Additional keyword arguments to be passed to the DataSaver constructor
    
    Example:
        image = qim3d.examples.blobs_256x256
        qim3d.io.save("image.tif",image,compression=True)
    """
        
    DataSaver(replace=replace, compression=compression, **kwargs).save(path, data)