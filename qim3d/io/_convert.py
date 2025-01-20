import difflib
import os
from itertools import product

import nibabel as nib
import numpy as np
import tifffile as tiff
import zarr
from tqdm import tqdm
import zarr.core

from qim3d.utils._misc import stringify_path
from qim3d.io import save


class Convert:
    def __init__(self, **kwargs):
        """Utility class to convert files to other formats without loading the entire file into memory

        Args:
            chunk_shape (tuple, optional): chunk size for the zarr file. Defaults to (64, 64, 64).
        """
        self.chunk_shape = kwargs.get("chunk_shape", (64, 64, 64))

    def convert(self, input_path: str, output_path: str):
        def get_file_extension(file_path):
            root, ext = os.path.splitext(file_path)
            if ext in ['.gz', '.bz2', '.xz']:  # handle common compressed extensions
                root, ext2 = os.path.splitext(root)
                ext = ext2 + ext
            return ext
        # Stringify path in case it is not already a string
        input_path = stringify_path(input_path)
        input_ext = get_file_extension(input_path)
        output_ext = get_file_extension(output_path)
        output_path = stringify_path(output_path)

        if os.path.isfile(input_path):
            match input_ext, output_ext:
                case (".tif", ".zarr") | (".tiff", ".zarr"):
                    return self.convert_tif_to_zarr(input_path, output_path)
                case (".nii", ".zarr") | (".nii.gz", ".zarr"):
                    return self.convert_nifti_to_zarr(input_path, output_path)
                case _:
                    raise ValueError("Unsupported file format")
        # Load a directory
        elif os.path.isdir(input_path):
            match input_ext, output_ext:
                case (".zarr", ".tif") | (".zarr", ".tiff"):
                    return self.convert_zarr_to_tif(input_path, output_path)
                case (".zarr", ".nii"):
                    return self.convert_zarr_to_nifti(input_path, output_path)
                case (".zarr", ".nii.gz"):
                    return self.convert_zarr_to_nifti(input_path, output_path, compression=True)
                case _:
                    raise ValueError("Unsupported file format")
        # Fail
        else:
            # Find the closest matching path to warn the user
            parent_dir = os.path.dirname(input_path) or "."
            parent_files = os.listdir(parent_dir) if os.path.isdir(parent_dir) else ""
            valid_paths = [os.path.join(parent_dir, file) for file in parent_files]
            similar_paths = difflib.get_close_matches(input_path, valid_paths)
            if similar_paths:
                suggestion = similar_paths[0]  # Get the closest match
                message = f"Invalid path. Did you mean '{suggestion}'?"
                raise ValueError(repr(message))
            else:
                raise ValueError("Invalid path")

    def convert_tif_to_zarr(self, tif_path: str, zarr_path: str) -> zarr.core.Array:
        """Convert a tiff file to a zarr file

        Args:
            tif_path (str): path to the tiff file
            zarr_path (str): path to the zarr file

        Returns:
            zarr.core.Array: zarr array containing the data from the tiff file
        """
        vol = tiff.memmap(tif_path)
        z = zarr.open(
            zarr_path, mode="w", shape=vol.shape, chunks=self.chunk_shape, dtype=vol.dtype
        )
        chunk_shape = tuple((s + c - 1) // c for s, c in zip(z.shape, z.chunks))
        # ! Fastest way is z[:] = vol[:], but does not have a progress bar
        for chunk_indices in tqdm(
            product(*[range(n) for n in chunk_shape]), total=np.prod(chunk_shape)
        ):
            slices = tuple(
                slice(c * i, min(c * (i + 1), s))
                for s, c, i in zip(z.shape, z.chunks, chunk_indices)
            )
            temp_data = vol[slices]
            # The assignment takes 99% of the cpu-time
            z.blocks[chunk_indices] = temp_data

        return z

    def convert_zarr_to_tif(self, zarr_path: str, tif_path: str) -> None:
        """Convert a zarr file to a tiff file

        Args:
            zarr_path (str): path to the zarr file
            tif_path (str): path to the tiff file

        returns:
            None
        """
        z = zarr.open(zarr_path)
        save(tif_path, z)

    def convert_nifti_to_zarr(self, nifti_path: str, zarr_path: str) -> zarr.core.Array:
        """Convert a nifti file to a zarr file

        Args:
            nifti_path (str): path to the nifti file
            zarr_path (str): path to the zarr file

        Returns:
            zarr.core.Array: zarr array containing the data from the nifti file
        """
        vol = nib.load(nifti_path).dataobj
        z = zarr.open(
            zarr_path, mode="w", shape=vol.shape, chunks=self.chunk_shape, dtype=vol.dtype
        )
        chunk_shape = tuple((s + c - 1) // c for s, c in zip(z.shape, z.chunks))
        # ! Fastest way is z[:] = vol[:], but does not have a progress bar
        for chunk_indices in tqdm(
            product(*[range(n) for n in chunk_shape]), total=np.prod(chunk_shape)
        ):
            slices = tuple(
                slice(c * i, min(c * (i + 1), s))
                for s, c, i in zip(z.shape, z.chunks, chunk_indices)
            )
            temp_data = vol[slices]
            # The assignment takes 99% of the cpu-time
            z.blocks[chunk_indices] = temp_data

        return z

    def convert_zarr_to_nifti(self, zarr_path: str, nifti_path: str, compression: bool = False) -> None:
        """Convert a zarr file to a nifti file

        Args:
            zarr_path (str): path to the zarr file
            nifti_path (str): path to the nifti file

        Returns:
            None
        """
        z = zarr.open(zarr_path)
        save(nifti_path, z, compression=compression)
        

def convert(input_path: str, output_path: str, chunk_shape: tuple = (64, 64, 64)) -> None:
    """Convert a file to another format without loading the entire file into memory

    Args:
        input_path (str): path to the input file
        output_path (str): path to the output file
        chunk_shape (tuple, optional): chunk size for the zarr file. Defaults to (64, 64, 64).
    """
    converter = Convert(chunk_shape=chunk_shape)
    converter.convert(input_path, output_path)
