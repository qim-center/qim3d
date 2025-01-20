"""
Provides functionality for loading data from various file formats.


Example:
    ```
    import qim3d
    data = qim3d.io.load("image.tif")
    ```

"""

import difflib
import os
import re
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import tifffile
from dask import delayed
from PIL import Image, UnidentifiedImageError

import qim3d
from qim3d.utils import log
from qim3d.utils._misc import get_file_size, sizeof, stringify_path
from qim3d.utils import Memory
from qim3d.utils._progress_bar import FileLoadingProgressBar
import trimesh

from typing import Optional, Dict

dask.config.set(scheduler="processes") 


class DataLoader:
    """Utility class for loading data from different file formats.

    Attributes:
        virtual_stack (bool): Specifies whether virtual stack is enabled.
        dataset_name (str): Specifies the name of the dataset to be loaded
            (only relevant for HDF5 files)
        return_metadata (bool): Specifies if metadata is returned or not
            (only relevant for HDF5, TXRM/TXM/XRM and NIfTI files)
        contains (str): Specifies a part of the name that is common for the
            TIFF file stack to be loaded (only relevant for TIFF stacks)

    Methods:
        load_tiff(path): Load a TIFF file from the specified path.
        load_h5(path): Load an HDF5 file from the specified path.
        load_tiff_stack(path): Load a stack of TIFF files from the specified path.
        load_txrm(path): Load a TXRM/TXM/XRM file from the specified path
        load_vol(path): Load a VOL file from the specified path. Path should point to the .vgi metadata file
        load(path): Load a file or directory based on the given path
    """

    def __init__(self, **kwargs):
        """Initializes a new instance of the DataLoader class.

        Args:
            virtual_stack (bool, optional): Specifies whether to use virtual
                stack when loading files. Default is False.
            dataset_name (str, optional): Specifies the name of the dataset to be loaded
                in case multiple dataset exist within the same file. Default is None (only for HDF5 files)
            return_metadata (bool, optional): Specifies whether to return metadata or not. Default is False (only for HDF5, TXRM/TXM/XRM and NIfTI files)
            contains (str, optional): Specifies a part of the name that is common for the TIFF file stack to be loaded (only for TIFF stacks).
                Default is None.
            force_load (bool, optional): If false and user tries to load file that exceeds available memory, throws a MemoryError. If true, this error is
                changed to warning and dataloader tries to load the file. Default is False.
            dim_order (tuple, optional): The order of the dimensions in the volume. Default is (2,1,0) which corresponds to (z,y,x)
        """
        self.virtual_stack = kwargs.get("virtual_stack", False)
        self.dataset_name = kwargs.get("dataset_name", None)
        self.return_metadata = kwargs.get("return_metadata", False)
        self.contains = kwargs.get("contains", None)
        self.force_load = kwargs.get("force_load", False)
        self.dim_order = kwargs.get("dim_order", (2, 1, 0))
        self.PIL_extensions = (".jp2", ".jpg", "jpeg", ".png", "gif", ".bmp", ".webp")

    def load_tiff(self, path: str|os.PathLike):
        """Load a TIFF file from the specified path.

        Args:
            path (str): The path to the TIFF file.

        Returns:
            numpy.ndarray or numpy.memmap: The loaded volume.
                If 'self.virtual_stack' is True, returns a numpy.memmap object.

        """
        # Get the number of TIFF series (some BigTIFF have multiple series)
        with tifffile.TiffFile(path) as tif:
            series = len(tif.series)

        if self.virtual_stack:
            vol = tifffile.memmap(path)
        else:
            vol = tifffile.imread(path, key=range(series) if series > 1 else None)

        log.info("Loaded shape: %s", vol.shape)

        return vol

    def load_h5(self, path: str|os.PathLike) -> tuple[np.ndarray, Optional[Dict]]:
        """Load an HDF5 file from the specified path.

        Args:
            path (str): The path to the HDF5 file.

        Returns:
            numpy.ndarray, h5py._hl.dataset.Dataset or tuple: The loaded volume.
                If 'self.virtual_stack' is True, returns a h5py._hl.dataset.Dataset object
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).

        Raises:
            ValueError: If the specified dataset_name is not found or is invalid.
            ValueError: If the dataset_name is not specified in case of multiple datasets in the HDF5 file
            ValueError: If no datasets are found in the file.
        """
        import h5py

        # Read file
        f = h5py.File(path, "r")
        data_keys = _get_h5_dataset_keys(f)
        datasets = []
        metadata = {}
        for key in data_keys:
            if (
                f[key].ndim > 1
            ):  # Data is assumed to be a dataset if it is two dimensions or more
                datasets.append(key)
            if f[key].attrs.keys():
                metadata[key] = {
                    "value": f[key][()],
                    **{attr_key: val for attr_key, val in f[key].attrs.items()},
                }

        # Only one dataset was found
        if len(datasets) == 1:
            if self.dataset_name:
                log.info(
                    "'dataset_name' argument is unused since there is only one dataset in the file"
                )
            name = datasets[0]
            vol = f[name]

        # Multiple datasets were found
        elif len(datasets) > 1:
            if self.dataset_name in datasets:  # Provided dataset name is valid
                name = self.dataset_name
                vol = f[name]
            else:
                if self.dataset_name:  # Dataset name is provided
                    similar_names = difflib.get_close_matches(
                        self.dataset_name, datasets
                    )  # Find closest matching name if any
                    if similar_names:
                        suggestion = similar_names[0]  # Get the closest match
                        raise ValueError(
                            f"Invalid dataset name. Did you mean '{suggestion}'?"
                        )
                    else:
                        raise ValueError(
                            f"Invalid dataset name. Please choose between the following datasets: {datasets}"
                        )
                else:
                    raise ValueError(
                        f"Found multiple datasets: {datasets}. Please specify which of them that you want to load with the argument 'dataset_name'"
                    )

        # No datasets were found
        else:
            raise ValueError(f"Did not find any data in the file: {path}")

        if not self.virtual_stack:
            vol = vol[()]  # Load dataset into memory
            f.close()

        log.info("Loaded the following dataset: %s", name)
        log.info("Loaded shape: %s", vol.shape)

        if self.return_metadata:
            return vol, metadata
        else:
            return vol

    def load_tiff_stack(self, path: str|os.PathLike) -> np.ndarray|np.memmap:
        """Load a stack of TIFF files from the specified path.

        Args:
            path (str): The path to the stack of TIFF files.

        Returns:
            numpy.ndarray or numpy.memmap: The loaded volume.
                If 'self.virtual_stack' is True, returns a numpy.memmap object.

        Raises:
            ValueError: If the 'contains' argument is not specified.
            ValueError: If the 'contains' argument matches multiple TIFF stacks in the directory
        """
        if not self.contains:
            raise ValueError(
                "Please specify a part of the name that is common for the TIFF file stack with the argument 'contains'"
            )

        tiff_stack = [
            file
            for file in os.listdir(path)
            if (file.endswith(".tif") or file.endswith(".tiff"))
            and self.contains in file
        ]
        tiff_stack.sort()  # Ensure proper ordering

        # Check that only one TIFF stack in the directory contains the provided string in its name
        tiff_stack_only_letters = []
        for filename in tiff_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            tiff_stack_only_letters.append(
                "".join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements from tiff_stack_only_letters
        unique_names = list(set(tiff_stack_only_letters))
        if len(unique_names) > 1:
            raise ValueError(
                f"The provided part of the filename for the TIFF stack matches multiple TIFF stacks: {unique_names}.\nPlease provide a string that is unique for the TIFF stack that is intended to be loaded"
            )

        vol = tifffile.imread(
            [os.path.join(path, file) for file in tiff_stack], out="memmap"
        )

        if not self.virtual_stack:
            vol = np.copy(vol)  # Copy to memory

        log.info("Found %s file(s)", len(tiff_stack))
        log.info("Loaded shape: %s", vol.shape)

        return vol

    def load_txrm(self, path: str|os.PathLike) -> tuple[dask.array.core.Array|np.ndarray, Optional[Dict]]:
        """Load a TXRM/XRM/TXM file from the specified path.

        Args:
            path (str): The path to the TXRM/TXM file.

        Returns:
            numpy.ndarray, dask.array.core.Array or tuple: The loaded volume.
                If 'virtual_stack' is True, returns a dask.array.core.Array object.
                If 'return_metadata' is True, returns a tuple (volume, metadata).

        Raises:
            ValueError: If the dxchange library is not installed
        """
        import olefile

        try:
            import dxchange
        except ImportError:
            raise ValueError(
                "The library dxchange is required to load TXRM files. Please find installation instructions at https://dxchange.readthedocs.io/en/latest/source/install.html"
            )

        if self.virtual_stack:
            if not path.endswith(".txm"):
                log.warning(
                    "Virtual stack is only thoroughly tested for reconstructed volumes in TXM format and is thus not guaranteed to load TXRM and XRM files correctly"
                )

            # Get metadata
            ole = olefile.OleFileIO(path)
            metadata = dxchange.reader.read_ole_metadata(ole)

            # Compute data offsets in bytes for each slice
            offsets = _get_ole_offsets(ole)

            if len(offsets) != metadata["number_of_images"]:
                raise ValueError(
                    f'Metadata is erroneous: number of images {metadata["number_of_images"]} is different from number of data offsets {len(offsets)}'
                )

            slices = []
            for _, offset in offsets.items():
                slices.append(
                    np.memmap(
                        path,
                        dtype=dxchange.reader._get_ole_data_type(metadata).newbyteorder(
                            "<"
                        ),
                        mode="r",
                        offset=offset,
                        shape=(1, metadata["image_height"], metadata["image_width"]),
                    )
                )

            vol = da.concatenate(slices, axis=0)
            log.warning(
                "Virtual stack volume will be returned as a dask array. To load certain slices into memory, use normal indexing followed by the compute() method, e.g. vol[:,0,:].compute()"
            )

        else:
            vol, metadata = dxchange.read_txrm(path)
            vol = (
                vol.squeeze()
            )  # In case of an XRM file, the third redundant dimension is removed

        if self.return_metadata:
            return vol, metadata
        else:
            return vol

    def load_nifti(self, path: str|os.PathLike):
        """Load a NIfTI file from the specified path.

        Args:
            path (str): The path to the NIfTI file.

        Returns:
            numpy.ndarray, nibabel.arrayproxy.ArrayProxy or tuple: The loaded volume.
                If 'self.virtual_stack' is True, returns a nibabel.arrayproxy.ArrayProxy object
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).
        """
        import nibabel as nib

        data = nib.load(path)

        # Get image array proxy
        vol = data.dataobj

        if not self.virtual_stack:
            vol = np.asarray(vol, dtype=data.get_data_dtype())

        if self.return_metadata:
            metadata = {}
            for key in data.header:
                metadata[key] = data.header[key]

            return vol, metadata
        else:
            return vol

    def load_pil(self, path: str|os.PathLike):
        """Load a PIL image from the specified path

        Args:
            path (str): The path to the image supported by PIL.

        Returns:
            numpy.ndarray: The loaded image/volume.
        """
        return np.array(Image.open(path))

    def load_PIL_stack(self, path: str|os.PathLike):
        """Load a stack of PIL files from the specified path.

        Args:
            path (str): The path to the stack of PIL files.

        Returns:
            numpy.ndarray or numpy.memmap: The loaded volume.
                If 'self.virtual_stack' is True, returns a numpy.memmap object.

        Raises:
            ValueError: If the 'contains' argument is not specified.
            ValueError: If the 'contains' argument matches multiple PIL stacks in the directory
        """
        if not self.contains:
            raise ValueError(
                "Please specify a part of the name that is common for the file stack with the argument 'contains'"
            )

        # List comprehension to filter files
        PIL_stack = [
            file
            for file in os.listdir(path)
            if file.endswith(self.PIL_extensions) and self.contains in file
        ]

        PIL_stack.sort()  # Ensure proper ordering

        # Check that only one stack in the directory contains the provided string in its name
        PIL_stack_only_letters = []
        for filename in PIL_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            PIL_stack_only_letters.append(
                "".join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements
        unique_names = list(set(PIL_stack_only_letters))
        if len(unique_names) > 1:
            raise ValueError(
                f"The provided part of the filename for the stack matches multiple stacks: {unique_names}.\nPlease provide a string that is unique for the image stack that is intended to be loaded"
            )

        if self.virtual_stack:
                
            full_paths = [os.path.join(path, file) for file in PIL_stack]

            def lazy_loader(path):
                with Image.open(path) as img:
                    return np.array(img)

            # Use delayed to load each image with PIL
            lazy_images = [delayed(lazy_loader)(path) for path in full_paths]
            # Compute the shape of the first image to define the array dimensions
            sample_image = np.array(Image.open(full_paths[0]))
            image_shape = sample_image.shape
            dtype = sample_image.dtype

            # Stack the images into a single Dask array
            dask_images = [
                da.from_delayed(img, shape=image_shape, dtype=dtype) for img in lazy_images
            ]
            stacked = da.stack(dask_images, axis=0)

            return stacked
        
        else:
            # Generate placeholder volume
            first_image = self.load_pil(os.path.join(path, PIL_stack[0]))
            vol = np.zeros((len(PIL_stack), *first_image.shape), dtype=first_image.dtype)

            # Load file sequence
            for idx, file_name in enumerate(PIL_stack):

                vol[idx] = self.load_pil(os.path.join(path, file_name))
            return vol
        
        

        # log.info("Found %s file(s)", len(PIL_stack))
        # log.info("Loaded shape: %s", vol.shape)

      

    def _load_vgi_metadata(self, path: str|os.PathLike):
        """Helper functions that loads metadata from a VGI file

        Args:
            path (str): The path to the VGI file.

        returns:
            dict: The loaded metadata.
        """
        meta_data = {}
        current_section = meta_data
        section_stack = [meta_data]

        should_indent = True

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                # {NAME} is start of a new object, so should indent
                if line.startswith("{") and line.endswith("}"):
                    section_name = line[1:-1]
                    current_section[section_name] = {}
                    section_stack.append(current_section)
                    current_section = current_section[section_name]

                    should_indent = True
                # [NAME] is start of a section, so should not indent
                elif line.startswith("[") and line.endswith("]"):
                    section_name = line[1:-1]

                    if not should_indent:
                        if len(section_stack) > 1:
                            current_section = section_stack.pop()

                    current_section[section_name] = {}
                    section_stack.append(current_section)
                    current_section = current_section[section_name]

                    should_indent = False
                # = is a key value pair
                elif "=" in line:
                    key, value = line.split("=", 1)
                    current_section[key.strip()] = value.strip()
                elif line == "":
                    if len(section_stack) > 1:
                        current_section = section_stack.pop()

        return meta_data

    def load_vol(self, path: str|os.PathLike):
        """Load a VOL filed based on the VGI metadata file

        Args:
            path (str): The path to the VGI file.

        Raises:
            ValueError: If path points to a .vol file and not a .vgi file

        returns:
            numpy.ndarray, numpy.memmap or tuple: The loaded volume.
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).
        """
        # makes sure path point to .VGI metadata file and not the .VOL file
        if path.endswith(".vol") and os.path.isfile(path.replace(".vol", ".vgi")):
            path = path.replace(".vol", ".vgi")
            log.warning("Corrected path to .vgi metadata file from .vol file")
        elif path.endswith(".vol") and not os.path.isfile(path.replace(".vol", ".vgi")):
            raise ValueError(
                f"Unsupported file format, should point to .vgi metadata file assumed to be in same folder as .vol file: {path}"
            )

        meta_data = self._load_vgi_metadata(path)

        # Extracts relevant information from the metadata
        file_name = meta_data["volume1"]["file1"]["Name"]
        path = path.rsplit("/", 1)[
            0
        ]  # Remove characters after the last "/" to be replaced with .vol filename
        vol_path = os.path.join(
            path, file_name
        )  # .vol and .vgi files are assumed to be in the same directory
        dims = meta_data["volume1"]["file1"]["Size"]
        dims = [int(n) for n in dims.split() if n.isdigit()]

        dt = meta_data["volume1"]["file1"]["Datatype"]
        match dt:
            case "float":
                dt = np.float32
            case "float32":
                dt = np.float32
            case "uint8":
                dt = np.uint8
            case "unsigned integer":
                dt = np.uint16
            case "uint16":
                dt = np.uint16
            case _:
                raise ValueError(f"Unsupported data type: {dt}")

        dims_order = (
            dims[self.dim_order[0]],
            dims[self.dim_order[1]],
            dims[self.dim_order[2]],
        )
        if self.virtual_stack:
            vol = np.memmap(vol_path, dtype=dt, mode="r", shape=dims_order)
        else:
            vol = np.fromfile(vol_path, dtype=dt, count=np.prod(dims))
            vol = np.reshape(vol, dims_order)

        if self.return_metadata:
            return vol, meta_data
        else:
            return vol

    def load_dicom(self, path: str|os.PathLike):
        """Load a DICOM file

        Args:
            path (str): Path to file
        """
        import pydicom

        dcm_data = pydicom.dcmread(path)

        if self.return_metadata:
            return dcm_data.pixel_array, dcm_data
        else:
            return dcm_data.pixel_array

    def load_dicom_dir(self, path: str|os.PathLike):
        """Load a directory of DICOM files into a numpy 3d array

        Args:
            path (str): Directory path
        
        returns:
            numpy.ndarray, numpy.memmap or tuple: The loaded volume.
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).
        """
        import pydicom

        if not self.contains:
            raise ValueError(
                "Please specify a part of the name that is common for the DICOM file stack with the argument 'contains'"
            )

        dicom_stack = [file for file in os.listdir(path) if self.contains in file]
        dicom_stack.sort()  # Ensure proper ordering

        # Check that only one DICOM stack in the directory contains the provided string in its name
        dicom_stack_only_letters = []
        for filename in dicom_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            dicom_stack_only_letters.append(
                "".join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements from tiff_stack_only_letters
        unique_names = list(set(dicom_stack_only_letters))
        if len(unique_names) > 1:
            raise ValueError(
                f"The provided part of the filename for the DICOM stack matches multiple DICOM stacks: {unique_names}.\nPlease provide a string that is unique for the DICOM stack that is intended to be loaded"
            )

        # dicom_list contains the dicom objects with metadata
        dicom_list = [pydicom.dcmread(os.path.join(path, f)) for f in dicom_stack]
        # vol contains the pixel data
        vol = np.stack([dicom.pixel_array for dicom in dicom_list], axis=0)

        if self.return_metadata:
            return vol, dicom_list
        else:
            return vol
        

    def load_zarr(self, path: str|os.PathLike):
        """ Loads a Zarr array from disk.

        Args:
            path (str): The path to the Zarr array on disk.

        Returns:
            dask.array | numpy.ndarray: The dask array loaded from disk.
                if 'self.virtual_stack' is True, returns a dask array object, else returns a numpy.ndarray object.
        """

        # Opens the Zarr array
        vol = da.from_zarr(path)

        # If virtual stack is disabled, return the computed array (np.ndarray)
        if not self.virtual_stack:
            vol = vol.compute()

        return vol

    def check_file_size(self, filename: str):
        """
        Checks if there is enough memory where the file can be loaded.
        Args:
        ------------
            filename: (str) Specifies path to file
            force_load: (bool, optional) If true, the memory error will not be raised. Warning will be printed insted and
                the loader will attempt to load the file.

        Raises:
        -----------
            MemoryError: If filesize is greater then available memory
        """

        if (
            self.virtual_stack
        ):  # If virtual_stack is True, then data is loaded from the disk, no need for loading into memory
            return
        file_size = get_file_size(filename)
        available_memory = Memory().free
        if file_size > available_memory:
            message = f"The file {filename} has {sizeof(file_size)} but only {sizeof(available_memory)} of memory is available."
            if self.force_load:
                log.warning(message)
            else:
                raise MemoryError(
                    message + " Set 'force_load=True' to ignore this error."
                )

    def load(self, path: str|os.PathLike):
        """
        Load a file or directory based on the given path.

        Args:
            path (str or os.PathLike): The path to the file or directory.

        Returns:
            vol (numpy.ndarray, numpy.memmap, h5py._hl.dataset.Dataset, nibabel.arrayproxy.ArrayProxy or tuple): The loaded volume

                If `virtual_stack=True`, returns `numpy.memmap`, `h5py._hl.dataset.Dataset` or `nibabel.arrayproxy.ArrayProxy` depending on file format
                If `return_metadata=True` and file format is either HDF5, NIfTI or TXRM/TXM/XRM, returns `tuple` (volume, metadata).

        Raises:
            ValueError: If the format is not supported
            ValueError: If the file or directory does not exist.
            MemoryError: If file size exceeds available memory and force_load is not set to True. In check_size function.
        """

        # Stringify path in case it is not already a string
        path = stringify_path(path)

        # Load a file
        if os.path.isfile(path):
            # Choose the loader based on the file extension
            self.check_file_size(path)
            if path.endswith(".tif") or path.endswith(".tiff"):
                return self.load_tiff(path)
            elif path.endswith(".h5"):
                return self.load_h5(path)
            elif path.endswith((".txrm", ".txm", ".xrm")):
                return self.load_txrm(path)
            elif path.endswith((".nii", ".nii.gz")):
                return self.load_nifti(path)
            elif path.endswith((".vol", ".vgi")):
                return self.load_vol(path)
            elif path.endswith((".dcm", ".DCM")):
                return self.load_dicom(path)
            else:
                try:
                    return self.load_pil(path)
                except UnidentifiedImageError:
                    raise ValueError("Unsupported file format")

        # Load a directory
        elif os.path.isdir(path):
            # load tiff stack if folder contains tiff files else load dicom directory
            if any(
                [f.endswith(".tif") or f.endswith(".tiff") for f in os.listdir(path)]
            ):
                return self.load_tiff_stack(path)

            elif any([f.endswith(self.PIL_extensions) for f in os.listdir(path)]):
                return self.load_PIL_stack(path)
            elif path.endswith(".zarr"):
                return self.load_zarr(path)
            else:
                return self.load_dicom_dir(path)

        # Fails
        else:
            # Find the closest matching path to warn the user
            similar_paths = qim3d.utils._misc.find_similar_paths(path)

            if similar_paths:
                suggestion = similar_paths[0]  # Get the closest match
                message = f"Invalid path. Did you mean '{suggestion}'?"
                raise ValueError(repr(message))
            else:
                raise ValueError("Invalid path")


def _get_h5_dataset_keys(f):
    import h5py

    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def _get_ole_offsets(ole):
    slice_offset = {}
    for stream in ole.listdir():
        if stream[0].startswith("ImageData"):
            sid = ole._find(stream)
            direntry = ole.direntries[sid]
            sect_start = direntry.isectStart
            offset = ole.sectorsize * (sect_start + 1)
            slice_offset[f"{stream[0]}/{stream[1]}"] = offset

    # sort dictionary after natural sorting (https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/)
    sorted_keys = sorted(
        slice_offset.keys(),
        key=lambda string_: [
            int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)
        ],
    )
    slice_offset_sorted = {key: slice_offset[key] for key in sorted_keys}

    return slice_offset_sorted


def load(
    path: str|os.PathLike,
    virtual_stack: bool = False,
    dataset_name: bool = None,
    return_metadata: bool = False,
    contains: bool = None,
    progress_bar: bool = True,
    force_load: bool = False,
    dim_order: tuple = (2, 1, 0),
    **kwargs,
) -> np.ndarray:
    """
    Load data from the specified file or directory.

    Supported formats:

    - `Tiff` (including file stacks)
    - `HDF5`
    - `TXRM`/`TXM`/`XRM`
    - `NIfTI`
    - `PIL` (including file stacks)
    - `VOL`/`VGI`
    - `DICOM`

    Args:
        path (str or os.PathLike): The path to the file or directory.
        virtual_stack (bool, optional): Specifies whether to use virtual
            stack when loading files. Default is False.
        dataset_name (str, optional): Specifies the name of the dataset to be loaded
            in case multiple dataset exist within the same file. Default is None (only for HDF5 files)
        return_metadata (bool, optional): Specifies whether to return metadata or not. Default is False (only for HDF5 and TXRM/TXM/XRM files)
        contains (str, optional): Specifies a part of the name that is common for the TIFF file stack to be loaded (only for TIFF stacks).
            Default is None.
        progress_bar (bool, optional): Displays tqdm progress bar. Useful for large files. So far works only for linux. Default is False.
        force_load (bool, optional): If the file size exceeds available memory, a MemoryError is raised.
            If force_load is True, the error is changed to warning and the loader tries to load it anyway. Default is False.
        dim_order (tuple, optional): The order of the dimensions in the volume for .vol files. Default is (2,1,0) which corresponds to (z,y,x)
        **kwargs (Any): Additional keyword arguments supported by `DataLoader`:
            - `virtual_stack` (bool)
            - `dataset_name` (str)
            - `return_metadata` (bool)
            - `contains` (str)
            - `force_load` (bool)
            - `dim_order` (tuple)

    Returns:
        vol (numpy.ndarray, numpy.memmap, h5py._hl.dataset.Dataset, nibabel.arrayproxy.ArrayProxy or tuple): The loaded volume

            If `virtual_stack=True`, returns `numpy.memmap`, `h5py._hl.dataset.Dataset` or `nibabel.arrayproxy.ArrayProxy` depending on file format
            If `return_metadata=True` and file format is either HDF5, NIfTI or TXRM/TXM/XRM, returns `tuple` (volume, metadata).

    Raises:
        MemoryError: if the given file size exceeds available memory

    Example:
        ```python
        import qim3d

        vol = qim3d.io.load("path/to/image.tif", virtual_stack=True)
        ```
    """

    loader = DataLoader(
        virtual_stack=virtual_stack,
        dataset_name=dataset_name,
        return_metadata=return_metadata,
        contains=contains,
        force_load=force_load,
        dim_order=dim_order,
        **kwargs,
    )

    if progress_bar and os.name == 'posix':
        with FileLoadingProgressBar(path):
            data = loader.load(path)
    else:
        data = loader.load(path)

    def log_memory_info(data):
        mem = Memory()
        log.info(
            "Volume using %s of memory\n",
            sizeof(data[0].nbytes if isinstance(data, tuple) else data.nbytes),
        )
        mem.report()

    if return_metadata and not isinstance(data, tuple):
        log.warning("The file format does not contain metadata")

    if not virtual_stack:
        log_memory_info(data)
    else:
        # Only log if file type is not a np.ndarray, i.e., it is some kind of memmap object
        if not isinstance(
            type(data[0]) if isinstance(data, tuple) else type(data), np.ndarray
        ):
            log.info("Using virtual stack")
        else:
            log.warning("Virtual stack is not supported for this file format")
            log_memory_info(data)

    return data

def load_mesh(filename: str) -> trimesh.Trimesh:
    """
    Load a mesh from an .obj file using trimesh.

    Args:
        filename (str or os.PathLike): The path to the .obj file.

    Returns:
        mesh (trimesh.Trimesh): A trimesh object containing the mesh data (vertices and faces).

    Example:
        ```python
        import qim3d

        mesh = qim3d.io.load_mesh("path/to/mesh.obj")
        ```
    """
    mesh = trimesh.load(filename)
    return mesh
