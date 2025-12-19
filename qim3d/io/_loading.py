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

import dask
import dask.array as da
import numpy as np
import tifffile
from dask import delayed
from PIL import Image, UnidentifiedImageError
from pygel3d import hmesh
import zarr

import qim3d
from qim3d.utils import Memory, log
from qim3d.utils._misc import get_file_size, sizeof, stringify_path
from qim3d.utils._progress_bar import FileLoadingProgressBar
from qim3d.io._txrm import read_txrm, _get_ole_data_type, read_ole_metadata

dask.config.set(scheduler='processes')


class DataLoader:
    """
    Utility class for loading data from different file formats.

    Attributes
        virtual_stack (bool): Specifies whether virtual stack is enabled.
        dataset_name (str): Specifies the name of the dataset to be loaded
            (only relevant for HDF5 files)
        return_metadata (bool): Specifies if metadata is returned or not
            (only relevant for HDF5, TXRM/TXM/XRM and NIfTI files)
        contains (str): Specifies a part of the name that is common for the
            TIFF file stack to be loaded (only relevant for TIFF stacks)

    Methods
        load_tiff(path): Load a TIFF file from the specified path.
        load_h5(path): Load an HDF5 file from the specified path.
        load_tiff_stack(path): Load a stack of TIFF files from the specified path.
        load_txrm(path): Load a TXRM/TXM/XRM file from the specified path
        load_vol(path): Load a VOL file from the specified path. Path should point to the .vgi metadata file
        load(path): Load a file or directory based on the given path

    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the DataLoader class.

        Args:
            kwargs (Any):
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
        self.virtual_stack = kwargs.get('virtual_stack', False)
        self.dataset_name = kwargs.get('dataset_name', None)
        self.return_metadata = kwargs.get('return_metadata', False)
        self.contains = kwargs.get('contains', None)
        self.force_load = kwargs.get('force_load', False)
        self.dim_order = kwargs.get('dim_order', (2, 1, 0))
        self.PIL_extensions = ('.jp2', '.jpg', 'jpeg', '.png', 'gif', '.bmp', '.webp')

    def load_tiff(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a TIFF file from the specified path.

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

        return vol

    def load_h5(self, path: str | os.PathLike) -> tuple[np.ndarray, dict | None]:
        """
        Load an HDF5 file from the specified path.

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
        f = h5py.File(path, 'r')
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
                    'value': f[key][()],
                    **dict(f[key].attrs),  # ruff: **{attr_key: val for attr_key, val in f[key].attrs.items()},
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
                        msg = f"Invalid dataset name. Did you mean '{suggestion}'?"
                        raise ValueError(msg)
                    else:
                        msg = f'Invalid dataset name. Please choose between the following datasets: {datasets}'
                        raise ValueError(msg)
                else:
                    msg = f"Found multiple datasets: {datasets}. Please specify which of them that you want to load with the argument 'dataset_name'"
                    raise ValueError(msg)

        # No datasets were found
        else:
            msg = f'Did not find any data in the file: {path}'
            raise ValueError(msg)

        if not self.virtual_stack:
            vol = vol[()]  # Load dataset into memory
            f.close()

        if self.return_metadata:
            return vol, metadata
        else:
            return vol

    def load_tiff_stack(self, path: str | os.PathLike) -> np.ndarray | np.memmap:
        """
        Load a stack of TIFF files from the specified path.

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
            msg = "Please specify a part of the name that is common for the TIFF file stack with the argument 'contains'"
            raise ValueError(msg)

        tiff_stack = [
            file
            for file in os.listdir(path)
            if (file.endswith(('.tif', '.tiff'))) and self.contains in file
        ]
        tiff_stack.sort()  # Ensure proper ordering

        # Check that only one TIFF stack in the directory contains the provided string in its name
        tiff_stack_only_letters = []
        for filename in tiff_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            tiff_stack_only_letters.append(
                ''.join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements from tiff_stack_only_letters
        unique_names = list(set(tiff_stack_only_letters))
        if len(unique_names) > 1:
            msg = f'The provided part of the filename for the TIFF stack matches multiple TIFF stacks: {unique_names}.\nPlease provide a string that is unique for the TIFF stack that is intended to be loaded'
            raise ValueError(msg)

        vol = tifffile.imread(
            [os.path.join(path, file) for file in tiff_stack], out='memmap'
        )

        if not self.virtual_stack:
            vol = np.copy(vol)  # Copy to memory

        return vol

    def load_txrm(
        self, path: str | os.PathLike
    ) -> tuple[dask.array.core.Array | np.ndarray, dict | None]:
        """
        Load a TXRM/XRM/TXM file from the specified path.

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

        if self.virtual_stack:
            if not path.endswith('.txm'):
                log.warning(
                    'Virtual stack is only thoroughly tested for reconstructed volumes in TXM format and is thus not guaranteed to load TXRM and XRM files correctly'
                )

            # Get metadata
            ole = olefile.OleFileIO(path)
            metadata = read_ole_metadata(ole)

            # Compute data offsets in bytes for each slice
            offsets = _get_ole_offsets(ole)

            if len(offsets) != metadata['number_of_images']:
                msg = f'Metadata is erroneous: number of images {metadata["number_of_images"]} is different from number of data offsets {len(offsets)}'
                raise ValueError(msg)

            slices = []
            for _, offset in offsets.items():
                slices.append(
                    np.memmap(
                        path,
                        dtype=_get_ole_data_type(metadata).newbyteorder(
                            '<'
                        ),
                        mode='r',
                        offset=offset,
                        shape=(1, metadata['image_height'], metadata['image_width']),
                    )
                )

            vol = da.concatenate(slices, axis=0)
            log.warning(
                'Virtual stack volume will be returned as a dask array. To load certain slices into memory, use normal indexing followed by the compute() method, e.g. vol[:,0,:].compute()'
            )

        else:
            vol, metadata = read_txrm(path)
            vol = (
                vol.squeeze()
            )  # In case of an XRM file, the third redundant dimension is removed

        if self.return_metadata:
            return vol, metadata
        else:
            return vol

    def load_nifti(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a NIfTI file from the specified path.

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

    def load_pil(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a PIL image from the specified path.

        Args:
            path (str): The path to the image supported by PIL.

        Returns:
            numpy.ndarray: The loaded image/volume.

        """
        return np.array(Image.open(path))

    def load_pil_stack(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a stack of PIL files from the specified path.

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
            msg = "Please specify a part of the name that is common for the file stack with the argument 'contains'"
            raise ValueError(msg)

        # List comprehension to filter files
        pil_stack = [
            file
            for file in os.listdir(path)
            if file.endswith(self.PIL_extensions) and self.contains in file
        ]

        pil_stack.sort()  # Ensure proper ordering

        # Check that only one stack in the directory contains the provided string in its name
        pil_stack_only_letters = []
        for filename in pil_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            pil_stack_only_letters.append(
                ''.join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements
        unique_names = list(set(pil_stack_only_letters))
        if len(unique_names) > 1:
            msg = f'The provided part of the filename for the stack matches multiple stacks: {unique_names}.\nPlease provide a string that is unique for the image stack that is intended to be loaded'
            raise ValueError(msg)

        if self.virtual_stack:
            full_paths = [os.path.join(path, file) for file in pil_stack]

            def lazy_loader(path: str) -> np.ndarray:
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
                da.from_delayed(img, shape=image_shape, dtype=dtype)
                for img in lazy_images
            ]
            stacked = da.stack(dask_images, axis=0)

            return stacked

        else:
            # Generate placeholder volume
            first_image = self.load_pil(os.path.join(path, pil_stack[0]))
            vol = np.zeros(
                (len(pil_stack), *first_image.shape), dtype=first_image.dtype
            )

            # Load file sequence
            for idx, file_name in enumerate(pil_stack):
                vol[idx] = self.load_pil(os.path.join(path, file_name))
            return vol

    def _load_vgi_metadata(self, path: str | os.PathLike) -> dict:
        """
        Helper functions that loads metadata from a VGI file.

        Args:
            path (str): The path to the VGI file.

        returns:
            dict: The loaded metadata.

        """
        meta_data = {}
        current_section = meta_data
        section_stack = [meta_data]

        should_indent = True

        with open(path) as f:
            for line in f:
                line = line.strip()
                # {NAME} is start of a new object, so should indent
                if line.startswith('{') and line.endswith('}'):
                    section_name = line[1:-1]
                    current_section[section_name] = {}
                    section_stack.append(current_section)
                    current_section = current_section[section_name]

                    should_indent = True
                # [NAME] is start of a section, so should not indent
                elif line.startswith('[') and line.endswith(']'):
                    section_name = line[1:-1]

                    if not should_indent and len(section_stack) > 1:
                        current_section = section_stack.pop()

                    current_section[section_name] = {}
                    section_stack.append(current_section)
                    current_section = current_section[section_name]

                    should_indent = False
                # = is a key value pair
                elif '=' in line:
                    key, value = line.split('=', 1)
                    current_section[key.strip()] = value.strip()
                elif line == '':
                    if len(section_stack) > 1:
                        current_section = section_stack.pop()

        return meta_data

    def load_vol(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a VOL filed based on the VGI metadata file.

        Args:
            path (str): The path to the VGI file.

        Raises:
            ValueError: If path points to a .vol file and not a .vgi file

        returns:
            numpy.ndarray, numpy.memmap or tuple: The loaded volume.
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).

        """
        # makes sure path point to .VGI metadata file and not the .VOL file
        if path.endswith('.vol') and os.path.isfile(path.replace('.vol', '.vgi')):
            path = path.replace('.vol', '.vgi')
            log.warning('Corrected path to .vgi metadata file from .vol file')
        elif path.endswith('.vol') and not os.path.isfile(path.replace('.vol', '.vgi')):
            msg = f'Unsupported file format, should point to .vgi metadata file assumed to be in same folder as .vol file: {path}'
            raise ValueError(msg)

        meta_data = self._load_vgi_metadata(path)

        # Extracts relevant information from the metadata
        file_name = meta_data['volume1']['file1']['Name']
        path = path.rsplit('/', 1)[
            0
        ]  # Remove characters after the last "/" to be replaced with .vol filename
        vol_path = os.path.join(
            path, file_name
        )  # .vol and .vgi files are assumed to be in the same directory
        dims = meta_data['volume1']['file1']['Size']
        dims = [int(n) for n in dims.split() if n.isdigit()]

        dt = meta_data['volume1']['file1']['Datatype']
        match dt:
            case 'float':
                dt = np.float32
            case 'float32':
                dt = np.float32
            case 'uint8':
                dt = np.uint8
            case 'unsigned integer':
                dt = np.uint16
            case 'uint16':
                dt = np.uint16
            case _:
                msg = f'Unsupported data type: {dt}'
                raise ValueError(msg)

        dims_order = (
            dims[self.dim_order[0]],
            dims[self.dim_order[1]],
            dims[self.dim_order[2]],
        )
        if self.virtual_stack:
            vol = np.memmap(vol_path, dtype=dt, mode='r', shape=dims_order)
        else:
            vol = np.fromfile(vol_path, dtype=dt, count=np.prod(dims))
            vol = np.reshape(vol, dims_order)

        if self.return_metadata:
            return vol, meta_data
        else:
            return vol

    def load_dicom(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a DICOM file.

        Args:
            path (str): Path to file

        """
        import pydicom

        dcm_data = pydicom.dcmread(path)

        if self.return_metadata:
            return dcm_data.pixel_array, dcm_data
        else:
            return dcm_data.pixel_array

    def load_dicom_dir(self, path: str | os.PathLike) -> np.ndarray:
        """
        Load a directory of DICOM files into a numpy 3d array.

        Args:
            path (str): Directory path

        returns:
            numpy.ndarray, numpy.memmap or tuple: The loaded volume.
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).

        """
        import pydicom

        if not self.contains:
            msg = "Please specify a part of the name that is common for the DICOM file stack with the argument 'contains'"
            raise ValueError(msg)

        dicom_stack = [file for file in os.listdir(path) if self.contains in file]
        dicom_stack.sort()  # Ensure proper ordering

        # Check that only one DICOM stack in the directory contains the provided string in its name
        dicom_stack_only_letters = []
        for filename in dicom_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            dicom_stack_only_letters.append(
                ''.join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements from tiff_stack_only_letters
        unique_names = list(set(dicom_stack_only_letters))
        if len(unique_names) > 1:
            f'The provided part of the filename for the DICOM stack matches multiple DICOM stacks: {unique_names}.\nPlease provide a string that is unique for the DICOM stack that is intended to be loaded'
            raise ValueError(msg)

        # dicom_list contains the dicom objects with metadata
        dicom_list = [pydicom.dcmread(os.path.join(path, f)) for f in dicom_stack]
        # vol contains the pixel data
        vol = np.stack([dicom.pixel_array for dicom in dicom_list], axis=0)

        if self.return_metadata:
            return vol, dicom_list
        else:
            return vol

    def load_zarr(self, path: str | os.PathLike) -> np.ndarray:
        """
        Loads a Zarr array from disk.

        Args:
            path (str): The path to the Zarr array on disk.

        Returns:
            numpy.ndarray | zarr.core.array.Array: The numpy array loaded from disk.
                If 'self.virtual_stack' is True, returns a Zarr array object.

        """

        if self.virtual_stack:
            vol = zarr.open(path)
        else:
            vol = zarr.load(path)

        return vol

    def check_file_size(self, filename: str) -> None:
        """
        Checks if there is enough memory where the file can be loaded.

        Args:
        ----
            filename: (str) Specifies path to file
            force_load: (bool, optional) If true, the memory error will not be raised. Warning will be printed insted and
                the loader will attempt to load the file.

        Raises:
        ------
            MemoryError: If filesize is greater then available memory

        """

        if self.virtual_stack:  # If virtual_stack is True, then data is loaded from the disk, no need for loading into memory
            return
        file_size = get_file_size(filename)
        available_memory = Memory().free
        if file_size > available_memory:
            message = f'The file {filename} has {sizeof(file_size)} but only {sizeof(available_memory)} of memory is available.'
            if self.force_load:
                log.warning(message)
            else:
                raise MemoryError(
                    message + " Set 'force_load=True' to ignore this error."
                )

    def load(self, path: str | os.PathLike) -> np.ndarray:
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
            if path.endswith(('.tif', '.tiff')):
                return self.load_tiff(path)
            elif path.endswith('.h5'):
                return self.load_h5(path)
            elif path.endswith(('.txrm', '.txm', '.xrm')):
                return self.load_txrm(path)
            elif path.endswith(('.nii', '.nii.gz')):
                return self.load_nifti(path)
            elif path.endswith(('.vol', '.vgi')):
                return self.load_vol(path)
            elif path.endswith(('.dcm', '.DCM')):
                return self.load_dicom(path)
            else:
                try:
                    return self.load_pil(path)
                except UnidentifiedImageError:
                    msg = 'Unsupported file format'
                    raise ValueError(msg) from None

        # Load a directory
        elif os.path.isdir(path):
            # load tiff stack if folder contains tiff files else load dicom directory
            if any(f.endswith(('.tif', '.tiff')) for f in os.listdir(path)):
                return self.load_tiff_stack(path)

            elif any(f.endswith(self.PIL_extensions) for f in os.listdir(path)):
                return self.load_pil_stack(path)
            elif path.endswith('.zarr'):
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
                msg = 'Invalid path'
                raise ValueError(msg)


def _get_h5_dataset_keys(f) -> list:
    import h5py

    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def _get_ole_offsets(ole) -> dict:
    slice_offset = {}
    for stream in ole.listdir():
        if stream[0].startswith('ImageData'):
            sid = ole._find(stream)
            direntry = ole.direntries[sid]
            sect_start = direntry.isectStart
            offset = ole.sectorsize * (sect_start + 1)
            slice_offset[f'{stream[0]}/{stream[1]}'] = offset

    # sort dictionary after natural sorting (https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/)
    sorted_keys = sorted(
        slice_offset.keys(),
        key=lambda string_: [
            int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)
        ],
    )
    slice_offset_sorted = {key: slice_offset[key] for key in sorted_keys}

    return slice_offset_sorted


def load(
    path: str | os.PathLike,
    virtual_stack: bool = False,
    dataset_name: bool = None,
    return_metadata: bool = False,
    contains: bool = None,
    force_load: bool = False,
    dim_order: tuple = (2, 1, 0),
    progress_bar: bool = False,
    display_memory_usage: bool = False,
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
        force_load (bool, optional): If the file size exceeds available memory, a MemoryError is raised.
            If force_load is True, the error is changed to warning and the loader tries to load it anyway. Default is False.
        dim_order (tuple, optional): The order of the dimensions in the volume for .vol files. Default is (2,1,0) which corresponds to (z,y,x)
        progress_bar (bool, optional): Displays tqdm progress bar. Useful for large files. So far works only for linux. Default is False.
        display_memory_usage (bool, optional): If true, prints used memory and available memory after loading file. Default is False.
        **kwargs (Any): Additional keyword arguments supported by `DataLoader`:
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

    Example: Loading from Tiff stack
        Volumes can also be loaded from a series of `.tiff` files. The stack means that we have one file per slice.

        ```python
        import qim3d

        # Generate volume
        vol = qim3d.generate.volume(noise_scale = 0.015)

        # Save as a .tiff stack
        # The paremeter `basename` is used for the prefix of the files.
        qim3d.io.save("data_directory", vol, basename="blob-slices", sliced_dim=0)

        # Load the volume from the .tiff stack
        # Here we use `contains` to check the files that have that string in their names
        loaded_vol = qim3d.io.load("data_directory" , contains="blob-slices", progress_bar=True)
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

    def log_memory_info(data: np.ndarray) -> None:
        mem = Memory()
        log.info(
            'Volume using %s of memory\n',
            sizeof(data[0].nbytes if isinstance(data, tuple) else data.nbytes),
        )
        mem.report()

    if return_metadata and not isinstance(data, tuple):
        log.warning('The file format does not contain metadata')

    if not virtual_stack:
        if display_memory_usage:
            log_memory_info(data)
    else:
        # Only log if file type is not a np.ndarray, i.e., it is some kind of memmap object
        if not isinstance(
            type(data[0]) if isinstance(data, tuple) else type(data), np.ndarray
        ):
            log.info('Using virtual stack')
        else:
            log.warning('Virtual stack is not supported for this file format')
            if display_memory_usage:
                log_memory_info(data)

    return data


def load_mesh(filename: str) -> hmesh.Manifold:
    """
    Load a mesh from a specific file.
    This function is based on the [PyGEL3D library's loading function implementation](https://www2.compute.dtu.dk/projects/GEL/PyGEL/pygel3d/hmesh.html#load).

    Supported formats:

    - `X3D`
    - `OBJ`
    - `OFF`
    - `PLY`

    Args:
        filename (str or os.PathLike): The path to the file.

    Returns:
        mesh (hmesh.Manifold or None): A hmesh object containing the mesh data or None if loading failed.

    Example:
        ```python
        import qim3d

        mesh = qim3d.io.load_mesh("path/to/mesh.obj")
        ```

    """
    mesh = hmesh.load(filename)

    return mesh
