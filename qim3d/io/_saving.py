"""
Provides functionality for saving data from various file formats.


Example:
    ```python
    import qim3d
    
    # Generate synthetic blob
    synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)
    
    qim3d.io.save("fly.tif", synthetic_blob)
    ```

    Volumes can also be saved with one file per slice:
    ```python
    import qim3d

    # Generate synthetic blob
    synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)
    
    qim3d.io.save("slices", synthetic_blob, basename="fly-slices", sliced_dim=0)
    ```

"""

import datetime
import os
import dask.array as da
import h5py
import nibabel as nib
import numpy as np
import PIL
import tifffile
import zarr
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import trimesh

from qim3d.utils import log
from qim3d.utils._misc import sizeof, stringify_path


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
        self.chunk_shape = kwargs.get("chunk_shape", "auto")

    def save_tiff(self, path: str|os.PathLike, data: np.ndarray):
        """Save data to a TIFF file to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        """
        tifffile.imwrite(path, data, compression=self.compression)

    def save_tiff_stack(self, path: str|os.PathLike, data: np.ndarray):
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

            pattern_string = (
                filepath[: -(len(extension) + zfill_val)] + "-" * zfill_val + extension
            )

            log.info(
                f"Total of {no_slices} files saved following the pattern '{pattern_string}'"
            )

    def save_nifti(self, path: str|os.PathLike, data: np.ndarray):
        """Save data to a NIfTI file to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        """
        import nibabel as nib

        # Create header
        header = nib.Nifti1Header()
        header.set_data_dtype(data.dtype)

        # Create NIfTI image object
        img = nib.Nifti1Image(data, np.eye(4), header)

        # nib does automatically compress if filetype ends with .gz
        if self.compression and not path.endswith(".gz"):
            path += ".gz"
            log.warning("File extension '.gz' is added since compression is enabled.")

        if not self.compression and path.endswith(".gz"):
            path = path[:-3]
            log.warning(
                "File extension '.gz' is ignored since compression is disabled."
            )

        # Save image
        nib.save(img, path)

    def save_vol(self, path: str|os.PathLike, data: np.ndarray):
        """Save data to a VOL file to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        """
        # No support for compression yet
        if self.compression:
            raise NotImplementedError(
                "Saving compressed .vol files is not yet supported"
            )

        # Create custom .vgi metadata file
        metadata = ""
        metadata += "{volume1}\n"  # .vgi organization
        metadata += "[file1]\n"  # .vgi organization
        metadata += "Size = {} {} {}\n".format(
            data.shape[1], data.shape[2], data.shape[0]
        )  # Swap axes to match .vol format
        metadata += "Datatype = {}\n".format(str(data.dtype))  # Get datatype as string
        metadata += "Name = {}.vol\n".format(
            path.rsplit("/", 1)[-1][:-4]
        )  # Get filename without extension

        # Save metadata
        with open(path[:-4] + ".vgi", "w") as f:
            f.write(metadata)

        # Save data using numpy in binary format
        data.tofile(path[:-4] + ".vol")

    def save_h5(self, path, data):
        """Save data to a HDF5 file to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        """
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset(
                "dataset", data=data, compression="gzip" if self.compression else None
            )

    def save_dicom(self, path: str|os.PathLike, data: np.ndarray):
        """Save data to a DICOM file to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        """
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import UID

        # based on https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_write_dicom.html

        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
        file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
        file_meta.ImplementationClassUID = UID("1.2.3.4")

        # Create the FileDataset instance (initially no data elements, but file_meta
        # supplied)
        ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.PatientName = "Test^Firstname"
        ds.PatientID = "123456"
        ds.StudyInstanceUID = "1.2.3.4.5"
        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 0
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = data.shape[1]
        ds.Columns = data.shape[2]
        ds.NumberOfFrames = data.shape[0]
        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # Set creation date/time
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime("%Y%m%d")
        timeStr = dt.strftime("%H%M%S.%f")  # long format with micro seconds
        ds.ContentTime = timeStr
        # Needs to be here because of bug in pydicom
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Reshape the data into a 1D array and convert to uint16
        data_1d = data.ravel().astype(np.uint16)
        # Convert the data to bytes
        data_bytes = data_1d.tobytes()
        # Add the data to the DICOM file
        ds.PixelData = data_bytes

        ds.save_as(path)

    def save_to_zarr(self, path: str|os.PathLike, data: da.core.Array):
        """Saves a Dask array to a Zarr array on disk.

        Args:
            path (str): The path to the Zarr array on disk.
            data (dask.array.core.Array): The Dask array to be saved to disk.

        Returns:
            zarr.core.Array: The Zarr array saved on disk.
        """

        if isinstance(data, da.Array):
            # If the data is a Dask array, save using dask
            if self.chunk_shape:
                log.info("Rechunking data to shape %s", self.chunk_shape)
                data = data.rechunk(self.chunk_shape)
            log.info("Saving Dask array to Zarr array on disk")
            da.to_zarr(data, path, overwrite=self.replace)

        else:
            zarr_array = zarr.open(
                path,
                mode="w",
                shape=data.shape,
                chunks=self.chunk_shape,
                dtype=data.dtype,
            )
            zarr_array[:] = data

    def save_PIL(self, path: str|os.PathLike, data: np.ndarray):
        """Save data to a PIL file to the given path.

        Args:
            path (str): The path to save file to
            data (numpy.ndarray): The data to be saved
        """
        # No support for compression yet
        if self.compression and path.endswith(".png"):
            raise NotImplementedError("png does not support compression")
        elif not self.compression and path.endswith((".jpeg", ".jpg")):
            raise NotImplementedError("jpeg does not support no compression")

        # Convert to PIL image
        img = PIL.Image.fromarray(data)

        # Save image
        img.save(path)

    def save(self, path: str|os.PathLike, data: np.ndarray):
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
        """

        path = stringify_path(path)
        isdir = os.path.isdir(path)
        _, ext = os.path.splitext(path)

        # If path is an existing directory
        if isdir:
            # Check if this is a Zarr directory
            if ".zarr" in path:
                if self.replace:
                    return self.save_to_zarr(path, data)
                if not self.replace:
                    raise ValueError(
                        "A Zarr directory with the provided path already exists. To replace it set 'replace=True'"
                    )
            # If basename is provided, user wants to save as tiff stack
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
                    elif path.endswith((".nii", "nii.gz")):
                        return self.save_nifti(path, data)
                    elif path.endswith(("TXRM", "XRM", "TXM")):
                        raise NotImplementedError(
                            "Saving TXRM files is not yet supported"
                        )
                    elif path.endswith((".h5")):
                        return self.save_h5(path, data)
                    elif path.endswith((".vol", ".vgi")):
                        return self.save_vol(path, data)
                    elif path.endswith((".dcm", ".DCM")):
                        return self.save_dicom(path, data)
                    elif path.endswith((".zarr")):
                        return self.save_to_zarr(path, data)
                    elif path.endswith((".jpeg", ".jpg", ".png")):
                        return self.save_PIL(path, data)
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
    path: str|os.PathLike,
    data: np.ndarray,
    replace: bool = False,
    compression: bool = False,
    basename: bool = None,
    sliced_dim: int = 0,
    chunk_shape: str = "auto",
    **kwargs,
) -> None:
    """Save data to a specified file path.

    Args:
        path (str or os.PathLike): The path to save file to. File format is chosen based on the extension. 
            Supported extensions are: <em>'.tif', '.tiff', '.nii', '.nii.gz', '.h5', '.vol', '.vgi', '.dcm', '.DCM', '.zarr', '.jpeg', '.jpg', '.png'</em>
        data (numpy.ndarray): The data to be saved
        replace (bool, optional): Specifies if an existing file with identical path should be replaced.
            Default is False.
        compression (bool, optional): Specifies if the file should be saved with Deflate compression (lossless).
            Default is False.
        basename (str, optional): Specifies the basename for a TIFF stack saved as several files
            (only relevant for TIFF stacks). Default is None
        sliced_dim (int, optional): Specifies the dimension that is sliced in case a TIFF stack is saved
            as several files (only relevant for TIFF stacks). Default is 0, i.e., the first dimension.
        **kwargs (Any): Additional keyword arguments to be passed to the DataSaver constructor

    Raises:
        ValueError: If the provided path is an existing directory and self.basename is not provided <strong>OR</strong>
            If the file format is not supported <strong>OR</strong>
            If the provided path does not exist and self.basename is not provided <strong>OR</strong>
            If a file extension is not provided <strong>OR</strong>
            if a file with the specified path already exists and replace=False.

    Example:
        ```python
        import qim3d

        # Generate synthetic blob
        vol = qim3d.generate.noise_object(noise_scale = 0.015)

        qim3d.io.save("blob.tif", vol, replace=True)
        ```

        Volumes can also be saved with one file per slice:
        ```python
        import qim3d

        # Generate synthetic blob
        vol = qim3d.generate.noise_object(noise_scale = 0.015)

        qim3d.io.save("slices", vol, basename="blob-slices", sliced_dim=0)
        ```
    """

    DataSaver(
        replace=replace,
        compression=compression,
        basename=basename,
        sliced_dim=sliced_dim,
        chunk_shape=chunk_shape,
        **kwargs,
    ).save(path, data)


def save_mesh(
        filename: str, 
        mesh: trimesh.Trimesh
        ) -> None:
    """
    Save a trimesh object to an .obj file.

    Args:
        filename (str or os.PathLike): The name of the file to save the mesh.
        mesh (trimesh.Trimesh): A trimesh.Trimesh object representing the mesh.

    Example:
        ```python
        import qim3d

        vol = qim3d.generate.noise_object(base_shape=(32, 32, 32),
                                  final_shape=(32, 32, 32),
                                  noise_scale=0.05,
                                  order=1,
                                  gamma=1.0,
                                  max_value=255,
                                  threshold=0.5)
        mesh = qim3d.mesh.from_volume(vol)
        qim3d.io.save_mesh("mesh.obj", mesh)
        ```
    """
    # Export the mesh to the specified filename
    mesh.export(filename)