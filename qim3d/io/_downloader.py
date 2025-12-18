# ruff: noqa: S310
"""Manages downloads and access to data."""

import os
import urllib.request
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse

import outputformat as ouf
from ome_zarr.utils import download
from tqdm import tqdm

import qim3d
from qim3d.io import load
from qim3d.utils import log

__all__ = ['Downloader', 'download_file']


class _Myfolder:
    """
    Class for extracting the files from each folder in the Downloader class.

    Args:
        folder(str): name of the folder of interest in the QIM data repository.

    Methods:
            _make_fn(folder,file): creates custom functions for each file found in the folder.
        [file_name_1](load_file,optional): Function to download file number 1 in the given folder.
        [file_name_2](load_file,optional): Function to download file number 2 in the given folder.
        ...
        [file_name_n](load_file,optional): Function to download file number n in the given folder.

    """

    def __init__(self, folder: str):
        files = _extract_names(folder)

        for _, file in enumerate(files):
            # Changes names to usable function name.
            file_name = file
            if ('%20' in file) or ('-' in file):
                file_name = file_name.replace('%20', '_')
                file_name = file_name.replace('-', '_')

            name = file_name.split('.')[0]
            setattr(self, name, self._make_fn(folder, file))

    def _make_fn(self, folder: str, file: str) -> Callable[[bool, bool], object]:
        """
        Private method that returns a function. The function downloads the chosen file from the folder.

        Args:
            folder(str): Folder where the file is located.
            file(str): Name of the file to be downloaded.

        Returns:
                function: the function used to download the file.

        """

        url_dl = 'https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository'

        def _download(load_file: bool = False, virtual_stack: bool = True) -> object:
            """
            Downloads the file and optionally also loads it.

            Args:
                load_file(bool,optional): Whether to simply download or also load the file.
                virtual_stack(bool,optional): Whether to load the file as a virtual stack.

            Returns:
                virtual_stack: The loaded image.

            """

            download_file(url_dl, folder, file)
            if load_file:
                log.info(f'\nLoading {file}')
                file_path = os.path.join(folder, file)

                return load(path=file_path, virtual_stack=virtual_stack)

        return _download


class Downloader:
    """
    Class for downloading large data files available on the [QIM data repository](https://data.qim.dk/).

    Attributes:
        folder_name (str or os.PathLike): Folder class with the name of the folder in <https://data.qim.dk/>

    Methods:
        list_files(): Prints the downloadable files from the QIM data repository.

    Syntax for downloading and loading a file is `qim3d.io.Downloader().{folder_name}.{file_name}(load_file=True)`

    ??? info "Overview of available data"
        Below is a table of the available folders and files on the [QIM data repository](https://data.qim.dk/).

        Folder name         | File name                                                                                                          | File size
        ------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------
        `Coal`              | `CoalBrikett` <br> `CoalBrikett_Zoom` <br> `CoalBrikettZoom_DOWNSAMPLED`                                           | 2.23 GB <br> 3.72 GB <br> 238 MB
        `Corals`            | `Coral_1` <br> `Coral_2` <br> `Coral2_DOWNSAMPLED` <br> `MexCoral`                                                 | 2.26 GB <br> 2.38 GB <br> 162 MB <br> 2.23 GB
        `Cowry_Shell`       | `Cowry_Shell` <br> `Cowry_DOWNSAMPLED`                                                                             | 1.82 GB <br> 116 MB
        `Crab`              | `HerrmitCrab` <br> `OkinawaCrab`                                                                                   | 2.38 GB <br> 1.86 GB
        `Deer_Mandible`     | `Animal_Mandible` <br> `DeerMandible_DOWNSAMPLED` <br>                                                             | 2.79 GB <br> 638 MB
        `Foam`              | `Foam` <br> `Foam_DOWNSAMPLED` <br> `Foam_2` <br> `Foam_2_zoom`                                                    | 3.72 GB <br> 238 MB <br> 3.72 GB <br> 3.72 GB
        `Hourglass`         | `Hourglass` <br> `Hourglass_4X_80kV_Air_9s_1_97um` <br> `Hourglass_longexp_rerun`                                  | 3.72 GB <br> 1.83 GB <br> 3.72 GB
        `Kiwi`              | `Kiwi`                                                                                                             | 2.86 GB
        `Loofah`            | `Loofah` <br> `Loofah_DOWNSAMPLED`                                                                                 | 2.23 GB <br> 143 MB
        `Marine_Gastropods` | `MarineGatropod_1` <br> `MarineGastropod1_DOWNSAMPLED` <br> `MarineGatropod_2` <br> `MarineGastropod2_DOWNSAMPLED` | 2.23 GB <br> 143 MB <br> 2.60 GB <br> 166 MB
        `Mussel`            | `ClosedMussel1` <br> `ClosedMussel1_DOWNSAMPLED`                                                                   | 2.23 GB <br> 143 MB
        `Oak_Branch`        | `Oak_branch` <br> `OakBranch_DOWNSAMPLED`                                                                          | 2.38 GB <br> 152 MB
        `Okinawa_Forams`    | `Okinawa_Foram_1` <br> `Okinawa_Foram_2`                                                                           | 1.84 GB <br> 1.84 GB
        `Physalis`          | `Physalis` <br> `Physalis_DOWNSAMPLED`                                                                             | 3.72 GB <br> 238 MB
        `Raspberry`         | `Raspberry2` <br> `Raspberry2_DOWNSAMPLED`                                                                         | 2.97 GB <br> 190 MB
        `Rope`              | `FibreRope1` <br> `FibreRope1_DOWNSAMPLED`                                                                         | 1.82 GB <br> 686 MB
        `Sea_Urchin`        | `SeaUrchin` <br> `Cordatum_Shell` <br> `Cordatum_Spine`                                                            | 2.60 GB <br> 1.85 GB <br> 183 MB
        `Snail`             | `Escargot`                                                                                                         | 2.60 GB
        `Sponge`            | `Sponge`                                                                                                           | 1.11 GB

    Example:
        ```python
        import qim3d

        downloader = qim3d.io.Downloader()
        downloader.list_files()
        data = downloader.Cowry_Shell.Cowry_DOWNSAMPLED(load_file=True)

        qim3d.viz.slicer_orthogonal(data, colormap="magma")
        ```
        ![cowry shell](../../assets/screenshots/cowry_shell_slicer.gif)

    """

    def __init__(self):
        folders = _extract_names()
        for folder in folders:
            setattr(self, folder, _Myfolder(folder))

    def __call__(
        self,
        url: str,
        output_dir: str = '.',
        load_file: bool = False,
        virtual_stack: bool = True,
        scale: int = 0,
    ) -> object:
        """
        Download any file given its URL.

        Args:
            url (str):
                URL of the file to download. Supported formats are regular files
                (TIFF, HDF5, TXRM/TXM/XRM, NIfTI, PIL, VOL/VGI, DICOM) and
                Zarr/OME-Zarr stores (.zarr).
            output_dir (str, optional):
                Base directory to save files. Default is the current working directory.
            load_file (bool, optional):
                If True, load the file after download. Default is False.
            virtual_stack (bool, optional):
                If True and the file format supports it, load the file on demand
                as a virtual stack (lazy loading). Default is True.
            scale (int, optional):
                If `load_file` is True and the file is a Zarr/OME-Zarr store, the scale parameter specifies the resolution level to load. Default is 0 (full resolution).

        Returns:
        str or numpy.ndarray or dask.array.Array:
            - If `load_file` is False, returns the path to the downloaded file
            or Zarr store.
            - If `load_file` is True and the file is a **regular file**:
                - Returns a NumPy array if `virtual_stack=False`.
                - Returns a virtual stack (lazy-loaded NumPy-like object) if
                `virtual_stack=True`.
            - If `load_file` is True and the file is a **Zarr/OME-Zarr store**:
                - Returns a NumPy array at the requested `scale` if
                `virtual_stack=False`.
                - Returns a Dask array at the requested `scale` if
                `virtual_stack=True`.

        Example:
            ```python
            import qim3d

            downloader = qim3d.io.Downloader()

            # Download a file without loading
            path = downloader(
                url="https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository/Cowry_Shell/Cowry_DOWNSAMPLED.tif",
                output_dir=".",
                load_file=False
            )

            # Download and load directly as a NumPy array
            data = downloader(
                url="https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository/Cowry_Shell/Cowry_DOWNSAMPLED.tif",
                load_file=True,
                virtual_stack=False
            )
            ```

        """

        parsed = urlparse(url)
        fname = os.path.basename(parsed.path.rstrip('/'))
        dest = os.path.join(output_dir, fname)

        # --- Zarr / OME-Zarr store ---
        if fname.endswith(('.zarr', '.ome.zarr')):
            if os.path.exists(dest):
                log.warning(f'Zarr store already downloaded:\n{os.path.abspath(dest)}')
            else:
                log.info(f'Downloading Zarr store {fname}\n{url}')
                download(url, output_dir=output_dir)  # return always None
            if load_file:
                # If virtual stack == True --> dask array --> need to call False in load (we don't want call .compute())
                # If virtual stack == False --> numpy array --> need to call True in load (we want call .compute())
                log.info(
                    f"\nLoading scale={scale} from {fname} as {'numpy array' if not virtual_stack else 'dask array'}"
                )
                return qim3d.io.import_ome_zarr(
                    dest, scale=scale, load=not virtual_stack
                )
            return dest

        # --- Regular single file ---
        if os.path.exists(dest):
            log.warning(f'File already downloaded:\n{os.path.abspath(dest)}')
            if load_file:
                return load(path=dest, virtual_stack=virtual_stack)
            return dest
        else:
            log.info(f'Downloading file {fname}\n{url}')
            try:
                total = _get_file_size(url)
            except (HTTPError, URLError):
                total = None

            os.makedirs(output_dir, exist_ok=True)
            with tqdm(
                total=total, unit='B', unit_scale=True, unit_divisor=1024, ncols=80
            ) as pbar:
                try:
                    urllib.request.urlretrieve(
                        url,
                        dest,
                        reporthook=lambda blocknum, bs, total: _update_progress(
                            pbar, blocknum, bs
                        ),
                    )
                except HTTPError as http_err:
                    msg = f'Failed to download {url!r}: server returned HTTP {http_err.code}'
                    raise FileNotFoundError(msg) from http_err
                except URLError as url_err:
                    msg = f'Failed to reach {url!r}: {url_err.reason}'
                    raise ConnectionError(msg) from url_err

        if load_file:
            log.info(f'\nLoading {fname}')
            return load(path=dest, virtual_stack=virtual_stack)

        return dest

    def list_files(self) -> None:
        """Generate and print formatted folder, file, and size information."""

        url_dl = 'https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository'

        folders = _extract_names()

        for folder in folders:
            log.info(f'\n{ouf.boxtitle(folder, return_str=True)}')
            files = _extract_names(folder)

            for file in files:
                url = os.path.join(url_dl, folder, file).replace('\\', '/')
                file_size = _get_file_size(url)
                formatted_file = (
                    f"{file[:-len(file.split('.')[-1])-1].replace('%20', '_')}"
                )
                formatted_size = _format_file_size(file_size)
                path_string = f'{folder}.{formatted_file}'

                log.info(f'{path_string:<50}({formatted_size})')


def _update_progress(pbar: tqdm, blocknum: int, bs: int) -> None:
    """Helper function for the ´download_file()´ function. Updates the progress bar."""

    pbar.update(blocknum * bs - pbar.n)


def _get_file_size(url: str) -> int:
    """Helper function for the ´download_file()´ function. Finds the size of the file."""

    return int(urllib.request.urlopen(url).info().get('Content-Length', -1))


def download_file(path: str, name: str, file: str) -> None:
    """
    Downloads the file from path / name / file.

    Args:
        path(str): path to the folders available.
        name(str): name of the folder of interest.
        file(str): name of the file to be downloaded.

    """

    if not os.path.exists(name):
        os.makedirs(name)

    url = os.path.join(path, name, file).replace('\\', '/')  # if user is on windows
    file_path = os.path.join(name, file)

    if os.path.exists(file_path):
        log.warning(f'File already downloaded:\n{os.path.abspath(file_path)}')
        return
    else:
        log.info(
            f'Downloading {ouf.b(file, return_str=True)}\n{os.path.join(path,name,file)}'
        )

    if ' ' in url:
        url = quote(url, safe=':/')

    with tqdm(
        total=_get_file_size(url),
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        ncols=80,
    ) as pbar:
        urllib.request.urlretrieve(
            url,
            file_path,
            reporthook=lambda blocknum, bs, total: _update_progress(pbar, blocknum, bs),
        )


def _extract_html(url: str) -> str:
    """
    Extracts the html content of a webpage in "utf-8".

    Args:
        url(str): url to the location where all the data is stored.

    Returns:
        html_content(str): decoded html.

    """
    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read().decode(
                'utf-8'
            )  # Assuming the content is in UTF-8 encoding
    except urllib.error.URLError as e:
        log.warning(f'Failed to retrieve data from {url}. Error: {e}')

    return html_content


def _extract_names(name: str = None) -> list[str]:
    """
    Extracts the names of the folders and files.

    Finds the names of either the folders if no name is given,
    or all the names of all files in the given folder.

    Args:
        name(str,optional): name of the folder from which the names should be extracted.

    Returns:
        list: If name is None, returns a list of all folders available.
              If name is not None, returns a list of all files available in the given 'name' folder.

    """

    url = 'https://archive.compute.dtu.dk/files/public/projects/viscomp_data_repository'
    if name:
        datapath = os.path.join(url, name).replace('\\', '/')
        html_content = _extract_html(datapath)

        data_split = html_content.split(
            'files/public/projects/viscomp_data_repository/'
        )[3:]
        data_files = [
            element.split(' ')[0][(len(name) + 1) : -3] for element in data_split
        ]

        return data_files
    else:
        html_content = _extract_html(url)
        split = html_content.split('"icon-folder-open">')[2:]
        folders = [element.split(' ')[0][4:-4] for element in split]

        return folders


def _format_file_size(size_in_bytes: int) -> str:
    # Define size units
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(size_in_bytes)
    unit_index = 0

    # Convert to appropriate unit
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    # Format the size with 1 decimal place
    return f'{size:.2f}{units[unit_index]}'
