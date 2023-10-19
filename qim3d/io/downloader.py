"Class for downloading larger images from the QIM Data Repository"

import os
import urllib.request

from urllib.parse import quote
from tqdm import tqdm
from pathlib import Path

from qim3d.io.load import load
from qim3d.io.logger import log
import outputformat as ouf


class Downloader:
    """Class for downloading large data files available on the QIM data repository.

    Attributes:
        [folder_name_1] (str): folder class with the name of the first folder in the QIM data repository.
        [folder_name_2] (str): folder class with the name of the second folder in the QIM data repository.
        ...
        [folder_name_n] (str): folder class with the name of the n-th folder in the QIM data repository.

    Example:
        dl = Downloader()
        # Downloads and Loads (optional) image:
        img = dl.Corals.Coral2_DOWNSAMPLED(load = True)
    """

    def __init__(self):
        folders = _extract_names()
        for idx, folder in enumerate(folders):
            exec(f"self.{folder} = self._Myfolder(folder)")

    class _Myfolder:
        """Class for extracting the files from each folder in the Downloader class.

        Args:
            folder(str): name of the folder of interest in the QIM data repository.

        Methods:
             _make_fn(folder,file): creates custom functions for each file found in the folder.
            [file_name_1](load_file,optional): Function to download file number 1 in the given folder.
            [file_name_2](load_file,optional): Function to download file number 2 in the given folder.
            ...
            [file_name_n](load_file,optional): Function to download file number n in the given folder.
        """

        def __init__(self, folder):
            files = _extract_names(folder)

            for idx, file in enumerate(files):
                # Changes names to usable function name.
                file_name = file
                if ("%20" in file) or ("-" in file):
                    file_name = file_name.replace("%20", "_")
                    file_name = file_name.replace("-", "_")

                setattr(self, f'{file_name.split(".")[0]}', self._make_fn(folder, file))

        def _make_fn(self, folder, file):
            """Private method that returns a function. The function downloads the chosen file from the folder.

            Args:
                folder(str): Folder where the file is located.
                file(str): Name of the file to be downloaded.

            Returns:
                    function: the function used to download the file.
            """

            url_dl = "https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository"

            def _download(load_file=False, virtual_stack=True):
                """Downloads the file and optionally also loads it.

                Args:
                    load_file(bool,optional): Whether to simply download or also load the file.

                Returns:
                    virtual_stack: The loaded image.
                """

                download_file(url_dl, folder, file)
                if load_file == True:
                    log.info(f"\nLoading {file}")
                    file_path = os.path.join(folder, file)

                    return load(path=file_path, virtual_stack=virtual_stack)

            return _download


def _update_progress(pbar, blocknum, bs):
    """
    Helper function for the ´download_file()´ function. Updates the progress bar.
    """

    pbar.update(blocknum * bs - pbar.n)


def _get_file_size(url):
    """
    Helper function for the ´download_file()´ function. Finds the size of the file.
    """

    return int(urllib.request.urlopen(url).info().get("Content-Length", -1))


def download_file(path, name, file):
    """Downloads the file from path / name / file.

    Args:
        path(str): path to the folders available.
        name(str): name of the folder of interest.
        file(str): name of the file to be downloaded.
    """

    if not os.path.exists(name):
        os.makedirs(name)

    url = os.path.join(path, name, file).replace("\\", "/")  # if user is on windows
    file_path = os.path.join(name, file)

    if os.path.exists(file_path):
        log.warning(f"File already downloaded:\n{os.path.abspath(file_path)}")
        return
    else:
        log.info(
            f"Downloading {ouf.b(file, return_str=True)}\n{os.path.join(path,name,file)}"
        )

    if " " in url:
        url = quote(url, safe=":/")

    with tqdm(
        total=_get_file_size(url),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        ncols=80,
    ) as pbar:
        urllib.request.urlretrieve(
            url,
            file_path,
            reporthook=lambda blocknum, bs, total: _update_progress(pbar, blocknum, bs),
        )


def _extract_html(url):
    """Extracts the html content of a webpage in "utf-8"

    Args:
        url(str): url to the location where all the data is stored.

    Returns:
        html_content(str): decoded html.
    """

    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read().decode(
                "utf-8"
            )  # Assuming the content is in UTF-8 encoding
    except urllib.error.URLError as e:
        log.warning(f"Failed to retrieve data from {url}. Error: {e}")

    return html_content


def _extract_names(name=None):
    """Extracts the names of the folders and files.

    Finds the names of either the folders if no name is given,
    or all the names of all files in the given folder.

    Args:
        name(str,optional): name of the folder from which the names should be extracted.

    Returns:
        list: If name is None, returns a list of all folders available.
              If name is not None, returns a list of all files available in the given 'name' folder.
    """

    url = "https://archive.compute.dtu.dk/files/public/projects/viscomp_data_repository"
    if name:
        datapath = os.path.join(url, name).replace("\\", "/")
        html_content = _extract_html(datapath)

        data_split = html_content.split(
            "files/public/projects/viscomp_data_repository/"
        )[3:]
        data_files = [
            element.split(" ")[0][(len(name) + 1) : -3] for element in data_split
        ]

        return data_files
    else:
        html_content = _extract_html(url)
        split = html_content.split('"icon-folder-open">')[2:]
        folders = [element.split(" ")[0][4:-4] for element in split]

        return folders
