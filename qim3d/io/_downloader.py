# ruff: noqa: S310
"""Discover QIM datasets and download their volumes."""

import json
import logging
import os
import tempfile
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from ome_zarr.utils import download
from tqdm import tqdm

import qim3d
from qim3d.io._loading import load
from qim3d.utils._misc import sizeof

_logger = logging.getLogger(__name__)

__all__ = ["Downloader"]

_MANIFEST_URL = "https://data-repository.qim.dk/datasets/index.json"


def _fetch_manifest(url: str, timeout: float) -> list[dict]:
    """Fetch the collection's dataset list."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        manifest = json.load(response)
    datasets = manifest.get("datasets") if isinstance(manifest, dict) else None
    if not isinstance(datasets, list):
        raise ValueError(f"Manifest at {url} has no list of datasets.")
    for dataset in datasets:
        volumes = (dataset.get("volumes") or []) if isinstance(dataset, dict) else None
        if not isinstance(volumes, list) or not all(
            isinstance(volume, dict) for volume in volumes
        ):
            raise ValueError(f"Manifest at {url} has a malformed dataset: {dataset!r}")
    return datasets


def _get_file_size(url: str) -> int:
    """Return the remote Content-Length, or -1 if unavailable."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return int(response.info().get("Content-Length", -1))


class Downloader:
    """
    Provides access to the QIM online data repository and manages file downloads.

    This utility allows users to easily fetch, download, and load sample datasets for testing,
    benchmarking, or educational purposes. It automatically handles local caching to avoid
    repeated downloads of the same file.

    The `Downloader` acts as an interface to the [QIM data repository](https://data-repository.qim.dk/),

    Attributes:
        manifest_url (str): URL of the dataset manifest.
        timeout (float): Timeout in seconds for fetching the manifest.

    Methods:
        get_datasets(): Returns the datasets, formats and sizes published in the manifest.
        show_datasets(): Prints a table of the datasets, their categories and sizes.
        download_dataset(dataset_id, format, ...): Downloads a volume and returns its local path.
        load_dataset(dataset_id, format, ...): Downloads a volume if needed and returns its image data.
        refresh(): Fetches the manifest again.

    Syntax for downloading and loading a dataset:
    `qim3d.io.Downloader().load_dataset("cowry-shell", format="zarr")`

    ??? info "Overview of available data"
        See the current datasets and formats on the [QIM data repository](https://data.qim.dk/),
        or call `get_datasets()` to inspect them in Python.

    Example:
        ```python
        import qim3d

        downloader = qim3d.io.Downloader()

        # Browse available datasets
        datasets = downloader.get_datasets()

        # Download and load a sample
        data = downloader.load_dataset("cowry-shell", format="zarr", scale="lowest")

        qim3d.viz.slicer_orthogonal(data, colormap="magma")
        ```
        ![cowry shell](../../assets/screenshots/cowry_shell_slicer.gif)
    """

    def __init__(
        self,
        manifest_url: str = _MANIFEST_URL,
        timeout: float = 10,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.manifest_url = manifest_url
        self.timeout = timeout
        self._datasets: list[dict] | None = None

    def refresh(self) -> None:
        """Fetch the manifest again, replacing the cached catalog on success."""
        self._datasets = _fetch_manifest(self.manifest_url, self.timeout)

    def _get_datasets(self) -> list[dict]:
        if self._datasets is None:
            self._datasets = _fetch_manifest(self.manifest_url, self.timeout)
        return self._datasets

    def get_datasets(self) -> list[dict[Any, Any]]:
        """Return a list of all available datasets.

        Each entry in a dataset's ``volumes`` has its ``format``, ``url`` and
        ``size_bytes`` (``None`` when the size is not published).
        """
        return deepcopy(self._get_datasets())

    def show_datasets(self) -> None:
        """Print a table of the datasets with their categories and volume sizes."""
        rows = [("ID", "Categories", "TIFF", "Zarr")]
        for dataset in self._get_datasets():
            sizes = {
                volume.get("format"): volume.get("size_bytes")
                for volume in dataset.get("volumes") or []
            }
            tiff, zarr = (
                sizeof(size) if type(size) is int and size > 0 else "-"
                for size in (sizes.get("tiff"), sizes.get("zarr"))
            )
            categories = ", ".join(map(str, dataset.get("categories") or []))
            rows.append((str(dataset.get("id")), categories, tiff, zarr))
        widths = [max(len(cell) for cell in column) for column in zip(*rows)]
        for row in rows:
            line = "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
            print(line.rstrip())

    def _get_volume(
        self, dataset_id: str, volume_format: str
    ) -> tuple[str, str, int | None]:
        """Return the URL, file name and size in bytes (None if unknown) of a volume."""
        dataset = next(
            (item for item in self._get_datasets() if item.get("id") == dataset_id),
            None,
        )
        if dataset is None:
            raise LookupError(
                f"Dataset {dataset_id!r} was not found. "
                "Use get_datasets() to see available IDs."
            )
        volumes = dataset.get("volumes") or []
        volume = next(
            (item for item in volumes if item.get("format") == volume_format), None
        )
        if volume is None:
            available = ", ".join(
                str(item.get("format")) for item in volumes if item.get("url")
            )
            raise LookupError(
                f"Dataset {dataset_id!r} has no {volume_format!r} volume. "
                f"Available formats: {available or 'none'}."
            )
        url = volume.get("url")
        if isinstance(url, str):
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            if (
                parsed.scheme in {"http", "https"}
                and parsed.netloc
                and filename not in {"", ".", ".."}
            ):
                size_bytes = volume.get("size_bytes")
                if type(size_bytes) is not int or size_bytes <= 0:
                    size_bytes = None
                return url, filename, size_bytes
        raise ValueError(
            f"Dataset {dataset_id!r} has no usable download URL for "
            f"format {volume_format!r}"
        )

    def download_dataset(
        self,
        dataset_id: str,
        *,
        format: Literal["tiff", "zarr"],
        output_dir: str | os.PathLike = ".",
    ) -> Path:
        """Download a manifest volume and return its local path.

        ``format`` is "tiff" or "zarr". The volume is stored under
        ``output_dir/dataset_id/``. An existing path is reused;
        new downloads are staged so failures do not leave a partial final path.
        """
        # The ID must be a single folder name, so it can't escape output_dir, for example "../coal-briquette"
        if (
            not isinstance(dataset_id, str)
            or dataset_id in {"", ".", ".."}
            or Path(dataset_id).name != dataset_id
        ):
            raise ValueError(f"Invalid dataset ID {dataset_id!r}")
        dataset_dir = Path(output_dir) / dataset_id

        url, filename, size_bytes = self._get_volume(dataset_id, format)
        destination = dataset_dir / filename
        if destination.exists():
            _logger.info("Dataset volume already downloaded: %s", destination)
            return destination
        if destination.is_symlink():
            raise FileNotFoundError(f"{destination} is a broken symbolic link")

        dataset_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".download-", dir=dataset_dir
        ) as staging:
            staged = Path(staging) / filename
            self._download_url(url, staged, format, size_bytes)
            if destination.exists():
                return destination
            os.replace(staged, destination)
        return destination

    def load_dataset(
        self,
        dataset_id: str,
        *,
        format: Literal["tiff", "zarr"],
        output_dir: str | os.PathLike = ".",
        virtual_stack: bool = True,
        scale: int | Literal["lowest", "highest"] = 0,
    ) -> object:
        """Download a volume if needed, then return its image data.

        ``format`` is "tiff" or "zarr". ``virtual_stack=True`` uses lazy loading
        where supported. ``scale`` selects
        an OME-Zarr resolution (0, a coarser integer, "highest", or "lowest");
        other formats only accept the default scale of 0.
        """
        if format != "zarr" and scale != 0:
            raise ValueError("scale is only supported for OME-Zarr volumes")
        path = self.download_dataset(dataset_id, format=format, output_dir=output_dir)
        if format == "zarr":
            return qim3d.io.import_ome_zarr(path, scale=scale, load=not virtual_stack)
        return load(path=path, virtual_stack=virtual_stack)

    def _download_url(
        self,
        url: str,
        destination: Path,
        volume_format: str,
        size_bytes: int | None = None,
    ) -> None:
        """Download a volume to ``destination``.

        ``size_bytes`` is the expected size, used for logging and the progress bar.
        """
        size = f" ({sizeof(size_bytes)})" if size_bytes else ""
        if volume_format == "zarr":
            _logger.info(
                "Downloading Zarr store %s%s from %s", destination.name, size, url
            )
            download(url, output_dir=str(destination.parent))
            return
        _logger.info("Downloading file %s%s from %s", destination.name, size, url)
        total = size_bytes
        if total is None:
            try:
                total = _get_file_size(url)
            except OSError:
                total = -1
        with tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=80,
        ) as pbar:
            urllib.request.urlretrieve(
                url,
                destination,
                reporthook=lambda blocknum, block_size, _total_size: pbar.update(
                    blocknum * block_size - pbar.n
                ),
            )
