# ruff: noqa: S310
"""Discover QIM datasets and download their volumes."""

import json
import logging
import os
import tempfile
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ome_zarr.utils import download
from tqdm import tqdm

import qim3d
from qim3d.io._loading import load

_logger = logging.getLogger(__name__)

__all__ = ["Downloader"]

MANIFEST_URL = "https://data-repository.qim.dk/datasets/index.json"


class ManifestError(ValueError):
    """The dataset manifest could not be used."""


class DatasetNotFoundError(LookupError):
    """The requested dataset ID is absent from the manifest."""


class VolumeNotFoundError(LookupError):
    """The requested format is absent from a dataset."""


class VolumeUnavailableError(ValueError):
    """The requested volume has no download URL."""


def _fetch_manifest(url: str, timeout: float) -> list[dict]:
    """Fetch the collection's dataset list."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)["datasets"]
    except Exception as exc:
        raise ManifestError("Could not load dataset manifest") from exc


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

    The `Downloader` acts as an interface to the [QIM data repository](https://data.qim.dk/),
    The `Downloader` acts as an interface to the [QIM data repository](https://data-repository.qim.dk/),

    Attributes:
        manifest_url (str): URL of the dataset manifest.
        timeout (float): Timeout in seconds for fetching the manifest.

    Methods:
        list_datasets(): Returns the datasets and formats published in the manifest.
        download_dataset(dataset_id, format, ...): Downloads a volume and returns its local path.
        load_dataset(dataset_id, format, ...): Downloads a volume if needed and returns its image data.
        refresh(): Fetches the manifest again.

    Syntax for downloading and loading a dataset:
    `qim3d.io.Downloader().load_dataset("cowry-shell", format="zarr")`

    ??? info "Overview of available data"
        See the current datasets and formats on the [QIM data repository](https://data.qim.dk/),
        or call `list_datasets()` to inspect them in Python.

    Example:
        ```python
        import qim3d

        downloader = qim3d.io.Downloader()

        # Browse available datasets
        datasets = downloader.list_datasets()

        # Download and load a sample
        data = downloader.load_dataset("cowry-shell", format="zarr", scale="lowest")

        qim3d.viz.slicer_orthogonal(data, colormap="magma")
        ```
        ![cowry shell](../../assets/screenshots/cowry_shell_slicer.gif)
    """

    def __init__(
        self,
        manifest_url: str = MANIFEST_URL,
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

    def list_datasets(self) -> list[dict[Any, Any]]:
        """Return a list of all available datasets."""
        if self._datasets is None:
            self.refresh()
        assert self._datasets is not None
        return deepcopy(self._datasets)

    def _get_volume_url(self, dataset_id: str, volume_format: str) -> str:
        if self._datasets is None:
            self.refresh()
        assert self._datasets is not None
        dataset = None
        for item in self._datasets:
            if isinstance(item, dict) and item.get("id") == dataset_id:
                dataset = item
                break
        if dataset is None:
            raise DatasetNotFoundError(
                f"Dataset {dataset_id!r} was not found. "
                "Use list_datasets() to see available IDs."
            )
        volume = None
        for item in dataset.get("volumes") or []:
            if isinstance(item, dict) and item.get("format") == volume_format:
                volume = item
                break
        if volume is None:
            raise VolumeNotFoundError(
                f"Dataset {dataset_id!r} has no {volume_format!r} volume."
            )
        url = volume.get("url")
        if isinstance(url, str):
            parsed = urlparse(url)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return url
        raise VolumeUnavailableError(
            f"Dataset {dataset_id!r} has no usable download URL for "
            f"format {volume_format!r}"
        )

    def download_dataset(
        self,
        dataset_id: str,
        *,
        format: str,
        output_dir: str | os.PathLike = ".",
    ) -> Path:
        """Download a manifest volume and return its local path.

        Store it under ``output_dir/dataset_id/``. An existing path is reused;
        new downloads are staged so failures do not leave a partial final path.
        """
        try:
            dataset_dir = Path(output_dir) / dataset_id
            # Checks that id doesn't contain folder-escaping sequences, for example "../coal-briquette"
            if (
                not isinstance(dataset_id, str)
                or Path(output_dir).resolve() not in dataset_dir.resolve().parents
            ):
                raise ValueError
        except Exception as exc:
            raise ValueError(f"Invalid dataset ID {dataset_id!r}") from exc

        url = self._get_volume_url(dataset_id, format)
        filename = Path(str(urlparse(url).path)).name
        if not filename or filename in {".", ".."}:
            raise VolumeUnavailableError(
                f"Dataset {dataset_id!r} has no usable download URL for "
                f"format {format!r}"
            )
        destination = dataset_dir / filename
        if destination.exists():
            _logger.info("Dataset volume already downloaded: %s", destination)
            return destination

        dataset_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".download-", dir=dataset_dir
        ) as staging:
            staged = self._download_url(url, output_dir=staging)
            if destination.exists():
                return destination
            os.replace(staged, destination)
        return destination

    def load_dataset(
        self,
        dataset_id: str,
        *,
        format: str,
        output_dir: str | os.PathLike = ".",
        virtual_stack: bool = True,
        scale: int | str = 0,
    ) -> object:
        """Download a volume if needed, then return its image data.

        ``virtual_stack=True`` uses lazy loading where supported. ``scale`` selects
        an OME-Zarr resolution (0, a coarser integer, "highest", or "lowest");
        other formats only accept the default scale of 0.
        """
        if format not in {"zarr", "ome-zarr"} and scale != 0:
            raise ValueError("scale is only supported for OME-Zarr volumes")
        path = self.download_dataset(dataset_id, format=format, output_dir=output_dir)
        if format in {"zarr", "ome-zarr"}:
            return qim3d.io.import_ome_zarr(path, scale=scale, load=not virtual_stack)
        return load(path=path, virtual_stack=virtual_stack)

    def _download_url(
        self,
        url: str,
        output_dir: str | os.PathLike,
    ) -> Path:
        """Download a volume into a staging directory and return its path."""
        filename = Path(str(urlparse(url).path)).name
        output_path = Path(output_dir)
        destination = output_path / filename

        if destination.exists():
            _logger.warning("Already downloaded: %s", destination.resolve())
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            if filename.endswith(".zarr"):
                _logger.info("Downloading Zarr store %s from %s", filename, url)
                download(url, output_dir=str(output_path))
            else:
                _logger.info("Downloading file %s from %s", filename, url)
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
                        reporthook=lambda blocknum, block_size, _total_size: (
                            pbar.update(blocknum * block_size - pbar.n)
                        ),
                    )

        return destination
