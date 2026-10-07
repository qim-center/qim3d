import io
import json
import urllib.request
from pathlib import Path
from unittest.mock import ANY, Mock
from urllib.error import URLError

import pytest

from qim3d.io import Downloader

MANIFEST_URL = "https://example.org/datasets/index.json"
TIFF_URL = "https://example.org/coral/coral.tif"
ZARR_URL = "https://example.org/coral/coral.zarr"


def published_volume(manifest, volume_format):
    """Return the manifest entry of the coral volume in the given format."""
    volumes = manifest["datasets"][0]["volumes"]
    return next(volume for volume in volumes if volume["format"] == volume_format)


@pytest.fixture
def manifest():
    """A manifest with one dataset, published as both TIFF and OME-Zarr."""
    return {
        "datasets": [
            {
                "id": "coral",
                "title": "Coral",
                "categories": ["animal"],
                "volumes": [
                    {"format": "zarr", "url": ZARR_URL, "size_bytes": 8192},
                    {"format": "tiff", "url": TIFF_URL, "size_bytes": 4096},
                ],
            }
        ],
    }


@pytest.fixture(autouse=True)
def urlopen(monkeypatch, manifest):
    """
    Serve the manifest and fail any other request.

    The only other requests are file size lookups, so sizes are never reported.
    """

    def serve_manifest(url, timeout=None):
        if url != MANIFEST_URL:
            raise URLError("size unavailable")
        return io.BytesIO(json.dumps(manifest).encode())

    mock = Mock(side_effect=serve_manifest)
    monkeypatch.setattr(urllib.request, "urlopen", mock)
    return mock


@pytest.fixture(autouse=True)
def urlretrieve(monkeypatch):
    """Write a placeholder file instead of downloading one."""

    def write_file(url, destination, reporthook=None):
        Path(destination).write_bytes(b"scan")

    mock = Mock(side_effect=write_file)
    monkeypatch.setattr(urllib.request, "urlretrieve", mock)
    return mock


@pytest.fixture(autouse=True)
def download_zarr(monkeypatch):
    """Create an empty `coral.zarr` folder instead of downloading the store."""

    def create_store(url, output_dir):
        (Path(output_dir) / "coral.zarr").mkdir()

    monkeypatch.setattr("qim3d.io._downloader.download", create_store)


# The downloaded placeholders hold no image data, so tests that load a dataset
# replace the loader and check what it was asked to load.


@pytest.fixture
def load(monkeypatch):
    """Replace the loader of single-file volumes, such as TIFF."""
    mock = Mock()
    monkeypatch.setattr("qim3d.io._downloader.load", mock)
    return mock


@pytest.fixture
def import_ome_zarr(monkeypatch):
    """Replace the loader of OME-Zarr stores."""
    mock = Mock()
    monkeypatch.setattr("qim3d.io.import_ome_zarr", mock)
    return mock


@pytest.fixture
def downloader():
    return Downloader(manifest_url=MANIFEST_URL)


# --- Creating a Downloader ---


@pytest.mark.parametrize("timeout", [0, -1])
def test_non_positive_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="timeout must be positive"):
        Downloader(timeout=timeout)


# --- Listing datasets ---


def test_get_datasets_returns_manifest_datasets(downloader, manifest):
    assert downloader.get_datasets() == manifest["datasets"]


def test_manifest_is_fetched_with_configured_timeout(urlopen):
    Downloader(manifest_url=MANIFEST_URL, timeout=3).get_datasets()

    urlopen.assert_called_once_with(MANIFEST_URL, timeout=3)


def test_manifest_is_fetched_once_and_cached(downloader, urlopen):
    downloader.get_datasets()
    downloader.get_datasets()

    assert urlopen.call_count == 1


def test_get_datasets_returns_a_copy(downloader, manifest):
    downloader.get_datasets()[0]["volumes"].clear()

    assert downloader.get_datasets() == manifest["datasets"]


def test_refresh_fetches_updated_manifest(downloader, manifest):
    downloader.get_datasets()
    manifest["datasets"][0]["title"] = "Brain coral"

    downloader.refresh()

    assert downloader.get_datasets()[0]["title"] == "Brain coral"


def test_failed_refresh_keeps_previous_catalog(downloader, urlopen, manifest):
    downloader.get_datasets()
    urlopen.side_effect = URLError("offline")

    with pytest.raises(URLError):
        downloader.refresh()

    assert downloader.get_datasets() == manifest["datasets"]


@pytest.mark.parametrize("body", [b"{}", b"[]", b'{"datasets": {}}'])
def test_manifest_without_dataset_list_is_rejected(downloader, urlopen, body):
    urlopen.side_effect = [io.BytesIO(body)]

    with pytest.raises(ValueError, match="no list of datasets"):
        downloader.get_datasets()


def test_show_datasets_prints_id_categories_and_sizes(downloader, manifest, capsys):
    del published_volume(manifest, "zarr")["size_bytes"]

    downloader.show_datasets()

    assert capsys.readouterr().out.splitlines() == [
        "ID     Categories  TIFF    Zarr",
        "coral  animal      4.0 KB  -",
    ]


# --- Finding a volume ---


def test_unknown_dataset_raises_lookup_error(downloader, tmp_path):
    with pytest.raises(LookupError, match="'unknown' was not found"):
        downloader.download_dataset("unknown", format="tiff", output_dir=tmp_path)


def test_unpublished_format_raises_lookup_error(downloader, tmp_path):
    with pytest.raises(LookupError, match="no 'nifti' volume"):
        downloader.download_dataset("coral", format="nifti", output_dir=tmp_path)


@pytest.mark.parametrize(
    "url", [None, "", "file:///data/coral.tif", "https://example.org/"]
)
def test_unusable_volume_url_is_rejected(downloader, manifest, tmp_path, url):
    published_volume(manifest, "tiff")["url"] = url

    with pytest.raises(ValueError, match="no usable download URL"):
        downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)

    assert not any(tmp_path.iterdir())


# --- Dataset ID safety ---


@pytest.mark.parametrize("dataset_id", ["", ".", "..", "../coral", None, 123])
def test_unsafe_dataset_id_is_rejected_before_any_request(
    downloader, urlopen, tmp_path, dataset_id
):
    with pytest.raises(ValueError, match="Invalid dataset ID"):
        downloader.download_dataset(dataset_id, format="tiff", output_dir=tmp_path)

    urlopen.assert_not_called()
    assert not any(tmp_path.iterdir())


def test_absolute_path_as_dataset_id_is_rejected(downloader, tmp_path):
    outside = tmp_path.parent / "outside"

    with pytest.raises(ValueError, match="Invalid dataset ID"):
        downloader.download_dataset(str(outside), format="tiff", output_dir=tmp_path)


# --- Downloading ---


@pytest.mark.parametrize(
    ("volume_format", "filename"), [("tiff", "coral.tif"), ("zarr", "coral.zarr")]
)
def test_volume_is_saved_alone_in_dataset_folder(
    downloader, tmp_path, volume_format, filename
):
    path = downloader.download_dataset(
        "coral", format=volume_format, output_dir=tmp_path
    )

    assert path == tmp_path / "coral" / filename
    assert list(path.parent.iterdir()) == [path]


def test_existing_download_is_reused(downloader, urlretrieve, tmp_path):
    first = downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)
    second = downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)

    assert second == first
    assert urlretrieve.call_count == 1


def test_interrupted_download_leaves_no_partial_file(downloader, urlretrieve, tmp_path):
    def interrupt(url, destination, reporthook=None):
        Path(destination).write_bytes(b"partial")
        raise ConnectionError("interrupted")

    urlretrieve.side_effect = interrupt

    with pytest.raises(ConnectionError):
        downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)

    assert not any((tmp_path / "coral").iterdir())


def test_published_size_avoids_asking_the_server(downloader, urlopen, tmp_path):
    downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)

    urlopen.assert_called_once_with(MANIFEST_URL, timeout=ANY)


@pytest.mark.parametrize("size_bytes", [None, 0, -1, "12"])
def test_unusable_published_size_is_asked_from_the_server(
    downloader, urlopen, manifest, tmp_path, size_bytes
):
    published_volume(manifest, "tiff")["size_bytes"] = size_bytes

    downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)

    urlopen.assert_called_with(TIFF_URL, timeout=ANY)


# --- Loading ---


@pytest.mark.parametrize("virtual_stack", [True, False])
def test_tiff_is_loaded_from_downloaded_file(downloader, load, tmp_path, virtual_stack):
    loaded = downloader.load_dataset(
        "coral", format="tiff", output_dir=tmp_path, virtual_stack=virtual_stack
    )

    load.assert_called_once_with(
        path=tmp_path / "coral" / "coral.tif", virtual_stack=virtual_stack
    )
    assert loaded is load.return_value


@pytest.mark.parametrize(
    ("scale", "virtual_stack"), [(0, True), ("lowest", True), (0, False)]
)
def test_zarr_is_loaded_from_downloaded_store(
    downloader, import_ome_zarr, tmp_path, scale, virtual_stack
):
    loaded = downloader.load_dataset(
        "coral",
        format="zarr",
        output_dir=tmp_path,
        scale=scale,
        virtual_stack=virtual_stack,
    )

    import_ome_zarr.assert_called_once_with(
        tmp_path / "coral" / "coral.zarr", scale=scale, load=not virtual_stack
    )
    assert loaded is import_ome_zarr.return_value


def test_scale_is_rejected_for_non_zarr_formats(downloader, urlopen, tmp_path):
    with pytest.raises(ValueError, match="scale is only supported for OME-Zarr"):
        downloader.load_dataset("coral", format="tiff", output_dir=tmp_path, scale=1)

    urlopen.assert_not_called()
