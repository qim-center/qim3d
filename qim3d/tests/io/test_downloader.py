"""Offline tests for manifest-backed dataset discovery and downloading."""

import io
import json
import urllib.request
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

import qim3d
from qim3d.io import _downloader


@pytest.fixture(autouse=True)
def block_external_requests(monkeypatch):
    """Any unmocked network call is a test failure, not a flaky integration test."""

    def blocked(*args, **kwargs):
        raise AssertionError("Unexpected external request in downloader test")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(urllib.request, "urlretrieve", blocked)
    monkeypatch.setattr(_downloader, "download", blocked)


@pytest.fixture
def manifest():
    return {
        "schema_version": 1,
        "datasets": [
            {
                "id": "coral",
                "title": "Coral",
                "summary": "A sample scan",
                "description": "A coral volume",
                "categories": ["biological"],
                "license": None,
                "contributors": [],
                "page_url": "/datasets/coral/",
                "volumes": [
                    {"format": "zarr", "url": "https://example.org/coral.zarr"},
                    {"format": "tiff", "url": "https://example.org/coral.tif"},
                ],
            },
            {
                "id": "oak-branch",
                "title": "Oak Branch",
                "volumes": [{"format": "zarr", "url": None}],
            },
        ],
    }


def mock_manifest_response(monkeypatch, manifest):
    fetch = Mock(
        side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(manifest).encode())
    )
    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    return fetch


def test_manifest_is_lazy_cached_and_refreshable(monkeypatch, manifest):
    fetch = mock_manifest_response(monkeypatch, manifest)
    downloader = qim3d.io.Downloader(manifest_url="https://example.org/index.json")

    assert fetch.call_count == 0
    datasets = downloader.list_datasets()
    assert [dataset["id"] for dataset in datasets] == ["coral", "oak-branch"]
    assert datasets[0]["summary"] == "A sample scan"
    assert datasets[1]["volumes"][0]["url"] is None
    datasets[0]["volumes"][0]["url"] = "changed"
    assert downloader.list_datasets()[0]["volumes"][0]["url"] == (
        "https://example.org/coral.zarr"
    )
    assert fetch.call_count == 1
    fetch.assert_called_with("https://example.org/index.json", timeout=10)

    downloader.refresh()
    assert fetch.call_count == 2

    updated = {**manifest, "datasets": manifest["datasets"][1:]}
    mock_manifest_response(monkeypatch, updated)
    downloader.refresh()
    assert [dataset["id"] for dataset in downloader.list_datasets()] == ["oak-branch"]


def test_invalid_timeout_does_not_fetch_manifest():
    with pytest.raises(ValueError, match="timeout"):
        qim3d.io.Downloader(timeout=0)


def test_refresh_failure_preserves_cached_catalog(monkeypatch, manifest):
    mock_manifest_response(monkeypatch, manifest)
    downloader = qim3d.io.Downloader()
    assert len(downloader.list_datasets()) == 2

    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(qim3d.io.ManifestError, match="Could not load dataset manifest"):
        downloader.refresh()
    assert len(downloader.list_datasets()) == 2


@pytest.mark.parametrize("bad_manifest", [{}, []])
def test_manifest_needs_datasets_key(monkeypatch, bad_manifest):
    mock_manifest_response(monkeypatch, bad_manifest)
    with pytest.raises(qim3d.io.ManifestError, match="Could not load dataset manifest"):
        qim3d.io.Downloader().list_datasets()


def test_manifest_does_not_validate_datasets_shape(monkeypatch):
    mock_manifest_response(monkeypatch, {"datasets": {}})
    assert qim3d.io.Downloader().list_datasets() == {}


def test_manifest_version_is_ignored(monkeypatch, manifest):
    manifest["schema_version"] = 2
    mock_manifest_response(monkeypatch, manifest)
    assert len(qim3d.io.Downloader().list_datasets()) == 2

    del manifest["schema_version"]
    mock_manifest_response(monkeypatch, manifest)
    assert len(qim3d.io.Downloader().list_datasets()) == 2


def test_unrelated_bad_metadata_does_not_block_listing_or_download(
    monkeypatch, manifest, tmp_path
):
    manifest["datasets"].append(
        {"id": "unrelated", "title": "", "volumes": [{"format": "zarr", "url": ""}]}
    )
    mock_manifest_response(monkeypatch, manifest)
    downloader = qim3d.io.Downloader()
    assert len(downloader.list_datasets()) == 3

    existing = tmp_path / "coral" / "coral.tif"
    existing.parent.mkdir()
    existing.write_bytes(b"cached")
    assert (
        downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)
        == existing
    )


@pytest.mark.parametrize(
    ("datasets", "error"),
    [
        ([{"id": "coral", "volumes": "bad"}], qim3d.io.VolumeNotFoundError),
        (
            [
                {
                    "id": "coral",
                    "volumes": [{"format": "zarr", "url": "file:///tmp/a.zarr"}],
                }
            ],
            qim3d.io.VolumeUnavailableError,
        ),
        (
            [{"id": "coral", "volumes": [{"format": "zarr", "url": ""}]}],
            qim3d.io.VolumeUnavailableError,
        ),
        (
            [{"id": "coral", "volumes": [{"format": "zarr", "url": 123}]}],
            qim3d.io.VolumeUnavailableError,
        ),
        (
            [
                {
                    "id": "coral",
                    "volumes": [{"format": "zarr", "url": "https://example.org/"}],
                }
            ],
            qim3d.io.VolumeUnavailableError,
        ),
    ],
)
def test_selected_volume_is_checked_when_downloaded(
    monkeypatch, datasets, error, tmp_path
):
    mock_manifest_response(monkeypatch, {"schema_version": 1, "datasets": datasets})
    with pytest.raises(error):
        qim3d.io.Downloader().download_dataset(
            "coral", format="zarr", output_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "dataset_id",
    ["", ".", "..", "../coral", "coral/../../outside", "coral\x00", None, 123],
)
def test_unsafe_dataset_id_is_rejected_before_accessing_manifest(dataset_id, tmp_path):
    with pytest.raises(ValueError, match="dataset ID"):
        qim3d.io.Downloader().download_dataset(
            dataset_id, format="zarr", output_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_absolute_dataset_id_is_rejected_before_accessing_manifest(tmp_path):
    with pytest.raises(ValueError, match="dataset ID"):
        qim3d.io.Downloader().download_dataset(
            str(tmp_path.parent / "outside"), format="zarr", output_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_dataset_id_cannot_follow_symlink_outside_output_dir(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (downloads / "linked").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable")

    with pytest.raises(ValueError, match="dataset ID"):
        qim3d.io.Downloader().download_dataset(
            "linked", format="zarr", output_dir=downloads
        )
    assert not list(outside.iterdir())


@pytest.mark.parametrize("dataset_id", ["Cowry_Shell", "Coral 2", "føram", "coral/2"])
def test_dataset_ids_are_not_limited_to_slug_format(
    monkeypatch, manifest, tmp_path, dataset_id
):
    manifest["datasets"][0]["id"] = dataset_id
    mock_manifest_response(monkeypatch, manifest)
    existing = tmp_path / dataset_id / "coral.tif"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"cached")

    assert (
        qim3d.io.Downloader().download_dataset(
            dataset_id, format="tiff", output_dir=tmp_path
        )
        == existing
    )


def test_manifest_fetch_and_json_errors(monkeypatch):
    def unavailable(*args, **kwargs):
        raise TimeoutError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(qim3d.io.ManifestError, match="Could not load dataset manifest"):
        qim3d.io.Downloader().list_datasets()

    for payload in (b"not JSON", b"\xff"):
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *args, **kwargs: io.BytesIO(payload),
        )
        with pytest.raises(
            qim3d.io.ManifestError, match="Could not load dataset manifest"
        ):
            qim3d.io.Downloader().list_datasets()


def test_dataset_lookup_errors(monkeypatch, manifest, tmp_path):
    mock_manifest_response(monkeypatch, manifest)
    downloader = qim3d.io.Downloader()

    with pytest.raises(qim3d.io.DatasetNotFoundError, match="unknown"):
        downloader.download_dataset("unknown", format="zarr", output_dir=tmp_path)
    with pytest.raises(qim3d.io.VolumeNotFoundError, match="tiff"):
        downloader.download_dataset("oak-branch", format="tiff", output_dir=tmp_path)
    with pytest.raises(qim3d.io.VolumeUnavailableError, match="oak-branch"):
        downloader.download_dataset("oak-branch", format="zarr", output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_dataset_stages_and_reuses_file(monkeypatch, manifest, tmp_path):
    mock_manifest_response(monkeypatch, manifest)
    downloader = qim3d.io.Downloader()
    calls = []

    def fake_retrieve(url, destination, reporthook):
        calls.append(url)
        Path(destination).write_bytes(b"scan")

    monkeypatch.setattr(_downloader, "_get_file_size", lambda url: 4)
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    path = downloader.download_dataset("coral", format="tiff", output_dir=tmp_path)

    assert isinstance(path, Path)
    assert path == tmp_path / "coral" / "coral.tif"
    assert path.read_bytes() == b"scan"
    assert list(path.parent.iterdir()) == [path]
    assert (
        downloader.download_dataset("coral", format="tiff", output_dir=tmp_path) == path
    )
    assert calls == ["https://example.org/coral.tif"]


def test_download_dataset_stages_zarr_directory(monkeypatch, manifest, tmp_path):
    mock_manifest_response(monkeypatch, manifest)

    def fake_zarr_download(url, output_dir):
        target = Path(output_dir) / "coral.zarr"
        target.mkdir()
        (target / "zarr.json").write_text("{}")

    monkeypatch.setattr(_downloader, "download", fake_zarr_download)
    downloader = qim3d.io.Downloader()
    path = downloader.download_dataset("coral", format="zarr", output_dir=tmp_path)
    assert path == tmp_path / "coral" / "coral.zarr"
    assert (path / "zarr.json").exists()
    assert list(path.parent.iterdir()) == [path]


def test_failed_download_does_not_leave_final_path(monkeypatch, manifest, tmp_path):
    mock_manifest_response(monkeypatch, manifest)

    def interrupted_download(self, url, output_dir="."):
        (Path(output_dir) / "coral.tif").write_bytes(b"partial")
        raise ConnectionError("interrupted")

    monkeypatch.setattr(_downloader.Downloader, "_download_url", interrupted_download)
    with pytest.raises(ConnectionError, match="interrupted"):
        qim3d.io.Downloader().download_dataset(
            "coral", format="tiff", output_dir=tmp_path
        )
    assert list((tmp_path / "coral").iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://example.org/coral.tif", 404, "Not Found", None, None),
        URLError("offline"),
    ],
)
def test_download_network_error_is_not_rewritten(
    monkeypatch, manifest, tmp_path, error
):
    mock_manifest_response(monkeypatch, manifest)
    monkeypatch.setattr(_downloader, "_get_file_size", lambda url: 4)
    monkeypatch.setattr(urllib.request, "urlretrieve", Mock(side_effect=error))

    with pytest.raises(type(error)):
        qim3d.io.Downloader().download_dataset(
            "coral", format="tiff", output_dir=tmp_path
        )
    assert not (tmp_path / "coral" / "coral.tif").exists()


def test_missing_staged_file_raises(monkeypatch, manifest, tmp_path):
    mock_manifest_response(monkeypatch, manifest)
    monkeypatch.setattr(
        _downloader.Downloader,
        "_download_url",
        lambda self, url, output_dir: Path(output_dir) / "coral.tif",
    )
    with pytest.raises(FileNotFoundError):
        qim3d.io.Downloader().download_dataset(
            "coral", format="tiff", output_dir=tmp_path
        )
    assert not (tmp_path / "coral" / "coral.tif").exists()


def test_load_dataset_uses_format_specific_loader(monkeypatch, tmp_path):
    downloader = qim3d.io.Downloader()
    zarr = tmp_path / "coral.zarr"
    tiff = tmp_path / "coral.tif"
    download_dataset = Mock(side_effect=[zarr, zarr, tiff])
    import_zarr = Mock(side_effect=["lazy zarr", "eager zarr"])
    load_tiff = Mock(return_value="lazy tiff")
    monkeypatch.setattr(downloader, "download_dataset", download_dataset)
    monkeypatch.setattr(qim3d.io, "import_ome_zarr", import_zarr)
    monkeypatch.setattr(_downloader, "load", load_tiff)

    assert (
        downloader.load_dataset(
            "coral", format="zarr", output_dir=tmp_path, scale="lowest"
        )
        == "lazy zarr"
    )
    import_zarr.assert_called_with(zarr, scale="lowest", load=False)
    assert (
        downloader.load_dataset(
            "coral", format="zarr", output_dir=tmp_path, virtual_stack=False
        )
        == "eager zarr"
    )
    import_zarr.assert_called_with(zarr, scale=0, load=True)
    assert (
        downloader.load_dataset("coral", format="tiff", output_dir=tmp_path)
        == "lazy tiff"
    )
    load_tiff.assert_called_once_with(path=tiff, virtual_stack=True)
    assert download_dataset.call_count == 3
    assert all(
        call.kwargs["output_dir"] == tmp_path
        for call in download_dataset.call_args_list
    )

    with pytest.raises(ValueError, match="scale"):
        downloader.load_dataset("coral", format="tiff", scale=1)
    assert download_dataset.call_count == 3


def test_old_dynamic_api_is_not_present():
    downloader = qim3d.io.Downloader()
    assert not callable(downloader)
    assert not hasattr(downloader, "list_files")
    assert not hasattr(downloader, "Cowry_Shell")
