import shutil
from pathlib import Path

import pytest

import qim3d


@pytest.fixture()
def setup_temp_folder():
    """Fixture to create and clean up a temporary folder for tests."""
    folder = 'Cowry_Shell'
    file = 'Cowry_DOWNSAMPLED.tif'
    path = Path(folder) / file

    # Ensure clean environment before running tests
    if Path(folder).exists():
        shutil.rmtree(folder)
    yield folder, path

    # Cleanup after tests
    if path.exists():
        path.unlink()
    if Path(folder).exists():
        shutil.rmtree(folder)


def test_download(setup_temp_folder):
    folder, path = setup_temp_folder

    dl = qim3d.io.Downloader()
    dl.Cowry_Shell.Cowry_DOWNSAMPLED()

    # Verify the file was downloaded correctly
    assert path.exists(), f'{path} does not exist after download.'

    img = qim3d.io.load(str(path))
    assert img.shape == (500, 350, 350)

    # Cleanup is handled by the fixture


def test_get_file_size():
    """Tests for correct and incorrect file size retrieval."""
    coal_file = 'https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository/Coal/CoalBrikett.tif'
    folder_url = (
        'https://archive.compute.dtu.dk/files/public/projects/viscomp_data_repository/'
    )

    # Correct file size
    size = qim3d.io._downloader._get_file_size(coal_file)
    assert size == 2_400_082_900, f'Expected size mismatch for {coal_file}.'

    # Wrong URL (not a file)
    size = qim3d.io._downloader._get_file_size(folder_url)
    assert size == -1, 'Expected size -1 for non-file URL.'


def test_extract_html():
    url = 'https://archive.compute.dtu.dk/files/public/projects/viscomp_data_repository'
    html = qim3d.io._downloader._extract_html(url)

    assert (
        'data-path="/files/public/projects/viscomp_data_repository"' in html
    ), 'Expected HTML content not found in extracted HTML.'
