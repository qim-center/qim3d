import pytest
import matplotlib
from unittest.mock import patch, MagicMock

from qim3d.tests import get_all_functions_by_module, check_docstring

matplotlib.use('Agg')

# Get dictionary of functions by module
functions_by_module = get_all_functions_by_module()

# Mock the qim3d.io.Downloader class and its methods
@pytest.fixture
def mock_downloader():
    with patch("qim3d.io.Downloader") as MockDownloader:

        # Create a mock instance of Downloader
        mock_instance = MockDownloader.return_value

        # Mock the Snail.Escargot method to return a fake dataset
        mock_instance.Snail.Escargot.return_value = MagicMock(name="FakeDataset")

        yield MockDownloader

# Mock the qim3d.io load and save functions
@pytest.fixture
def mock_io_functions():

    # List of functions to mock
    functions_to_mock = [
        "qim3d.io.load",
        "qim3d.io.save",
        "qim3d.io.load_mesh",
        "qim3d.io.save_mesh",
    ]

    patches = []
    for func_path in functions_to_mock:

        # Mock the function to return a fake dataset 
        patcher = patch(func_path, return_value=MagicMock(name=f"Mocked_{func_path.split('.')[-1]}"))
        patches.append(patcher)
        patcher.start()

    # Provide the mocks to the unit test
    yield

    # Stop all patches after the test
    for patcher in patches:
        patcher.stop()

# Mock the qim3d.io OME-Zarr functions
@pytest.fixture
def mock_ome_zarr_functions(tmp_path):
    # Temporary directory for mock files
    mock_file_path = tmp_path / "Escargot.zarr"

    functions_to_mock = [
        "qim3d.io.import_ome_zarr",
        "qim3d.io.export_ome_zarr",
    ]

    patches = []
    for func_path in functions_to_mock:
        if func_path == "qim3d.io.export_ome_zarr":

            # Mock export_ome_zarr to simulate creating a file
            patcher = patch(func_path, side_effect=lambda *args, **kwargs: mock_file_path.touch())
        else:

            # Mock import_ome_zarr to return a fake dataset
            patcher = patch(func_path, return_value=MagicMock(name=f"Mocked_{func_path.split('.')[-1]}"))
        patches.append(patcher)
        patcher.start()

    # Provide the mocks to the unit test
    yield mock_file_path

    # Stop all patches after the test
    for patcher in patches:
        patcher.stop()

# Mock the qim3d.viz.chunks function
@pytest.fixture
def mock_viz_chunks():
    with patch("qim3d.viz.chunks") as MockVizChunks:

        # Mock the function to do nothing
        MockVizChunks.return_value = None

        yield MockVizChunks

@pytest.mark.parametrize('func', functions_by_module["io"], ids=lambda d: d.__name__)
def test_docstrings_io(func, mock_downloader, mock_io_functions, mock_ome_zarr_functions):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["generate"], ids=lambda d: d.__name__)
def test_docstrings_generate(func):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["viz"], ids=lambda d: d.__name__)
def test_docstrings_viz(func, mock_downloader, mock_ome_zarr_functions, mock_viz_chunks):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["mesh"], ids=lambda d: d.__name__)
def test_docstrings_mesh(func):
    check_docstring(obj=func)