import pytest
import matplotlib
from unittest.mock import patch, MagicMock

from qim3d.tests import get_all_functions_by_module, check_docstring, temp_data

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

@pytest.mark.parametrize('func', functions_by_module["features"], ids=lambda d: d.__name__)
def test_docstrings_features(func):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["filters"], ids=lambda d: d.__name__)
def test_docstrings_filters(func):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["detection"], ids=lambda d: d.__name__)
def test_docstrings_detection(func):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["segmentation"], ids=lambda d: d.__name__)
def test_docstrings_segmentation(func):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["operations"], ids=lambda d: d.__name__)
def test_docstrings_operations(func):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["processing"], ids=lambda d: d.__name__)
def test_docstrings_processing(func):

    # Exclude qim3d.processing.segment_layers function, since it uses unavailable example data
    if func.__name__ == "segment_layers":
        return
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["mesh"], ids=lambda d: d.__name__)
def test_docstrings_mesh(func):
    check_docstring(obj=func)

@pytest.mark.parametrize('func', functions_by_module["ml"], ids=lambda d: d.__name__)
def test_docstrings_ml(func):

    # Exclude train_model, load_checkpoint, and test_model functions
    if func.__name__ in ["train_model", "load_checkpoint", "test_model"]:
        return

    temp_data(folder='dataset', img_shape=(32, 32, 32), n=5)
    check_docstring(obj=func)
    temp_data(folder='dataset', remove=True)
