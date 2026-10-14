import importlib
import inspect
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import matplotlib
import pytest
from mktestdocs import grab_code_blocks

matplotlib.use("Agg")


def get_all_functions_by_module():
    """
    Creates and returns a dictionary of functions from the qim3d modules.
    """

    # List of qim3d modules
    # TODO: Get this list automatically from the qim3d package information
    modules = [
        "io",
        "generate",
        "viz",
        "features",
        "filters",
        "detection",
        "segmentation",
        "operations",
        "processing",
        "mesh",
        "ml",
    ]

    # Dictionary to store functions from each module
    functions_by_module = {}

    for module_name in modules:
        # Dynamically import the module
        module = importlib.import_module(f"qim3d.{module_name}")

        # Retrieve all functions listed in the __all__ variable
        functions = [
            getattr(module, name)
            for name in getattr(module, "__all__", [])
            if callable(getattr(module, name))
        ]

        # Only keep functions (not classes) in the list
        functions = [func for func in functions if inspect.isfunction(func)]

        # Store the functions in the dictionary
        functions_by_module[module_name] = functions

    return functions_by_module


def exec_python(source):
    """
    Execute a Python code block.
    """
    try:
        exec(source, {"__name__": "__main__"})
    except Exception:
        print(source)
        raise


def merge_code_blocks(code_blocks):
    """
    Merge code blocks such that every time there is an 'import qim3d',
    a new code block starts, and all preceding code blocks are merged into the previous one.
    """
    merged_blocks = []
    current_block = ""

    for block in code_blocks:
        if "import qim3d" in block:
            # If there's an existing block, add it to the merged list
            if current_block.strip():
                merged_blocks.append(current_block.strip())

            # Start a new block
            current_block = block
        else:
            # Append to the current block
            current_block += "\n" + block

    # Add the last block if it exists
    if current_block.strip():
        merged_blocks.append(current_block.strip())

    return merged_blocks


def filter_code_blocks(code_blocks):
    """Filter out code blocks that contain bibtex references."""

    filtered_blocks = []

    for block in code_blocks:
        if not any(line.startswith("@") for line in block.splitlines()):
            filtered_blocks.append(block)

    return filtered_blocks


def check_docstring(obj):
    """
    Given a function, test the contents of the docstring.
    Custom function inspired by the mktestdocs.check_docstring function.
    """

    # Get all code blocks from the docstring
    code_blocks = grab_code_blocks(obj.__doc__, lang="")

    # Filter out code blocks that contain bibtex references
    code_blocks = filter_code_blocks(code_blocks)

    # Merge code blocks based on 'import qim3d'
    merged_code_blocks = merge_code_blocks(code_blocks)

    # Execute each merged code block
    for b in merged_code_blocks:
        exec_python(b)


# Get dictionary of functions by module
functions_by_module = get_all_functions_by_module()


def noop_mock():
    return MagicMock()


# Per-(module, function) mocks to apply while a docstring's code blocks execute.
# Each target maps a dotted import path to a callable that should return a mock object.
# Note that the target depends on how it is imported in the module.
# (see viz -> export_rotation below as example)
MOCK_TARGETS: dict[str, dict[str, dict[str, Callable[[], object]]]] = {
    "viz": {
        "chunks": {
            "qim3d.viz.chunks": noop_mock,
        },
        # screenshot() segfaults without a graphics environment, so it is mocked out.
        # It returns MagicMock objects, which okay since the mimsave method is also mocked.
        # See: https://github.com/qim-center/qim3d/issues/249
        "export_rotation": {
            "imageio.v2.mimsave": noop_mock,
            "imageio.v2.get_writer": noop_mock,
            "qim3d.viz._data_exploration.Image": noop_mock,
            "qim3d.viz._data_exploration.display": noop_mock,
            "pyvista.Plotter.screenshot": noop_mock,
        },
        "mesh": {
            "pyvista.Plotter": noop_mock,
        },
        "streamlines": {
            "pyvista.Plotter": noop_mock,
        },
    },
    "mesh": {
        "from_volume": {
            "pyvista.Plotter": noop_mock,
        },
    },
}


@pytest.fixture(autouse=True)
def _apply_mock_targets(request):
    # request.node.originalname is the test function name without the
    # parametrize id suffix, e.g. "test_docstrings_processing".
    module_name = request.node.originalname.removeprefix("test_docstrings_")
    func = request.node.callspec.params.get("func")
    targets = MOCK_TARGETS.get(module_name, {}).get(func.__name__, {})

    patches = [patch(path, new=factory()) for path, factory in targets.items()]
    for p in patches:
        p.start()

    yield

    for p in patches:
        p.stop()


# Mock the qim3d.io.Downloader class and its methods
@pytest.fixture(autouse=True)
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
        patcher = patch(
            func_path, return_value=MagicMock(name=f"Mocked_{func_path.split('.')[-1]}")
        )
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
            patcher = patch(
                func_path, side_effect=lambda *args, **kwargs: mock_file_path.touch()
            )
        else:
            # Mock import_ome_zarr to return a fake dataset
            patcher = patch(
                func_path,
                return_value=MagicMock(name=f"Mocked_{func_path.split('.')[-1]}"),
            )
        patches.append(patcher)
        patcher.start()

    # Provide the mocks to the unit test
    yield mock_file_path

    # Stop all patches after the test
    for patcher in patches:
        patcher.stop()


@pytest.mark.parametrize("func", functions_by_module["io"], ids=lambda d: d.__name__)
def test_docstrings_io(func, mock_io_functions, mock_ome_zarr_functions):
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "func", functions_by_module["generate"], ids=lambda d: d.__name__
)
def test_docstrings_generate(func):
    check_docstring(obj=func)


@pytest.mark.parametrize("func", functions_by_module["viz"], ids=lambda d: d.__name__)
def test_docstrings_viz(func):
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "func", functions_by_module["features"], ids=lambda d: d.__name__
)
def test_docstrings_features(func):
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "func", functions_by_module["filters"], ids=lambda d: d.__name__
)
def test_docstrings_filters(func):
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "func", functions_by_module["detection"], ids=lambda d: d.__name__
)
def test_docstrings_detection(func):
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "func", functions_by_module["segmentation"], ids=lambda d: d.__name__
)
def test_docstrings_segmentation(func):
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "func", functions_by_module["operations"], ids=lambda d: d.__name__
)
def test_docstrings_operations(func):
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "func", functions_by_module["processing"], ids=lambda d: d.__name__
)
def test_docstrings_processing(func):
    # Exclude qim3d.processing.segment_layers function, since it uses unavailable example data
    if func.__name__ == "segment_layers":
        return
    check_docstring(obj=func)


@pytest.mark.parametrize("func", functions_by_module["mesh"], ids=lambda d: d.__name__)
def test_docstrings_mesh(func):
    check_docstring(obj=func)


@pytest.mark.parametrize("func", functions_by_module["ml"], ids=lambda d: d.__name__)
def test_docstrings_ml(func, make_dataset, tmp_path, monkeypatch):
    # Exclude train_model, load_checkpoint, and test_model functions
    if func.__name__ in ["train_model", "load_checkpoint", "test_model"]:
        return

    # The docstring examples load from the relative path "dataset"
    monkeypatch.chdir(tmp_path)
    make_dataset(folder=tmp_path / "dataset", img_shape=(32, 32, 32), n=5)
    check_docstring(obj=func)
